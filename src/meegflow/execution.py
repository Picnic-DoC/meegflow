"""Execution backends for dispatching one pipeline job per recording.

MEEGFlow's job manager (``MEEGFlowPipeline.run_pipeline``) discovers the list
of recordings to process, then dispatches one self-contained job per
recording through the backend selected here:

- ``sequential`` (default): today's single-process, in-order loop. Requires
  no new dependency.
- ``local``: an in-process Dask cluster (``distributed.LocalCluster``),
  comparable to ``concurrent.futures.ProcessPoolExecutor`` or ``joblib``.
- ``slurm`` / ``pbs`` / ``sge`` / ``lsf`` / ``htcondor``: a ``dask-jobqueue``
  cluster, submitting one Dask worker job per HPC scheduler job.

``dask`` and ``dask-jobqueue`` are optional dependencies (see
``extras_require`` in ``setup.py``) and are only imported lazily, inside the
functions that need them, so importing this module (and therefore
``meegflow.pipeline``) never requires Dask to be installed unless a
non-sequential backend is actually requested.

See ``docs/dask_parallel_execution.md`` for the full design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from mne.utils import logger

if TYPE_CHECKING:
    from .readers import DatasetReader

SEQUENTIAL_BACKEND = "sequential"
LOCAL_BACKEND = "local"

# Maps a jobqueue backend name (as used in the 'execution.backend' config
# key) to the corresponding dask_jobqueue cluster class name.
JOBQUEUE_CLUSTERS = {
    "slurm": "SLURMCluster",
    "pbs": "PBSCluster",
    "sge": "SGECluster",
    "lsf": "LSFCluster",
    "htcondor": "HTCondorCluster",
}

KNOWN_BACKENDS = [SEQUENTIAL_BACKEND, LOCAL_BACKEND] + list(JOBQUEUE_CLUSTERS)


@dataclass
class ExecutionConfig:
    """Execution backend selection, parsed from the ``execution`` config block.

    Parameters
    ----------
    backend : str
        One of ``'sequential'`` (default), ``'local'``, or a
        ``dask-jobqueue`` cluster type (``'slurm'``, ``'pbs'``, ``'sge'``,
        ``'lsf'``, ``'htcondor'``).
    n_workers : int
        Number of parallel workers (local processes, or cluster jobs).
        Ignored for the ``sequential`` backend.
    cluster_kwargs : dict
        Extra keyword arguments forwarded verbatim to the underlying Dask
        cluster constructor (e.g. ``queue``, ``cores``, ``memory``,
        ``walltime`` for ``dask-jobqueue`` backends).
    """

    backend: str = SEQUENTIAL_BACKEND
    n_workers: int = 1
    cluster_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "ExecutionConfig":
        """Parse the ``execution`` block of a pipeline config, if present.

        Parameters
        ----------
        config : dict or None
            The full pipeline configuration dictionary. Recognizes a top-level
            ``execution`` mapping with keys ``backend``, ``n_workers``, and
            ``cluster_kwargs``. Missing or absent -> sequential execution.

        Returns
        -------
        ExecutionConfig
        """
        exec_cfg = (config or {}).get("execution") or {}
        backend = exec_cfg.get("backend", SEQUENTIAL_BACKEND)
        if backend not in KNOWN_BACKENDS:
            raise ValueError(
                f"Unknown execution backend '{backend}'. Choose from: {KNOWN_BACKENDS}."
            )
        return cls(
            backend=backend,
            n_workers=int(exec_cfg.get("n_workers", 1)),
            cluster_kwargs=dict(exec_cfg.get("cluster_kwargs", {}) or {}),
        )


def _run_recording_job(
    reader: "DatasetReader",
    output_root: Optional[Union[str, Path]],
    config: Dict[str, Any],
    paths: List[Any],
    metadata: Dict[str, Any],
    io_backend: str,
) -> Dict[str, Any]:
    """Module-level, picklable unit of work: run the full pipeline for one recording.

    This is the function actually submitted to Dask workers. It rebuilds the
    step registry (built-ins from ``STEP_REGISTRY`` plus any custom steps
    declared via ``config['custom_steps_folder']``) independently, in
    whichever process executes it, since functions loaded dynamically via
    ``importlib`` cannot be relied upon to pickle by reference across
    process (or host) boundaries. See ``docs/dask_parallel_execution.md``
    §4.3 for the full rationale.
    """
    # Local import: avoids a module-level circular import between
    # pipeline.py (which imports this module) and execution.py.
    from .pipeline import build_step_functions, process_recording

    step_functions = build_step_functions(config)
    return process_recording(
        reader=reader,
        output_root=output_root,
        config=config,
        step_functions=step_functions,
        paths=paths,
        metadata=metadata,
        io_backend=io_backend,
    )


def _build_cluster(exec_config: ExecutionConfig):
    """Construct the Dask cluster for the requested non-sequential backend."""
    if exec_config.backend == LOCAL_BACKEND:
        try:
            from distributed import LocalCluster
        except ImportError as exc:
            raise ImportError(
                "The 'local' execution backend requires Dask's distributed "
                "scheduler. Install it with: pip install meegflow[dask]"
            ) from exc
        return LocalCluster(n_workers=exec_config.n_workers, **exec_config.cluster_kwargs)

    cluster_cls_name = JOBQUEUE_CLUSTERS.get(exec_config.backend)
    if cluster_cls_name is None:
        raise ValueError(
            f"Unknown execution backend '{exec_config.backend}'. Choose from: {KNOWN_BACKENDS}."
        )

    try:
        import dask_jobqueue
    except ImportError as exc:
        raise ImportError(
            f"The '{exec_config.backend}' execution backend requires "
            "dask-jobqueue. Install it with: pip install meegflow[dask-jobqueue]"
        ) from exc

    cluster_cls = getattr(dask_jobqueue, cluster_cls_name)
    cluster = cluster_cls(**exec_config.cluster_kwargs)
    cluster.scale(jobs=exec_config.n_workers)
    return cluster


def _subject_key(metadata: Dict[str, Any]) -> str:
    """Pick the dict key under which a recording's result is grouped in ``all_results``."""
    return metadata.get("subject", list(metadata.values())[0] if metadata else "unknown")


def run_sequential(
    recordings: List[Dict[str, Any]],
    reader: "DatasetReader",
    output_root: Optional[Union[str, Path]],
    config: Dict[str, Any],
    step_functions: Dict[str, Callable],
    io_backend: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Process recordings one at a time, in this process (today's default behavior).

    Parameters
    ----------
    recordings : list of dict
        Output of ``reader.find_recordings(...)``.
    reader : DatasetReader
        Reader used to load each recording's files.
    output_root : str or Path, optional
        Derivatives root override.
    config : dict
        Full pipeline configuration.
    step_functions : dict
        Mapping of step name -> callable (built-in + custom).
    io_backend : str
        MNE IO backend used to read raw files.

    Returns
    -------
    all_results : dict
        Mapping ``{subject: [result_or_error, ...]}``, matching the shape
        returned by ``run_dask``.
    """
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    from .pipeline import process_recording

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        overall_task = progress.add_task("[green]Processing recordings", total=len(recordings))

        for i, recording in enumerate(recordings):
            paths = recording["paths"]
            metadata = recording["metadata"]
            recording_name = recording["recording_name"]
            subject_key = _subject_key(metadata)

            progress.update(overall_task, description=f"[cyan]{recording_name}")

            try:
                result = process_recording(
                    reader=reader,
                    output_root=output_root,
                    config=config,
                    step_functions=step_functions,
                    paths=paths,
                    metadata=metadata,
                    io_backend=io_backend,
                )
                all_results.setdefault(subject_key, []).append(result)
                logger.info(f"Successfully completed {recording_name}")
            except Exception as exc:
                # Do not stop the whole batch if one recording fails.
                logger.error(f"Error processing {recording_name}: {exc}")
                all_results.setdefault(subject_key, []).append({"error": str(exc)})
            finally:
                progress.update(overall_task, completed=i + 1)

    return all_results


def run_dask(
    recordings: List[Dict[str, Any]],
    reader: "DatasetReader",
    output_root: Optional[Union[str, Path]],
    config: Dict[str, Any],
    io_backend: str,
    exec_config: ExecutionConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """Dispatch one job per recording through a Dask cluster.

    Used for the ``local`` backend and every ``dask-jobqueue`` backend
    (``slurm``, ``pbs``, ``sge``, ``lsf``, ``htcondor``). Preserves the
    sequential backend's contract: one recording's failure is captured as an
    ``{'error': ...}`` entry rather than aborting the batch, and results are
    returned in the same ``{subject: [result, ...]}`` shape as
    :func:`run_sequential`.

    Parameters
    ----------
    recordings : list of dict
        Output of ``reader.find_recordings(...)``.
    reader : DatasetReader
        Reader used to load each recording's files. Must be picklable.
    output_root : str or Path, optional
        Derivatives root override.
    config : dict
        Full pipeline configuration (plain, picklable dict).
    io_backend : str
        MNE IO backend used to read raw files.
    exec_config : ExecutionConfig
        Selects and configures the Dask cluster (backend, worker count,
        cluster-specific kwargs).

    Returns
    -------
    all_results : dict
        Mapping ``{subject: [result_or_error, ...]}``.
    """
    from distributed import Client, as_completed

    cluster = _build_cluster(exec_config)
    client = None
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    try:
        client = Client(cluster)
        logger.info(f"Dask dashboard: {client.dashboard_link}")

        futures = {}
        for recording in recordings:
            future = client.submit(
                _run_recording_job,
                reader,
                output_root,
                config,
                recording["paths"],
                recording["metadata"],
                io_backend,
                pure=False,
            )
            futures[future] = recording

        n_total = len(recordings)
        n_done = 0
        for future in as_completed(futures):
            recording = futures[future]
            recording_name = recording["recording_name"]
            subject_key = _subject_key(recording["metadata"])
            n_done += 1
            try:
                result = future.result()
                all_results.setdefault(subject_key, []).append(result)
                logger.info(f"[{n_done}/{n_total}] Completed {recording_name}")
            except Exception as exc:
                # Do not stop the whole batch if one recording fails.
                logger.error(f"[{n_done}/{n_total}] Error processing {recording_name}: {exc}")
                all_results.setdefault(subject_key, []).append({"error": str(exc)})
    finally:
        if client is not None:
            client.close()
        cluster.close()

    return all_results


def dispatch(
    recordings: List[Dict[str, Any]],
    reader: "DatasetReader",
    output_root: Optional[Union[str, Path]],
    config: Dict[str, Any],
    step_functions: Dict[str, Callable],
    io_backend: str,
    exec_config: ExecutionConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """Dispatch one job per recording, routing to the backend named in ``exec_config``."""
    if exec_config.backend == SEQUENTIAL_BACKEND:
        return run_sequential(
            recordings,
            reader=reader,
            output_root=output_root,
            config=config,
            step_functions=step_functions,
            io_backend=io_backend,
        )
    return run_dask(
        recordings,
        reader=reader,
        output_root=output_root,
        config=config,
        io_backend=io_backend,
        exec_config=exec_config,
    )
