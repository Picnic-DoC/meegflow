"""Tests for the Dask-based parallel execution scheduling abstraction.

Covers ``meegflow.execution``: backend selection/parsing (``ExecutionConfig``),
the sequential backend (today's default, unchanged behavior), and the local
Dask backend (mocked ``dask-jobqueue`` backends, and a real, in-process
``distributed.LocalCluster`` for the ``local`` backend so the actual
scheduling code path is exercised, not just eyeballed).

See ``docs/dask_parallel_execution.md`` for the design this implements.
"""
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


# --------------------------------------------------------------------- #
# Test fixtures: a minimal, picklable fake reader + step functions       #
# --------------------------------------------------------------------- #

class FakeReader:
    """Minimal DatasetReader stand-in: no real files, just echoes paths back."""

    def __init__(self, root="/fake/root"):
        self._root = Path(root)

    @property
    def root(self):
        return self._root

    def read(self, paths, io_backend="read_raw_bids"):
        # Return one lightweight placeholder "raw" per path.
        return [{"path": str(p)} for p in paths]


def _step_double_counter(data, step_config):
    """Trivial step: increments data['counter'] by step_config['by']."""
    data["counter"] = data.get("counter", 0) + step_config.get("by", 1)
    data["preprocessing_steps"].append({"step": "double_counter"})
    return data


def _step_fail_for_subject(data, step_config):
    """Raises for a specific subject, to test per-recording error isolation."""
    if data.get("subject") == step_config.get("bad_subject"):
        raise RuntimeError(f"boom for subject {data.get('subject')}")
    data["preprocessing_steps"].append({"step": "fail_for_subject"})
    return data


FAKE_STEP_FUNCTIONS = {
    "double_counter": _step_double_counter,
    "fail_for_subject": _step_fail_for_subject,
}


def _make_recordings(subjects):
    return [
        {
            "paths": [f"/fake/root/sub-{s}_eeg.vhdr"],
            "metadata": {"subject": s, "session": None, "task": "rest", "acquisition": None},
            "recording_name": f"sub-{s}",
        }
        for s in subjects
    ]


# --------------------------------------------------------------------- #
# ExecutionConfig                                                        #
# --------------------------------------------------------------------- #

class TestExecutionConfig:
    def test_defaults_to_sequential(self):
        from meegflow.execution import ExecutionConfig, SEQUENTIAL_BACKEND

        exec_config = ExecutionConfig.from_config({})
        assert exec_config.backend == SEQUENTIAL_BACKEND
        assert exec_config.n_workers == 1
        assert exec_config.cluster_kwargs == {}

    def test_defaults_to_sequential_when_no_execution_key(self):
        from meegflow.execution import ExecutionConfig, SEQUENTIAL_BACKEND

        exec_config = ExecutionConfig.from_config({"pipeline": []})
        assert exec_config.backend == SEQUENTIAL_BACKEND

    def test_parses_local_backend_and_workers(self):
        from meegflow.execution import ExecutionConfig

        config = {
            "execution": {
                "backend": "local",
                "n_workers": 4,
                "cluster_kwargs": {"threads_per_worker": 1},
            }
        }
        exec_config = ExecutionConfig.from_config(config)
        assert exec_config.backend == "local"
        assert exec_config.n_workers == 4
        assert exec_config.cluster_kwargs == {"threads_per_worker": 1}

    def test_parses_jobqueue_backend(self):
        from meegflow.execution import ExecutionConfig

        config = {"execution": {"backend": "slurm", "n_workers": 8}}
        exec_config = ExecutionConfig.from_config(config)
        assert exec_config.backend == "slurm"
        assert exec_config.n_workers == 8

    def test_unknown_backend_raises(self):
        from meegflow.execution import ExecutionConfig

        with pytest.raises(ValueError, match="Unknown execution backend"):
            ExecutionConfig.from_config({"execution": {"backend": "not-a-backend"}})


# --------------------------------------------------------------------- #
# run_sequential                                                         #
# --------------------------------------------------------------------- #

class TestRunSequential:
    def test_all_recordings_succeed(self):
        from meegflow.execution import run_sequential

        recordings = _make_recordings(["01", "02", "03"])
        results = run_sequential(
            recordings,
            reader=FakeReader(),
            output_root=None,
            config={"pipeline": [{"name": "double_counter", "by": 5}]},
            step_functions=FAKE_STEP_FUNCTIONS,
            io_backend="read_raw_bids",
        )

        assert set(results.keys()) == {"01", "02", "03"}
        for subject, subject_results in results.items():
            assert len(subject_results) == 1
            result = subject_results[0]
            assert "error" not in result
            assert result["subject"] == subject
            assert result["preprocessing_steps"] == [{"step": "double_counter"}]

    def test_one_recording_failure_does_not_abort_batch(self):
        from meegflow.execution import run_sequential

        recordings = _make_recordings(["01", "02", "03"])
        results = run_sequential(
            recordings,
            reader=FakeReader(),
            output_root=None,
            config={
                "pipeline": [{"name": "fail_for_subject", "bad_subject": "02"}],
            },
            step_functions=FAKE_STEP_FUNCTIONS,
            io_backend="read_raw_bids",
        )

        assert set(results.keys()) == {"01", "02", "03"}
        assert "error" not in results["01"][0]
        assert "error" in results["02"][0]
        assert "boom for subject 02" in results["02"][0]["error"]
        assert "error" not in results["03"][0]


# --------------------------------------------------------------------- #
# run_dask (local backend, real in-process distributed.LocalCluster)    #
# --------------------------------------------------------------------- #

pytest.importorskip("distributed")

# Unlike run_sequential (which takes a ready-made step_functions dict),
# run_dask's workers each call build_step_functions(config) themselves (see
# docs/dask_parallel_execution.md §4.3) -- they can't be handed arbitrary,
# already-loaded step functions directly. To exercise a step that isn't a
# built-in, these tests write it to a real custom_steps_folder on disk and
# point 'config["custom_steps_folder"]' at it, exactly like a real user
# would for the 'local' backend (same-machine workers, same filesystem).
_CUSTOM_STEPS_SOURCE = '''
def double_counter(data, step_config):
    data["counter"] = data.get("counter", 0) + step_config.get("by", 1)
    data["preprocessing_steps"].append({"step": "double_counter"})
    return data


def fail_for_subject(data, step_config):
    if data.get("subject") == step_config.get("bad_subject"):
        raise RuntimeError(f"boom for subject {data.get('subject')}")
    data["preprocessing_steps"].append({"step": "fail_for_subject"})
    return data
'''


@pytest.fixture
def custom_steps_folder(tmp_path):
    steps_file = tmp_path / "fake_steps.py"
    steps_file.write_text(_CUSTOM_STEPS_SOURCE)
    return str(tmp_path)


class TestRunDaskLocalBackend:
    """Exercises the real local-backend code path with an actual (threaded,
    for speed/determinism inside a sandboxed test run) distributed.LocalCluster,
    not a mock -- validates ExecutionConfig -> cluster construction -> Client
    submission -> as_completed gathering end to end."""

    def test_all_recordings_succeed(self, custom_steps_folder):
        from meegflow.execution import ExecutionConfig, run_dask

        recordings = _make_recordings(["01", "02"])
        exec_config = ExecutionConfig(
            backend="local",
            n_workers=1,
            cluster_kwargs={"processes": False, "threads_per_worker": 1},
        )

        results = run_dask(
            recordings,
            reader=FakeReader(),
            output_root=None,
            config={
                "custom_steps_folder": custom_steps_folder,
                "pipeline": [{"name": "double_counter", "by": 2}],
            },
            io_backend="read_raw_bids",
            exec_config=exec_config,
        )

        assert set(results.keys()) == {"01", "02"}
        for subject, subject_results in results.items():
            result = subject_results[0]
            assert "error" not in result
            assert result["subject"] == subject
            assert result["preprocessing_steps"] == [{"step": "double_counter"}]

    def test_one_recording_failure_does_not_abort_batch(self, custom_steps_folder):
        from meegflow.execution import ExecutionConfig, run_dask

        recordings = _make_recordings(["01", "02", "03"])
        exec_config = ExecutionConfig(
            backend="local",
            n_workers=1,
            cluster_kwargs={"processes": False, "threads_per_worker": 1},
        )

        results = run_dask(
            recordings,
            reader=FakeReader(),
            output_root=None,
            config={
                "custom_steps_folder": custom_steps_folder,
                "pipeline": [{"name": "fail_for_subject", "bad_subject": "02"}],
            },
            io_backend="read_raw_bids",
            exec_config=exec_config,
        )

        assert set(results.keys()) == {"01", "02", "03"}
        assert "error" not in results["01"][0]
        assert "error" in results["02"][0]
        assert "boom for subject 02" in results["02"][0]["error"]
        assert "error" not in results["03"][0]


# --------------------------------------------------------------------- #
# _build_cluster: jobqueue backends (mocked -- no real cluster to test    #
# against) and missing-dependency error messages                        #
# --------------------------------------------------------------------- #

class TestBuildCluster:
    def test_local_backend_missing_distributed_raises_helpful_error(self, monkeypatch):
        from meegflow import execution

        monkeypatch.setitem(sys.modules, "distributed", None)
        exec_config = execution.ExecutionConfig(backend="local")
        with pytest.raises(ImportError, match="meegflow\\[dask\\]"):
            execution._build_cluster(exec_config)

    def test_jobqueue_backend_missing_dask_jobqueue_raises_helpful_error(self, monkeypatch):
        from meegflow import execution

        monkeypatch.setitem(sys.modules, "dask_jobqueue", None)
        exec_config = execution.ExecutionConfig(backend="slurm")
        with pytest.raises(ImportError, match="meegflow\\[dask-jobqueue\\]"):
            execution._build_cluster(exec_config)

    def test_jobqueue_backend_builds_matching_cluster_class(self, monkeypatch):
        """Mocks dask_jobqueue itself (no real Slurm/PBS/... scheduler is
        available to test against) and checks the right cluster class is
        instantiated with cluster_kwargs, then scaled to n_workers."""
        from meegflow import execution
        from unittest.mock import MagicMock

        fake_module = MagicMock()
        fake_cluster_instance = MagicMock()
        fake_module.SLURMCluster.return_value = fake_cluster_instance
        monkeypatch.setitem(sys.modules, "dask_jobqueue", fake_module)

        exec_config = execution.ExecutionConfig(
            backend="slurm",
            n_workers=3,
            cluster_kwargs={"queue": "normal", "cores": 4, "memory": "16GB"},
        )
        cluster = execution._build_cluster(exec_config)

        fake_module.SLURMCluster.assert_called_once_with(
            queue="normal", cores=4, memory="16GB"
        )
        fake_cluster_instance.scale.assert_called_once_with(jobs=3)
        assert cluster is fake_cluster_instance


# --------------------------------------------------------------------- #
# dispatch(): routes to the right backend                                #
# --------------------------------------------------------------------- #

class TestDispatch:
    def test_sequential_backend_calls_run_sequential(self, monkeypatch):
        from meegflow import execution

        called = {}

        def fake_run_sequential(recordings, **kwargs):
            called["ran"] = "sequential"
            return {}

        def fake_run_dask(recordings, **kwargs):
            called["ran"] = "dask"
            return {}

        monkeypatch.setattr(execution, "run_sequential", fake_run_sequential)
        monkeypatch.setattr(execution, "run_dask", fake_run_dask)

        execution.dispatch(
            [],
            reader=FakeReader(),
            output_root=None,
            config={},
            step_functions={},
            io_backend="read_raw_bids",
            exec_config=execution.ExecutionConfig(backend="sequential"),
        )
        assert called["ran"] == "sequential"

    def test_non_sequential_backend_calls_run_dask(self, monkeypatch):
        from meegflow import execution

        called = {}

        def fake_run_sequential(recordings, **kwargs):
            called["ran"] = "sequential"
            return {}

        def fake_run_dask(recordings, **kwargs):
            called["ran"] = "dask"
            return {}

        monkeypatch.setattr(execution, "run_sequential", fake_run_sequential)
        monkeypatch.setattr(execution, "run_dask", fake_run_dask)

        execution.dispatch(
            [],
            reader=FakeReader(),
            output_root=None,
            config={},
            step_functions={},
            io_backend="read_raw_bids",
            exec_config=execution.ExecutionConfig(backend="local"),
        )
        assert called["ran"] == "dask"
