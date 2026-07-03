# Parallel Execution

By default, MEEGFlow processes recordings **sequentially**, in a single
process — this is unchanged and remains the default with no config changes
required. To process recordings in parallel instead, add an `execution`
block to your YAML config.

## Backends

| `backend` | What it does | Extra required |
|---|---|---|
| `sequential` (default) | Today's single-process, in-order loop | none |
| `local` | An in-process Dask cluster (comparable to `ProcessPoolExecutor`/`joblib`) | `meegflow[dask]` |
| `slurm` / `pbs` / `sge` / `lsf` / `htcondor` | One [`dask-jobqueue`](https://jobqueue.dask.org/) cluster job per worker, for HPC schedulers | `meegflow[dask-jobqueue]` |

Install the extra you need:

```bash
pip install "meegflow[dask]"            # local backend
pip install "meegflow[dask-jobqueue]"   # slurm/pbs/sge/lsf/htcondor backends
```

Whichever backend runs, **one Dask job processes exactly one recording**,
end to end (read → all configured steps → save/report) — the same unit of
work `MEEGFlowPipeline.run_pipeline` already runs sequentially today.

## Config reference

```yaml
execution:
  backend: local        # sequential | local | slurm | pbs | sge | lsf | htcondor
  n_workers: 4           # number of local processes / cluster jobs
  cluster_kwargs: {}      # forwarded verbatim to the underlying cluster constructor
```

- `backend`: selects the scheduler. Omit the whole `execution` block (or set
  `backend: sequential`) to keep today's behavior.
- `n_workers`: number of parallel workers. For `local`, this is the number
  of worker processes on the current machine. For `dask-jobqueue` backends,
  this is the number of scheduler jobs requested (e.g. Slurm jobs).
- `cluster_kwargs`: passed straight through to the underlying constructor —
  [`distributed.LocalCluster`](https://distributed.dask.org/en/stable/api.html#distributed.LocalCluster)
  for `local`, or the matching
  [`dask_jobqueue`](https://jobqueue.dask.org/en/latest/api.html) cluster
  class (`SLURMCluster`, `PBSCluster`, `SGECluster`, `LSFCluster`,
  `HTCondorCluster`) for the HPC backends. MEEGFlow does not validate these
  keys — any error surfaces as whatever `dask`/`dask-jobqueue` itself
  raises.

### Example: local, multi-core workstation

```yaml
execution:
  backend: local
  n_workers: 4
```

### Example: Slurm cluster

```yaml
execution:
  backend: slurm
  n_workers: 8
  cluster_kwargs:
    queue: normal
    cores: 4
    memory: 16GB
    walltime: "02:00:00"
```

## Custom steps on a cluster

Custom step functions (`custom_steps_folder`) are re-loaded independently by
each worker from disk, rather than shipped across the network as already
-loaded Python objects (see the
[design doc](https://github.com/Picnic-DoC/meegflow/blob/main/docs/dask_parallel_execution.md)
for why). Practically, this means:

- **`local` backend**: works out of the box — workers run on the same
  machine, so they see the same filesystem.
- **`dask-jobqueue` backends**: `custom_steps_folder` must be on a
  filesystem shared with (or otherwise reachable from) every compute node —
  the same requirement you already have for the BIDS dataset itself.

## Progress reporting

- **`sequential`**: unchanged — a single `rich` progress bar over
  "recordings processed / total".
- **`local` / `dask-jobqueue`**: a live progress bar can't meaningfully
  represent work happening in other processes (or other machines), so
  MEEGFlow instead logs one line per recording as it's submitted, completed,
  or failed, plus the Dask dashboard URL for the richer live view Dask
  itself provides.

## Failure isolation

Exactly like the sequential backend, one recording's failure does not abort
the batch: it's captured as an `{'error': ...}` entry in the returned
results (and in `pipeline_results.json`), and the remaining recordings still
run.

## Memory

Each worker preloads its recording's data into memory; running `n_workers`
recordings concurrently multiplies peak memory use roughly by `n_workers`.
MEEGFlow does not currently cap `n_workers` automatically based on available
RAM or recording size — start conservative and watch memory usage,
especially for large recordings.
