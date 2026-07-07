# Configuration Reference

Pipelines are defined in YAML. The top-level key is `pipeline`, which is a list of step objects. Each step has a `name` field plus any parameters that step accepts.

## Minimal example

```yaml
pipeline:
  - name: bandpass_filter
    l_freq: 0.5
    h_freq: 40.0
  - name: epoch
    tmin: -0.2
    tmax: 0.8
  - name: save_clean_instance
```

## Available steps

### Data loading / setup

| Step | Key parameters |
|------|---------------|
| `set_montage` | `montage` (str), `match_case` (bool) |
| `drop_unused_channels` | `channels` (list) |
| `strip_recording` | `tmin`, `tmax` |
| `copy_instance` | `source`, `dest` |
| `concatenate_recordings` | `instances` (list) |

### Dynamic module call

| Step | Key parameters |
|------|---------------|
| `call_module` | `module` (str), `target` (str, optional), `var_name` (str\|null), `unpack_as` (list, optional), `args` (list, optional), plus any keyword arguments |

`call_module` dynamically imports and calls any Python callable, or calls a method on an object already in the pipeline data dict. It is a lightweight escape hatch for using MNE functions or any other library directly from the config without writing a custom step.

**Parameters**

- `module`: Fully-qualified dotted path to the callable (e.g. `mne.channels.make_standard_montage`) when calling a module-level function. Just the method name (e.g. `set_montage`) when `target` is also provided.
- `target` *(optional)*: A `data__`-prefixed reference to an object already in `data`. When present, `module` is treated as a method name on that object.
- `var_name`: Key under which the return value is stored in `data`. Set to `null` to discard the result (useful for in-place methods). Mutually exclusive with `unpack_as`.
- `unpack_as` *(optional)*: A list of data keys to unpack a multi-value return into, in order. Mutually exclusive with `var_name`.
- `args` *(optional)*: A YAML list of positional arguments forwarded to the callable in order.
- Any additional key/value pairs are forwarded as keyword arguments.

**Referencing pipeline data**

Any string value (in `args`, keyword arguments, or `target`) that starts with `data__` is resolved as a path into the pipeline data dict, using `__` as the key separator:

| Config value | Resolved as |
|---|---|
| `"data__raw"` | `data['raw']` |
| `"data__house__dog"` | `data['house']['dog']` |

**Examples**

```yaml
# Module-level function with keyword arguments
- name: call_module
  module: mne.channels.make_standard_montage
  var_name: montage
  kind: standard_1020

# Method call on a data object (the canonical MNE pattern)
- name: call_module
  target: "data__raw"
  module: set_montage
  var_name: null
  montage: "data__montage"
  on_missing: ignore

# Positional-only function via the args list
- name: call_module
  module: os.path.join
  var_name: out_path
  args:
    - "/derivatives"
    - "data__subject"

# Unpack a multi-value return into separate data keys
- name: call_module
  module: mne.events_from_annotations
  unpack_as: [events, event_id]
  args:
    - "data__raw"

# In-place method call — discard return value
- name: call_module
  target: "data__raw"
  module: filter
  var_name: null
  l_freq: 1.0
  h_freq: 40.0
```

!!! note
    For complex logic that transforms data across multiple keys, writes conditional branches, or needs error handling, a [custom step](../api/pipeline.md) is a better fit than chaining many `call_module` steps.

### Filtering

| Step | Key parameters |
|------|---------------|
| `bandpass_filter` | `l_freq`, `h_freq`, `picks`, `n_jobs` |
| `notch_filter` | `freqs` (list), `picks`, `n_jobs` |
| `resample` | `sfreq`, `npad`, `n_jobs`, `resample_events` |

### Referencing

| Step | Key parameters |
|------|---------------|
| `reference` | `ref_channels` (default `'average'`), `instance` |

### Bad channel detection

| Step | Key parameters |
|------|---------------|
| `find_flat_channels` | `threshold` (default `1e-12`), `picks`, `excluded_channels` |
| `find_bads_channels_threshold` | `reject` (dict), `n_epochs_bad_ch`, `picks`, `apply_on` |
| `find_bads_channels_variance` | `zscore_thresh`, `max_iter`, `picks`, `instance`, `apply_on` |
| `find_bads_channels_high_frequency` | `zscore_thresh`, `max_iter`, `picks`, `instance`, `apply_on` |

### Bad channel handling

| Step | Key parameters |
|------|---------------|
| `interpolate_bad_channels` | `instance`, `picks` |
| `drop_bad_channels` | `instance` |

### ICA

| Step | Key parameters |
|------|---------------|
| `ica` | `n_components`, `method`, `fit_params`, `picks`, `eog_channel`, `ecg_channel` |

### Epoching

| Step | Key parameters |
|------|---------------|
| `find_events` | `get_events_from` (`'annotations'`\|`'stim_channel'`), `shortest_event`, `event_id`, `stim_channel` |
| `epoch` | `event_id`, `tmin`, `tmax`, `baseline`, `reject` |
| `chunk_in_epoch` | `duration` |
| `find_bads_epochs_threshold` | `reject` (dict), `n_channels_bad_epoch`, `picks` |

### Output

| Step | Key parameters |
|------|---------------|
| `save_clean_instance` | `instance` (`'raw'`\|`'epochs'`), `overwrite` |
| `generate_json_report` | *(no parameters)* |
| `generate_html_report` | `picks`, `excluded_channels`, `outlines`, `compare_instances` |

## Common parameter patterns

### `picks`

Accepts any value that MNE's `pick_types` understands (e.g. `'eeg'`, `['eeg', 'meg']`, or a list of channel names).

### `excluded_channels`

A list of channel names to exclude from the step (e.g. reference electrodes).

### `apply_on`

Some bad-channel detection steps accept `apply_on: [raw, epochs]` to mark the found bad channels on multiple instances simultaneously.

### `instance`

Steps that can act on either raw or epoched data accept an `instance` key: `'raw'` or `'epochs'`.

## Execution

A top-level `execution` block (optional) controls whether recordings are processed sequentially (default) or in parallel via Dask. See [Parallel Execution](parallel-execution.md) for full details.

```yaml
execution:
  backend: local   # sequential (default) | local | slurm | pbs | sge | lsf | htcondor
  n_workers: 4
  cluster_kwargs: {}   # forwarded to the underlying Dask / dask-jobqueue cluster constructor
```
