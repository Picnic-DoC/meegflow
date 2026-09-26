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

## Top-level `datatype`

An optional top-level `datatype` key declares whether the pipeline processes EEG or MEG. It sets the default of every parameter that depends on the kind of data; a parameter given in a step always wins.

```yaml
datatype: meg      # 'eeg' (default) or 'meg'
pipeline:
  - name: bandpass_filter      # filters the MEG channels, since picks is omitted
    l_freq: 1.0
    h_freq: 40.0
```

| Default | `datatype: eeg` (or key absent) | `datatype: meg` |
|---|---|---|
| `picks` of every step | EEG channels | MEG channels (magnetometers and gradiometers, not the reference channels) |
| `reject` of the threshold detectors | `{eeg: 100e-6}` | `{mag: 4e-12, grad: 4e-10}` (4000 fT, 4000 fT/cm) |
| BIDS reader datatype | detected from the dataset (EEG preferred) | `meg` |
| Datatype folder of outputs and reports, and the suffix of a saved raw recording | as before the key existed | `meg` |
| CLI `--extension` | `.vhdr` | `.fif` |

Without the key, a configuration behaves exactly as before. `--datatype` on the command line, or a `datatype` given to `BIDSReader`, takes precedence for the reader.

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
| `find_flat_channels` | `threshold` (a variance: a number for every channel, or a dict per channel type; defaults `mag: 1e-30`, `grad: 1e-26`, `1e-12` for every other type), `picks`, `excluded_channels` |

### Bad channel handling

| Step | Key parameters |
|------|---------------|
| `interpolate_bad_channels` | `instance`, `picks` |
| `drop_bad_channels` | `instance` |

### ICA

| Step | Key parameters |
|------|---------------|
| `ica` | `n_components`, `method`, `fit_params`, `picks`, `eog_channel`, `ecg_channel` |

### MEG

Wrappers around MNE-Python's standard MEG operations. Any keyword argument of the underlying MNE function can be given and is passed through, so MNE's defaults apply.

| Step | Key parameters |
|------|---------------|
| `maxwell_filter` | `calibration`, `cross_talk` (fine-calibration and cross-talk files), `st_duration` (enables tSSS), `head_pos` (context entry or `.pos` file, for movement compensation), `instance`, and any argument of `mne.preprocessing.maxwell_filter` |
| `find_bads_maxwell` | `calibration`, `cross_talk`, `apply_on`, `instance`, and any argument of `mne.preprocessing.find_bad_channels_maxwell` |
| `compute_head_pos` | `pos_file` (read instead of compute), `var_name` (default `head_pos`), `save`, and the cHPI parameters `t_step_min`, `t_window`, `dist_limit`, `gof_limit`, ... |
| `compute_ssp` | `artifact` (`ecg` or `eog`, required), `n_grad`, `n_mag`, `n_eeg`, `ch_name`, `apply` (default `false`), and any argument of `mne.preprocessing.compute_proj_ecg` / `compute_proj_eog` |
| `apply_gradient_compensation` | `grade` (CTF compensation grade, default `3`), `instance` |

```yaml
datatype: meg
pipeline:
  - name: concatenate_recordings
  - name: compute_head_pos           # from the cHPI coils
  - name: find_bads_maxwell
    calibration: sss_cal.dat
    cross_talk: ct_sparse.fif
  - name: maxwell_filter             # tSSS with movement compensation
    calibration: sss_cal.dat
    cross_talk: ct_sparse.fif
    st_duration: 10
    head_pos: head_pos
  - name: compute_ssp
    artifact: ecg
```

A complete MEG pipeline is in [Example Configurations](examples.md#meg).

### Epoching

| Step | Key parameters |
|------|---------------|
| `find_events` | `get_events_from` (`'annotations'`\|`'stim_channel'`), `shortest_event`, `event_id`, `stim_channel` |
| `epoch` | `event_id`, `tmin`, `tmax`, `baseline`, `reject` |
| `chunk_in_epoch` | `duration` |

### Output

| Step | Key parameters |
|------|---------------|
| `save_clean_instance` | `instance` (`'raw'`\|`'epochs'`), `overwrite` |
| `generate_json_report` | *(no parameters)* |
| `generate_html_report` | `picks`, `excluded_channels`, `outlines`, `compare_instances` |

## Common parameter patterns

### `picks`

A list of channel types understood by `mne.pick_types` (e.g. `['eeg']`, `['meg']`, `['eeg', 'eog']`). If omitted, the channels of the top-level `datatype`: EEG by default, or MEG (without reference channels) with `datatype: meg`.

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
