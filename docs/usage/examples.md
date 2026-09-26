# Example Configurations

MEEGFlow does not ship configuration files: a configuration is a YAML file you write for your study (see the [Configuration Reference](configuration.md)). The examples below are complete configurations to copy and adapt. Save one as `config.yaml` and pass it to the command line with `--config config.yaml`.

## Dropping bad channels

Detects flat channels on the continuous recording, then drops them instead of interpolating them. Channels marked bad on `raw` are inherited by the epochs, so `drop_bad_channels` removes them from both instances.

```yaml
pipeline:
  - name: concatenate_recordings

  - name: set_montage
    montage: standard_1020

  - name: bandpass_filter
    l_freq: 0.1
    h_freq: 40.0

  - name: notch_filter
    freqs: [50.0, 100.0]

  - name: resample
    instance: raw
    sfreq: 250.0
    npad: auto

  # Detect flat channels on the continuous data, before epoching: the epochs
  # inherit info['bads'] from raw.
  - name: find_flat_channels

  - name: find_events
    get_events_from: annotations
    shortest_event: 1

  - name: epoch
    tmin: -0.2
    tmax: 1.2
    baseline: [null, 0.0]
    reject: null

  - name: reference
    instance: 'epochs'
    ref_channels: average

  # Drop bad channels instead of interpolating them
  - name: drop_bad_channels
    instance: epochs

  - name: reference
    instance: 'raw'
    ref_channels: average

  # Drop bad channels from raw data as well
  - name: drop_bad_channels
    instance: raw

  - name: save_clean_instance
    instance: epochs
    overwrite: true

  - name: save_clean_instance
    instance: raw
    overwrite: true

  - name: generate_json_report

  - name: generate_html_report
```

## Excluding channels

`excluded_channels` keeps specific channels (here the reference channel, Cz) out of filtering and bad-channel detection.

```yaml
pipeline:
  - name: set_montage
    montage: standard_1020

  # Bandpass filter - excluding Cz from filtering to avoid reference problems
  - name: bandpass_filter
    l_freq: 0.5
    h_freq: 45.0
    excluded_channels: ['Cz']  # Exclude Cz from bandpass filtering

  # Notch filter - also excluding Cz
  - name: notch_filter
    freqs: [50.0, 100.0]
    excluded_channels: ['Cz']  # Exclude Cz from notch filtering

  - name: resample
    instance: raw
    sfreq: 250.0
    npad: 'auto'

  - name: find_events
    shortest_event: 1

  # Find bad channels - excluding Cz from bad channel detection.
  # Runs on raw, before epoching: the epochs inherit info['bads'].
  - name: find_flat_channels
    excluded_channels: ['Cz']  # Don't mark Cz as bad

  - name: epoch
    tmin: -0.2
    tmax: 0.8
    baseline: [null, 0.0]
    event_id: null
    reject: null

  # After bad channel detection, re-reference to average
  - name: reference
    instance: epochs
    ref_channels: average

  - name: interpolate_bad_channels
    instance: epochs

  - name: save_clean_instance
    instance: epochs
    overwrite: true

  - name: generate_json_report

  - name: generate_html_report
```

## Custom steps

Custom steps are plain Python functions that take `(data, step_config)` and return `data`. Put them in a Python file inside a folder, name that folder in `custom_steps_folder`, and use each function's name as a step name. Custom and built-in steps can be mixed freely.

```yaml
# Path to folder containing custom step files
# For local use: absolute or relative path
# For Docker: mount your local folder and use the container path
custom_steps_folder: /path/to/your/custom_steps

# Pipeline steps execute in order
# Mix custom and built-in steps as needed
pipeline:
  # Load and prepare data
  - name: concatenate_recordings  # Built-in step

  # Set montage (built-in step)
  - name: set_montage
    montage: standard_1020

  # Apply custom filtering
  - name: my_custom_filter  # Custom step, defined in the Python file below
    cutoff_freq: 30.0

  # Apply built-in bandpass filter
  - name: bandpass_filter  # Built-in step
    l_freq: 0.5
    h_freq: 40.0

  # Mark specific channels as bad (custom step)
  - name: mark_bad_channels_by_name  # Custom step
    channels: ['T7', 'T8']  # Example: mark these as bad
    instance: 'raw'

  # Find more bad channels using built-in detection (operates on raw)
  - name: find_flat_channels  # Built-in step

  # Re-reference (built-in step)
  - name: reference
    ref_channels: average
    instance: 'raw'

  # Find events (built-in step)
  - name: find_events
    shortest_event: 1

  # Epoch (built-in step)
  - name: epoch
    tmin: -0.2
    tmax: 0.8
    baseline: [null, 0]

  # Interpolate bad channels, including those marked above (built-in step)
  - name: interpolate_bad_channels

  # Compute custom quality metric
  - name: compute_custom_metric  # Custom step
    metric_name: data_quality_score
    instance: epochs

  # Save results (built-in steps)
  - name: save_clean_instance
    instance: epochs

  - name: generate_json_report

  - name: generate_html_report
```

`/path/to/your/custom_steps/example_custom_steps.py`:

```python
"""
Example custom preprocessing steps for MEEGFlow.

Each public function that takes exactly two parameters, (data, step_config),
becomes a step named after the function. Functions whose name starts with an
underscore, or that take a different number of parameters, are not loaded.
"""

from typing import Dict, Any


def my_custom_filter(data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example custom preprocessing step: Apply a custom low-pass filter.
    
    This step demonstrates the required structure for custom preprocessing steps.
    All custom steps must:
    1. Accept exactly 2 parameters: data (Dict) and step_config (Dict)
    2. Return the updated data dictionary
    3. Validate that required data instances exist (e.g., 'raw', 'epochs')
    4. Append a summary to data['preprocessing_steps']
    5. Document parameters clearly
    
    Parameters (via step_config)
    -----------------------------
    cutoff_freq : float, optional
        Low-pass filter cutoff frequency in Hz (default: 30.0)
    
    Updates
    -------
    data['raw'] : mne.io.Raw
        Applies low-pass filter in-place
    data['preprocessing_steps'] : list
        Appends step information
    
    Returns
    -------
    data : dict
        Updated data dictionary with filtered raw data
    
    Raises
    ------
    ValueError
        If 'raw' is not present in data
    """
    # Validate that required data exists
    if 'raw' not in data:
        raise ValueError("my_custom_filter requires 'raw' to be in data")
    
    # Get parameters from step_config with defaults
    cutoff_freq = step_config.get('cutoff_freq', 30.0)
    
    # Apply the custom processing
    data['raw'].filter(h_freq=cutoff_freq, l_freq=None)
    
    # Record what was done for the report
    data['preprocessing_steps'].append({
        'step': 'my_custom_filter',
        'cutoff_freq': cutoff_freq,
        'description': f'Applied custom low-pass filter at {cutoff_freq} Hz'
    })
    
    return data


def mark_bad_channels_by_name(data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example custom step: Mark specific channels as bad by name.
    
    This demonstrates how to create a custom step that marks channels as bad
    without dropping them, allowing later interpolation.
    
    Parameters (via step_config)
    -----------------------------
    channels : list of str
        Channel names to mark as bad
    instance : str, optional
        Which data instance to work on: 'raw' or 'epochs' (default: 'raw')
    
    Updates
    -------
    data[instance].info['bads'] : list
        Adds specified channels to the bad channels list
    data['preprocessing_steps'] : list
        Appends step information
    
    Returns
    -------
    data : dict
        Updated data dictionary with marked bad channels
    """
    channels = step_config.get('channels', [])
    instance = step_config.get('instance', 'raw')
    
    if instance not in data:
        raise ValueError(f"mark_bad_channels_by_name requires '{instance}' to be in data")
    
    # Mark channels as bad
    existing_bads = set(data[instance].info['bads'])
    new_bads = set(channels)
    data[instance].info['bads'] = list(existing_bads | new_bads)
    
    # Record the action
    data['preprocessing_steps'].append({
        'step': 'mark_bad_channels_by_name',
        'instance': instance,
        'channels_marked': list(new_bads),
        'total_bads': len(data[instance].info['bads'])
    })
    
    return data


def compute_custom_metric(data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example custom step: Compute a custom data quality metric.
    
    This demonstrates how to compute custom metrics without modifying the data.
    The metric is stored in the data dictionary and will appear in reports.
    
    Parameters (via step_config)
    -----------------------------
    metric_name : str, optional
        Name for the metric (default: 'custom_quality')
    instance : str, optional
        Which data instance to analyze: 'raw' or 'epochs' (default: 'raw')
    
    Updates
    -------
    data[metric_name] : float
        Computed metric value
    data['preprocessing_steps'] : list
        Appends step information
    
    Returns
    -------
    data : dict
        Updated data dictionary with computed metric
    """
    import numpy as np
    
    metric_name = step_config.get('metric_name', 'custom_quality')
    instance = step_config.get('instance', 'raw')
    
    if instance not in data:
        raise ValueError(f"compute_custom_metric requires '{instance}' to be in data")
    
    # Example metric: compute standard deviation across all channels
    raw_data = data[instance].get_data()
    metric_value = float(np.std(raw_data))
    
    # Store the metric
    data[metric_name] = metric_value
    
    # Record the computation
    data['preprocessing_steps'].append({
        'step': 'compute_custom_metric',
        'metric_name': metric_name,
        'metric_value': metric_value,
        'instance': instance
    })
    
    return data


# Note: Functions starting with underscore are NOT loaded as steps
def _helper_function():
    """Helper functions starting with _ are ignored by the loader."""
    pass


# Note: Functions with wrong signatures are NOT loaded as steps
def wrong_signature(only_one_param):
    """Functions without exactly 2 parameters are ignored."""
    pass
```

## Parallel execution

Processes the recordings in parallel through Dask. The `execution` block is optional: without it (or with `backend: sequential`) recordings are processed one after another in a single process. See [Parallel Execution](parallel-execution.md) for the full reference.

```yaml
execution:
  # 'local' runs an in-process Dask cluster (comparable to
  # ProcessPoolExecutor/joblib) -- good for a multi-core workstation.
  # Requires the 'dask' extra: pip install "meegflow[dask]"
  backend: local
  n_workers: 4

  # For an HPC cluster instead, use one of: slurm | pbs | sge | lsf | htcondor
  # (requires: pip install "meegflow[dask-jobqueue]"), e.g.:
  #
  # backend: slurm
  # n_workers: 8
  # cluster_kwargs:
  #   queue: normal
  #   cores: 4
  #   memory: 16GB
  #   walltime: "02:00:00"

pipeline:
  - name: bandpass_filter
    l_freq: 0.5
    h_freq: 40.0
  - name: reference
    ref_channels: average
    instance: 'raw'
  - name: find_events
    shortest_event: 1
  - name: epoch
    tmin: -0.2
    tmax: 0.8
    baseline: [null, 0]
    event_id: null
    reject:
      eeg: 1.5e-04
  - name: save_clean_instance
    instance: epochs
  - name: generate_json_report
  - name: generate_html_report
```

## MEG

A complete pipeline for Elekta/MEGIN Neuromag data. `datatype: meg` makes MEG the default wherever a parameter is omitted: steps pick the MEG channels (magnetometers and gradiometers, without reference channels), the threshold detectors use MEG rejection thresholds, the BIDS reader searches the `meg` datatype, and outputs are labelled `meg`.

```yaml
# Run on a BIDS dataset:
#   meegflow --bids-root /path/to/bids --config config_meg.yaml
# or on any other layout:
#   meegflow --reader glob --data-root /path/to/data \
#       --glob-pattern 'sub-{subject}/{task}_raw.fif' --io-backend read_raw_fif \
#       --config config_meg.yaml

datatype: meg

pipeline:
  - name: concatenate_recordings

  # Head position from the cHPI coils, for movement compensation below.
  # Remove this step (and `head_pos` below) for recordings without cHPI.
  - name: compute_head_pos
    save: true

  # Noisy and flat MEG channels, marked bad before Maxwell filtering.
  - name: find_bads_maxwell
    calibration: /path/to/sss_cal.dat
    cross_talk: /path/to/ct_sparse.fif

  # tSSS with movement compensation (MNE's defaults for everything else).
  - name: maxwell_filter
    calibration: /path/to/sss_cal.dat
    cross_talk: /path/to/ct_sparse.fif
    st_duration: 10
    head_pos: head_pos

  - name: bandpass_filter
    l_freq: 0.1
    h_freq: 40.0

  - name: notch_filter
    freqs: [50.0, 100.0]

  # Cardiac and ocular projectors (SSP). Without an ECG channel, MNE builds
  # a synthetic ECG from the magnetometers. They are applied by `epoch`.
  - name: compute_ssp
    artifact: ecg
  - name: compute_ssp
    artifact: eog

  - name: find_events
    get_events_from: stim_channel
    stim_channel: STI 014

  - name: epoch
    tmin: -0.2
    tmax: 0.5

  - name: save_clean_instance
    instance: epochs

  - name: generate_json_report
  - name: generate_html_report
```
