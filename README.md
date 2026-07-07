# MEEGFlow: MEEG Preprocessing Pipeline

A modular, configuration-driven MEEG preprocessing pipeline using MNE-BIDS. The pipeline uses auxiliary functions for each preprocessing step, allowing you to choose which steps to run, their order, and their parameters through a simple YAML configuration.

## Documentation
https://picnic-doc.github.io/meegflow/

## Features

- **Flexible File Discovery**: Support for both BIDS-formatted datasets and custom glob patterns
- **MNE-BIDS Integration**: Seamlessly reads MEEG data in BIDS format
- **Modular Design**: Each preprocessing step is a separate function
- **Configuration-Driven**: Choose steps, their order, and parameters via YAML
- **Custom Steps Support**: Extend the pipeline with your own preprocessing functions
- **Progress Tracking**: Rich progress bars show real-time progress for recordings (per-recording log lines when running in parallel)
- **Comprehensive Logging**: MNE logger integration with optional log file output
- **Multiple Output Formats**: Save preprocessed data in `.fif`, `.pkl` (pickle), `.h5` (HDF5), or `.npy` (NumPy) format via the `save_clean_instance` step, plus interactive HTML and JSON reports
- **Modular Architecture**: Savers, readers, and pipeline steps are each in their own module for easy extension
- **Batch Processing**: Process multiple subjects sequentially (default) or in parallel via Dask — locally, or on a Slurm/PBS/SGE/LSF/HTCondor cluster
- **Command-line Interface**: Easy to use from the terminal

## Installation

### Option 1: Docker (Recommended)

Using Docker is the easiest way to get started, as it includes all dependencies and system libraries.

1. Build the Docker image:
```bash
git clone https://github.com/Laouen/meegflow.git
cd meegflow
docker build -t meegflow .
```

2. Run the container:
```bash
docker run --rm -v /path/to/bids/data:/data meegflow \
    --bids-root /data \
    --subjects 01 02 \
    --tasks rest \
    --config /app/configs/config_example.yaml
```

### Option 2: Local Installation

1. Clone this repository:
```bash
pip install meegflow
```

## Usage

### Using Docker

To use the Docker image, mount your BIDS dataset directory to `/data` in the container. The outputs will be written to the `derivatives/meegflow` subdirectory within your BIDS root.

**Basic usage:**
```bash
docker run --rm \
    -v /path/to/bids:/data \
    meegflow \
    --bids-root /data \
    --tasks rest
```

**With custom configuration:**
```bash
docker run --rm \
    -v /path/to/bids:/data \
    -v /path/to/custom/config.yaml:/config.yaml \
    meegflow \
    --bids-root /data \
    --subjects 01 02 03 \
    --tasks rest \
    --config /config.yaml
```

**With log file output:**
```bash
docker run --rm \
    -v /path/to/bids:/data \
    -v /path/to/logs:/logs \
    meegflow \
    --bids-root /data \
    --tasks rest \
    --log-file /logs/pipeline.log
```

**Processing specific sessions:**
```bash
docker run --rm \
    -v /path/to/bids:/data \
    meegflow \
    --bids-root /data \
    --subjects 01 02 \
    --sessions 01 02 \
    --tasks rest
```

### Using Local Installation

#### Process Multiple Subjects

Run the preprocessing pipeline on multiple subjects:

```bash
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 03 \
    --tasks rest \
    --config configs/config_example.yaml
```

If you installed the package with `pip install -e .`, you can use the `meegflow` command:

```bash
meegflow \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 03 \
    --tasks rest \
    --config configs/config_example.yaml
```

Process all subjects with a specific task:

```bash
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --tasks rest
```

Process specific subjects with multiple tasks:

```bash
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 \
    --tasks rest task1 task2
```

#### Python API Usage

You can also use the pipeline directly in Python:

```python
from meegflow import MEEGFlowPipeline
from meegflow.readers import BIDSReader

# Load configuration
import yaml
with open('configs/config_example.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create a BIDS reader
reader = BIDSReader('/path/to/bids/dataset')

# Initialize pipeline
pipeline = MEEGFlowPipeline(
    reader=reader,
    output_root='/path/to/derivatives',
    config=config
)

# Run preprocessing on multiple subjects
results = pipeline.run_pipeline(
    subjects=['01', '02', '03'],
    tasks='rest'
)

# Access results for each subject
for subject, result in results.items():
    print(f"Subject {subject}: {result}")
```

## File Discovery with Readers

The pipeline supports two types of file readers for discovering data files:

### BIDS Reader (Default)

The BIDS reader uses MNE-BIDS to automatically discover files in BIDS-formatted datasets:

```bash
# BIDS reader is the default (--reader bids can be omitted)
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 \
    --tasks rest \
    --config configs/config_example.yaml
```

### Glob Reader

The glob reader allows you to work with custom directory structures using glob patterns with variable extraction:

```bash
python src/cli.py \
    --reader glob \
    --data-root /path/to/data \
    --glob-pattern "sub-{subject}/ses-{session}/eeg/sub-{subject}_task-{task}_eeg.vhdr" \
    --subjects 01 02 \
    --tasks rest \
    --config configs/config_example.yaml
```

**Pattern syntax:** Use `{variable_name}` placeholders which:
- Convert to `*` wildcards for file matching
- Extract matched values as metadata

**Python API:**

```python
from meegflow.readers import GlobReader

# Create a glob reader with your custom pattern
reader = GlobReader(
    data_root='/path/to/data',
    pattern='sub-{subject}/ses-{session}/eeg/sub-{subject}_task-{task}_eeg.vhdr'
)

# Initialize pipeline with the glob reader
pipeline = MEEGFlowPipeline(
    reader=reader,
    config=config
)

# Run pipeline
results = pipeline.run_pipeline(subjects=['01', '02'], tasks='rest')
```

For detailed information on readers, pattern examples, and troubleshooting, see the [Readers documentation](docs/usage/readers.md).

## Output Structure

The pipeline creates outputs in a BIDS-derivatives structure:

```
derivatives/meegflow/
├── epochs/              # When saving epochs with save_clean_instance
│   └── sub-01/
│       └── eeg/
│           └── sub-01_task-rest_proc-clean_desc-cleaned_epo.fif
├── raw/                 # When saving raw data with save_clean_instance
│   └── sub-01/
│       └── eeg/
│           └── sub-01_task-rest_proc-clean_desc-cleaned_eeg.fif
└── reports/
    └── sub-01/
        └── eeg/
            ├── sub-01_task-rest_proc-clean_desc-cleaned_report.json
            └── sub-01_task-rest_proc-clean_desc-cleaned_report.html
```

### Output Details

1. **epochs/** or **raw/**: Contains MNE data objects saved in `.fif` format (if `save_clean_instance` step is included)
   - Epochs can be loaded with `mne.read_epochs()`
   - Raw data can be loaded with `mne.io.read_raw_fif()`
   - Includes all preprocessing (filtering, artifact removal, baseline correction)

2. **reports/**: Contains preprocessing reports
   - **JSON report**: Preprocessing parameters, quality metrics, steps performed (generated by `generate_json_report` step)
   - **HTML report**: Interactive visualization (generated by `generate_html_report` step)

## Configuration

The pipeline is configuration-driven. You define a list of preprocessing steps, their order, and parameters in a YAML file.

### Available Steps

Data Organization:
- **strip_recording**: Crop recordings to remove data outside the first and last events
- **concatenate_recordings**: Concatenate multiple raw recordings into a single continuous recording
- **copy_instance**: Create a copy of a data instance for comparison or backup purposes
- **call_module**: Dynamically call any importable function or method on a pipeline object (MNE, NumPy, or any library) and store the result in the pipeline data dict — a lightweight escape hatch for one-off calls that don't warrant a custom step. Supports positional args, `data__` references to pipeline objects, method calls via `target`, and multi-value unpacking via `unpack_as`

Setup:
- **set_montage**: Set channel montage for EEG data
- **drop_unused_channels**: Explicitly drop specified channels by name

Filtering:
- **bandpass_filter**: Apply bandpass filtering
- **notch_filter**: Apply notch filtering

Preprocessing:
- **resample**: Resample data to different sampling frequency
- **reference**: Apply re-referencing
- **ica**: ICA-based artifact removal

Bad Channel Detection:
- **find_flat_channels**: Find flat/disconnected channels based on variance
- **find_bads_channels_threshold**: Find bad channels using threshold-based rejection
- **find_bads_channels_variance**: Find bad channels using variance-based detection
- **find_bads_channels_high_frequency**: Find bad channels using high-frequency variance

Bad Channel Handling:
- **interpolate_bad_channels**: Interpolate bad channels
- **drop_bad_channels**: Drop bad channels without interpolation

Epoching:
- **find_events**: Find events in the data
- **epoch**: Create epochs around events
- **chunk_in_epoch**: Create fixed-length epochs from continuous data
- **find_bads_epochs_threshold**: Find and remove bad epochs using threshold-based rejection

Output:
- **save_clean_instance**: Save raw or epochs data to .fif file
- **generate_json_report**: Generate JSON report
- **generate_html_report**: Generate HTML report

### Example Configuration

See `configs/config_example.yaml` for a full pipeline with epochs:

```yaml
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

See `configs/config_raw_only.yaml` for a simpler pipeline without epoching:

```yaml
pipeline:
  - name: bandpass_filter
    l_freq: 1.0
    h_freq: 30.0
  - name: reference
    ref_channels: average
  - name: ica
    n_components: 15
    method: fastica
    find_eog: true
    apply: true
  - name: generate_json_report
```

See `configs/config_with_adaptive_reject.yaml` for a pipeline with adaptive autoreject steps. This config includes additional preprocessing steps like montage setting, notch filtering, and resampling:

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
  
  - name: find_events
    get_events_from: annotations
    shortest_event: 1
    event_id:
      stim/12hz: 10001
      stim/15hz: 10002
  
  - name: epoch
    tmin: -0.2
    tmax: 1.2
    baseline: [null, 0.0]
    reject: null
  
  - name: reference
    instance: 'epochs'
    ref_channels: average
  
  - name: reference
    instance: 'raw'
    ref_channels: average
  
  - name: generate_html_report
```

Note: This config file also includes commented-out examples of bad channel detection steps (find_bads_channels_threshold, find_bads_channels_variance, find_bads_channels_high_frequency) that can be uncommented and customized as needed.

See `configs/config_minimal.yaml` for a comprehensive pipeline including strip_recording, copy_instance, and ICA:

```yaml
pipeline:
  - name: strip_recording
    instance: all_raw
    get_events_from: annotations
    shortest_event: 5
    start_padding: 1
    end_padding: 1
  
  - name: concatenate_recordings
  
  - name: set_montage
    montage: GSN-HydroCel-256
  
  - name: copy_instance
    from_instance: raw
    to_instance: raw_before_cleaning
  
  - name: find_flat_channels
    threshold: 1.0e-12
  
  - name: bandpass_filter
    l_freq: 0.1
    h_freq: 40.0
  
  - name: chunk_in_epoch
    duration: 1
  
  - name: ica
    n_components: 20
    method: fastica
    find_eog: true
    apply: true
  
  - name: save_clean_instance
    instance: epochs
    overwrite: true
  
  - name: generate_html_report
    compare_instances:
      - title: 'Before vs After Cleaning'
        instance_a:
          name: 'raw'
          label: 'After Cleaning'
        instance_b:
          name: 'raw_before_cleaning'
          label: 'Before Cleaning'
```

Additional example configurations available in `configs/`:
- `config_with_drop_bad_channels.yaml` - Example using drop_bad_channels instead of interpolation
- `config_with_excluded_channels.yaml` - Example using excluded_channels parameter to preserve reference channels
- `config_with_custom_steps.yaml` - Example showing how to integrate custom preprocessing steps
- `config_with_parallel_execution.yaml` - Example enabling parallel recording execution via Dask (see [Parallel Execution with Dask](#parallel-execution-with-dask))

## Command-Line Arguments

### Required Arguments
- `--bids-root`: Path to BIDS root directory

### Optional Filter Arguments
These arguments use the same matching logic as `mne-bids` `find_matching_paths`. If not specified, all matching files will be processed.

- `--subjects`: Subject ID(s) to process, space-separated (e.g., `--subjects 01 02 03`)
- `--sessions`: Session ID(s) to process, space-separated
- `--tasks`: Task name(s) to process, space-separated (e.g., `--tasks rest task1`)
- `--acquisitions`: Acquisition parameter(s) to process
- `--extension`: File extension to process (default: `.vhdr`)

### Other Arguments
- `--output-root`: Custom output path (optional, defaults to `bids-root/derivatives/meegflow`)
- `--config`: Path to YAML configuration file (optional)
- `--io-backend`: MNE IO backend function used to read files (default: `read_raw_bids`)
- `--log-file`: Path to log file (optional, defaults to console output)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, or ERROR (optional, default: INFO)

## Custom Preprocessing Steps

The pipeline supports custom preprocessing steps, allowing you to extend the pipeline with your own processing functions without modifying the core code.

### Creating Custom Steps

1. **Create a Python file** with your custom step functions:

```python
# my_custom_steps.py
def my_custom_filter(data, step_config):
    """Apply custom filtering to raw data."""
    if 'raw' not in data:
        raise ValueError("my_custom_filter requires 'raw' in data")
    
    # Get parameters from step_config
    cutoff_freq = step_config.get('cutoff_freq', 30.0)
    
    # Apply custom processing
    data['raw'].filter(h_freq=cutoff_freq, l_freq=None)
    
    # Record the step for reporting
    data['preprocessing_steps'].append({
        'step': 'my_custom_filter',
        'cutoff_freq': cutoff_freq
    })
    
    return data
```

2. **Place the file in a dedicated folder**, for example: `/path/to/my_custom_steps/`

3. **Update your config file** to specify the custom steps folder:

```yaml
custom_steps_folder: /path/to/my_custom_steps

pipeline:
  - name: my_custom_filter
    cutoff_freq: 30.0
  - name: bandpass_filter  # Built-in steps still work
    l_freq: 0.5
    h_freq: 40.0
```

4. **Run the pipeline** as usual - custom steps are automatically loaded and available.

### Custom Step Requirements

Custom step functions must follow these rules:

- **Signature**: Accept exactly 2 parameters: `data` (Dict) and `step_config` (Dict)
- **Return**: Return the updated `data` dictionary
- **Validation**: Check that required data instances exist (e.g., `'raw'`, `'epochs'`)
- **Recording**: Append a summary to `data['preprocessing_steps']` for reporting
- **Naming**: Function names become step names; avoid starting with underscore

See `configs/example_custom_steps.py` for complete examples.

### Using Custom Steps with Docker

Mount your custom steps folder when running the container:

```bash
docker run -v /host/bids:/data \
           -v /host/custom_steps:/custom_steps \
           -v /host/config:/config \
           meegflow \
           --bids-root /data \
           --subjects 01 02 \
           --tasks rest \
           --config /config/my_config.yaml
```

In your config file, use the container path:

```yaml
custom_steps_folder: /custom_steps
pipeline:
  - name: my_custom_filter
    cutoff_freq: 30.0
```

### Advanced Features

- **Override built-in steps**: Custom steps with the same name as built-in steps will override them
- **Multiple files**: Place multiple `.py` files in the custom steps folder - all will be loaded
- **Error handling**: If a custom step file has errors, other files will still be loaded
- **Private functions**: Functions starting with `_` are ignored and not loaded as steps

## Preprocessing Steps Details

Each step can be customized through the configuration:

### Excluding Channels from Analysis

Many preprocessing steps support an `excluded_channels` parameter that allows you to exclude specific channels (e.g., reference channels like 'Cz') from analysis to avoid reference problems. This is useful when you want to preserve a reference channel or exclude channels that should not be analyzed in certain steps.

**Steps that support `excluded_channels`:**
- `bandpass_filter` - Exclude channels from filtering
- `notch_filter` - Exclude channels from notch filtering
- `ica` - Exclude channels from ICA decomposition
- `find_flat_channels` - Exclude channels from flat channel detection
- `find_bads_channels_threshold` - Exclude channels from bad channel detection
- `find_bads_channels_variance` - Exclude channels from variance-based detection
- `find_bads_channels_high_frequency` - Exclude channels from high-frequency analysis
- `find_bads_epochs_threshold` - Exclude channels from epoch rejection criteria
- `interpolate_bad_channels` - Exclude channels from interpolation even if marked as bad
- `drop_bad_channels` - Exclude channels from dropping even if marked as bad

**Steps where exclusion doesn't apply:**
- `reference` - Reference computation uses selected channels; use `ref_channels` parameter instead
- `resample` - Resamples all data uniformly
- `set_montage` - Sets electrode positions for all channels
- `drop_unused_channels` - Use this for explicit channel removal

**Example usage:**
```yaml
- name: bandpass_filter
  l_freq: 0.5
  h_freq: 45.0
  excluded_channels: ['Cz']  # Exclude Cz from filtering

- name: find_bads_channels_threshold
  reject:
    eeg: 1.0e-4
  excluded_channels: ['Cz', 'FCz']  # Don't mark these as bad

- name: drop_bad_channels
  instance: epochs
  excluded_channels: ['Cz']  # Don't drop Cz even if marked as bad
```

See `configs/config_with_excluded_channels.yaml` for a complete example.

### Data Organization Steps

#### strip_recording
Crop recordings to remove data outside the first and last events. This is useful for removing unnecessary data at the beginning and end of recordings that don't contain task-relevant data.
- `instance`: Which data instance to crop - 'all_raw' or 'raw' (default: 'raw')
- `get_events_from`: How to extract events - 'stim' or 'annotations' (default: 'annotations')
- `shortest_event`: Minimum number of samples for an event (default: 1)
- `event_id`: Event IDs to use for finding start/end points. Can be a dict mapping event names to IDs or 'auto' (default: 'auto')
- `start_padding`: Time in seconds to keep before the first event (default: 1)
- `end_padding`: Time in seconds to keep after the last event (default: 1)

**Example:**
```yaml
- name: strip_recording
  instance: all_raw
  get_events_from: annotations
  shortest_event: 1
  event_id:
    Stimulus/CatNewRepeated/CR: 91
    Stimulus/CatOld/Hit: 101
  start_padding: 1.0
  end_padding: 1.0
```

#### concatenate_recordings
Concatenate multiple raw recordings into a single continuous recording. This is useful when data is split across multiple files but needs to be processed as a single session.
- No parameters required
- Requires 'all_raw' to be present in data
- Creates a single 'raw' instance from all recordings in 'all_raw'

**Example:**
```yaml
- name: concatenate_recordings
```

#### copy_instance
Create a copy of a data instance. This is useful for comparing data at different stages of preprocessing (e.g., before/after cleaning or ICA).
- `from_instance`: Name of the instance to copy from (default: 'raw')
- `to_instance`: Name of the new instance to create (default: 'raw_cleaned')

**Example:**
```yaml
- name: copy_instance
  from_instance: raw
  to_instance: raw_before_ica
```

### Preprocessing Steps

### 1. set_montage
Set channel montage for EEG data. Useful when data lacks electrode position information.
- `montage`: Name of standard montage to use (default: 'standard_1020')
  - Examples: 'standard_1020', 'standard_1005', 'biosemi64', etc.
  - See MNE documentation for available montages

### 2. drop_unused_channels
Explicitly drop specified channels from the data by name. Different from drop_bad_channels, this drops channels regardless of whether they're marked as bad.
- `channels_to_drop`: List of channel names to drop
- `instance`: Which data instance to drop channels from - 'raw' or 'epochs' (default: 'raw')

### 3. bandpass_filter
Apply bandpass filtering.
- `l_freq`: High-pass filter frequency (Hz)
- `h_freq`: Low-pass filter frequency (Hz)
- `l_freq_order`: Filter order for high-pass (default: 6)
- `h_freq_order`: Filter order for low-pass (default: 8)
- `picks`: Optional channel indices to filter
- `excluded_channels`: List of channel names to exclude from filtering (optional)
- `n_jobs`: Number of parallel jobs (default: 1)

### 4. notch_filter
Apply notch filtering to remove line noise.
- `freqs`: Frequencies to notch filter (e.g., [50.0, 100.0])
- `notch_widths`: Width of notch filters (optional)
- `method`: Filtering method (default: 'fft')
- `picks`: Optional channel indices to filter
- `excluded_channels`: List of channel names to exclude from filtering (optional)
- `n_jobs`: Number of parallel jobs (default: 1)

### 5. resample
Resample the data to a different sampling frequency.
- `instance`: Which data instance to resample - 'raw' or 'epochs' (default: 'raw')
- `sfreq`: Target sampling frequency in Hz (default: 250)
- `npad`: Padding to use for resampling (default: 'auto')
- `resample_events`: Whether to also resample events (default: false)
- `n_jobs`: Number of parallel jobs (default: 1)

### 6. reference
Apply re-referencing.
- `ref_channels`: Reference channels ('average' or channel names)
- `instance`: Which data instance to reference - 'raw' or 'epochs' (default: 'epochs')

### 7. find_flat_channels
Find flat/disconnected channels based on variance threshold. Channels with variance below the threshold are marked as bad.
- `picks`: Channel indices to check (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from flat channel detection (optional)
- `threshold`: Variance threshold below which channels are considered flat (default: 1e-12)

### 8. interpolate_bad_channels
Interpolate bad channels using spherical spline interpolation.
- `instance`: Which data instance to interpolate - 'raw' or 'epochs' (default: 'epochs')
- `excluded_channels`: List of channel names to exclude from interpolation (optional)

### 9. drop_bad_channels
Drop bad channels without interpolation. This step removes channels marked as bad from the data instead of interpolating them.
- `instance`: Which data instance to drop channels from - 'raw' or 'epochs' (default: 'epochs')
- `excluded_channels`: List of channel names to exclude from dropping even if marked as bad (optional)

### 10. ica
ICA-based artifact removal.
- `n_components`: Number of ICA components (default: 20)
- `method`: ICA method ('fastica', 'infomax', 'picard', default: 'fastica')
- `random_state`: Random state for reproducibility (default: 97)
- `picks`: Channel types to include in ICA (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from ICA decomposition (optional)
- `ica_fit_l_freq`: High-pass frequency for filtering data before ICA fit (default: 1.0 Hz)
- `ica_fit_h_freq`: Low-pass frequency for filtering data before ICA fit (optional, default: None)
- `find_eog`: Automatically find EOG artifacts (true/false, default: false)
  - `eog_channels`: List of channel names to use for EOG detection (optional, auto-detects if not provided)
  - `eog_threshold`: Correlation threshold for EOG component detection (default: 'auto')
  - `eog_measure`: Measure for EOG detection ('correlation' or 'ctps', default: 'correlation')
  - `eog_l_freq`: High-pass frequency for EOG correlation (default: 1.0 Hz)
  - `eog_h_freq`: Low-pass frequency for EOG correlation (default: 10.0 Hz)
- `find_ecg`: Automatically find ECG artifacts (true/false, default: false)
  - `ecg_channels`: List of channel names to use for ECG detection (optional)
  - `ecg_threshold`: Correlation threshold for ECG component detection (default: 'auto')
  - `ecg_measure`: Measure for ECG detection ('correlation' or 'ctps', default: 'correlation')
  - `ecg_l_freq`: High-pass frequency for ECG correlation (default: 1.0 Hz)
  - `ecg_h_freq`: Low-pass frequency for ECG correlation (default: 10.0 Hz)
- `selected_indices`: Manually specify component indices to exclude (optional, list of integers)
- `apply`: Apply ICA to remove artifacts (true/false, default: true)

### 11. find_events
Find events in the data.
- `get_events_from`: How to extract events - 'stim' or 'annotations' (default: 'annotations')
- `shortest_event`: Minimum event duration in samples (default: 1)
- `event_id`: Event IDs to extract. Can be 'auto' for all events or a dict mapping event names to IDs (default: 'auto')

### 12. epoch
Create epochs around events.
- `tmin`: Start time before event (seconds, default: -0.2)
- `tmax`: End time after event (seconds, default: 0.5)
- `baseline`: Baseline correction window (tuple or null, default: (null, 0.0))
- `event_id`: Event IDs to include (dict or null for all)
- `reject`: Rejection criteria (dict with channel type keys, optional)

### 13. chunk_in_epoch
Create fixed-length epochs from continuous raw data. This is an alternative to event-based epoching that splits the data into equal-duration segments.
- `duration`: Duration of each epoch in seconds (default: 1.0)

**Example:**
```yaml
- name: chunk_in_epoch
  duration: 1.0  # Create 1-second epochs
```

### 14. find_bads_channels_threshold
Find bad channels using threshold-based rejection. Marks channels as bad if they exceed rejection thresholds in too many epochs.
- `picks`: Channel indices to check (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from bad channel detection (optional)
- `reject`: Rejection thresholds by channel type (e.g., `{"eeg": 150e-6}`)
- `n_epochs_bad_ch`: Fraction or number of epochs a channel must be bad in to be marked as bad (default: 0.5)
- `apply_on`: List of instances to mark bad channels on (default: ['epochs'])

### 15. find_bads_channels_variance
Find bad channels using variance-based detection. Identifies channels with abnormally high or low variance.
- `instance`: Which data instance to use - 'raw' or 'epochs' (default: 'epochs')
- `picks`: Channel indices to check (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from variance analysis (optional)
- `zscore_thresh`: Z-score threshold for outlier detection (default: 4)
- `max_iter`: Maximum iterations for iterative outlier removal (default: 2)
- `apply_on`: List of instances to mark bad channels on (default: [instance])

### 16. find_bads_channels_high_frequency
Find bad channels using high-frequency variance. Detects channels with excessive high-frequency noise.
- `instance`: Which data instance to use - 'raw' or 'epochs' (default: 'epochs')
- `picks`: Channel indices to check (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from high-frequency analysis (optional)
- `zscore_thresh`: Z-score threshold for outlier detection (default: 4)
- `max_iter`: Maximum iterations for iterative outlier removal (default: 2)
- `apply_on`: List of instances to mark bad channels on (default: [instance])

### 17. find_bads_epochs_threshold
Find and remove bad epochs using threshold-based rejection. Drops epochs that have too many bad channels.
- `picks`: Channel indices to check (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from epoch rejection criteria (optional)
- `reject`: Rejection thresholds by channel type (e.g., `{"eeg": 150e-6}`)
- `n_channels_bad_epoch`: Fraction or number of channels that must be bad for an epoch to be rejected (default: 0.1)

### 18. save_clean_instance
Save a preprocessed MNE object to the BIDS derivatives tree. The output path follows BIDS conventions and the format is configurable. Supported formats are handled by `meegflow/savers.py`.
- `instance`: Key in the pipeline data dict to save — typically `'raw'` or `'epochs'` (default: `'epochs'`)
- `format`: Output format — `'fif'` (default for MNE objects), `'pickle'`, `'hdf5'`, or `'numpy'` (default for other objects: `'pickle'`). Auto-detected from the object type if omitted.
- `overwrite`: Whether to overwrite existing files (default: `true`)
- `processing`: BIDS `proc` entity for the output path (optional)
- `description`: BIDS `desc` entity for the output path (optional)
- `datatype`: BIDS datatype subfolder (optional)
- `suffix`: BIDS suffix — defaults to `'epo'` for epochs and `'eeg'` for raw (optional)
- `extension`: File extension — inferred from `format` if omitted (optional)

**Example with explicit format:**
```yaml
- name: save_clean_instance
  instance: epochs
  format: hdf5
  overwrite: true
  processing: clean
  description: preprocessed
```

### 19. generate_json_report
Generate JSON report with preprocessing information. No parameters needed.

### 20. generate_html_report
Generate HTML report with interactive visualizations.
- `picks`: Channel types to include in plots (optional, default: EEG channels)
- `excluded_channels`: List of channel names to exclude from plots (optional)
- `compare_instances`: List of instance comparisons to plot (optional, see config_minimal.yaml for example)
- `n_time_points`: Number of time points shown in evoked plots (optional, default: MNE default)
- `plot_raw_kwargs`: Additional keyword arguments for raw data plots (optional, dict)
- `plot_ica_kwargs`: Additional keyword arguments for ICA plots (optional, dict)
- `plot_events_kwargs`: Additional keyword arguments for event plots (optional, dict)
- `plot_epochs_kwargs`: Additional keyword arguments for epoch plots (optional, dict)
- `plot_evokeds_kwargs`: Additional keyword arguments for evoked response plots (optional, dict)

## Batch Processing

By default, the pipeline processes multiple subjects and files **sequentially**, in a single process:

```bash
# Process specific subjects with a specific task
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 03 04 05 \
    --tasks rest \
    --config configs/config_example.yaml

# Process all subjects in the dataset
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --config configs/config_example.yaml

# Process specific sessions for specific subjects
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 \
    --sessions 01 02 \
    --tasks rest
```

### Parallel Execution with Dask

To process recordings in parallel instead, add an `execution` block to your YAML
config — one full pipeline run per recording is dispatched as a separate Dask
job. This is entirely opt-in: omitting `execution` (or setting
`backend: sequential`) keeps today's single-process behavior.

```yaml
execution:
  backend: local     # sequential (default) | local | slurm | pbs | sge | lsf | htcondor
  n_workers: 4
```

- `local` runs an in-process Dask cluster (comparable to
  `ProcessPoolExecutor`/`joblib`) — good for a multi-core workstation.
- `slurm` / `pbs` / `sge` / `lsf` / `htcondor` submit one Dask worker job per
  HPC scheduler job via [`dask-jobqueue`](https://jobqueue.dask.org/), for
  running on a cluster. Pass scheduler-specific parameters (queue, cores,
  memory, walltime, ...) via `cluster_kwargs`:

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

Parallel backends require the optional `dask` (for `local`) or
`dask-jobqueue` (for the HPC backends) extras — see [Installation](#installation).
`custom_steps_folder`, if used, must be on a filesystem reachable from every
worker (trivially true for `local`; for `dask-jobqueue`, it must be a shared
filesystem also mounted on the compute nodes).

See `docs/dask_parallel_execution.md` for the full design rationale.

## Progress Tracking and Logging

The pipeline includes comprehensive progress tracking and logging features:

### Progress Bars

The sequential backend (today's default) shows a `rich` progress bar with:
- Spinner animation
- Progress across all recordings being processed, with percentage
- Time remaining estimate
- The recording currently being processed

Parallel backends (`local` and `dask-jobqueue`) instead log one line per
recording as it's submitted/completed/failed (a live, in-place-updating
progress bar can't meaningfully represent state changing in other
processes or on other machines), plus a Dask dashboard link for the
richer live view Dask itself provides.

### Logging

The pipeline uses MNE's logger for all output messages. You can:

**Console Output (default)**:
```bash
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02
```

**Log to File**:
```bash
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 \
    --log-file /path/to/logs/pipeline.log
```

**Adjust Logging Level**:
```bash
python src/cli.py \
    --bids-root /path/to/bids/dataset \
    --subjects 01 02 \
    --log-level DEBUG
```

Available log levels: `DEBUG`, `INFO` (default), `WARNING`, `ERROR`

The pipeline also saves a summary of results to `derivatives/meegflow/pipeline_results.json` for easy programmatic access.

## Docker Notes

### Volume Mounting

When using Docker, you need to mount your local directories to paths inside the container using the `-v` flag:

- **BIDS dataset**: Mount your BIDS root directory to `/data` or any path you specify with `--bids-root`
- **Configuration files**: Mount custom config files if not using the built-in configs in `/app/configs/`
- **Output directory**: The pipeline writes outputs to `<bids-root>/derivatives/meegflow/` by default
- **Log files**: If using `--log-file`, mount a directory for log output

### File Permissions

The Docker container runs as root by default. Files created by the container will be owned by root. To avoid permission issues:

1. Run with your user ID:
```bash
docker run --rm --user $(id -u):$(id -g) \
    -v /path/to/bids:/data \
    meegflow \
    --bids-root /data \
    --tasks rest
```

2. Or fix permissions after processing:
```bash
sudo chown -R $USER:$USER /path/to/bids/derivatives
```

### Using Built-in Configurations

The Docker image includes several pre-configured pipeline examples in `/app/configs/`:
- `/app/configs/config_example.yaml` - Standard pipeline with epochs
- `/app/configs/config_raw_only.yaml` - Raw data processing without epoching
- `/app/configs/config_with_adaptive_reject.yaml` - Advanced pipeline with concatenation and event-based epochs
- `/app/configs/config_minimal.yaml` - Comprehensive pipeline with strip_recording, ICA, and instance comparison
- `/app/configs/config_with_drop_bad_channels.yaml` - Pipeline using drop_bad_channels instead of interpolation
- `/app/configs/config_with_excluded_channels.yaml` - Pipeline demonstrating excluded_channels parameter
- `/app/configs/config_with_custom_steps.yaml` - Example template for using custom preprocessing steps
- `/app/configs/config_with_parallel_execution.yaml` - Example enabling parallel recording execution via Dask

Example using a built-in config:
```bash
docker run --rm \
    -v /path/to/bids:/data \
    meegflow \
    --bids-root /data \
    --tasks rest \
    --config /app/configs/config_with_adaptive_reject.yaml
```

### Building from Source

If you want to customize the Docker image or use a development version:

```bash
git clone https://github.com/Laouen/meegflow.git
cd meegflow
docker build -t meegflow:custom .
```

**Building in CI/CD environments with self-signed certificates:**

If you're building in a CI/CD environment with self-signed SSL certificates, use the `PIP_TRUSTED_HOST` build argument:

```bash
docker build --build-arg PIP_TRUSTED_HOST=1 -t meegflow:custom .
```

Note: This disables SSL verification for PyPI and should only be used in trusted CI/CD environments, not for production builds.

## Requirements

- Python >= 3.8
- mne >= 1.5.0
- mne-bids >= 0.14
- numpy >= 1.24.0
- scipy >= 1.11.0
- rich >= 13.0.0
- matplotlib >= 3.7.0 (recommended)
- pandas >= 2.0.0 (recommended)

Optional, for parallel execution (`pip install meegflow[dask]` /
`meegflow[dask-jobqueue]`):
- dask[distributed] >= 2024.1.0
- dask-jobqueue >= 0.8.2 (Slurm/PBS/SGE/LSF/HTCondor backends only)

## License

This project is ready to use for several projects and includes scripts for SLURM execution.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on the GitHub repository. 
