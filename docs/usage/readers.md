# Reader System Documentation

The meegflow pipeline now supports two types of file readers:

1. **BIDSReader** - For BIDS-formatted datasets (default)
2. **GlobReader** - For custom directory structures using glob patterns with variable extraction

## BIDSReader (Default)

The BIDS reader uses MNE-BIDS to discover files in BIDS-formatted datasets.

### Command-Line Usage

```bash
# Basic usage (--reader bids is the default, so it can be omitted)
python src/cli.py --bids-root /path/to/bids --config config.yaml

# Explicit BIDS reader specification
python src/cli.py --reader bids --bids-root /path/to/bids --tasks rest

# With subject/task filtering
python src/cli.py --bids-root /path/to/bids --subjects 01 02 --tasks rest

# Using the meegflow command (after pip install -e .)
meegflow --bids-root /path/to/bids --config config.yaml
```

### Programmatic Usage

```python
from meegflow import MEEGFlowPipeline
from readers import BIDSReader

# Create a BIDS reader
reader = BIDSReader('/path/to/bids')

# Initialize pipeline with the reader
pipeline = MEEGFlowPipeline(
    reader=reader,
    config=config
)

# Run the pipeline
results = pipeline.run_pipeline(
    subjects=['01', '02'],
    tasks='rest'
)
```

### Selecting the datatype

The BIDS reader searches one datatype at a time: `eeg`, `meg`, `ieeg` or
`nirs`. If you do not name one, it is detected from the dataset with
`mne_bids.get_datatypes`, with `eeg` preferred whenever the dataset contains
EEG. Name it explicitly to process another datatype, or to disambiguate a
dataset that holds several:

```bash
meegflow --bids-root /path/to/bids --datatype meg --extension .fif --config config.yaml
```

```python
reader = BIDSReader('/path/to/bids', datatype='meg')
```

The BIDS suffix follows the datatype (`sub-01_task-rest_meg.fif`), and can be
overridden with the reader's `suffix` argument. `--extension` is not derived
from the datatype: MEG recordings in FIF format need `--extension .fif`, since
the default is the BrainVision `.vhdr`.

## GlobReader

The glob reader allows you to work with custom directory structures by specifying a glob pattern with variable placeholders.

### Pattern Syntax

Variables are specified using `{variable_name}` syntax, which:
- Converts to `*` wildcards for file matching
- Extracts the matched values and assigns them to the variable name

### Supported Variables

Standard BIDS entities are recognized for filtering:
- `{subject}` - Subject identifier
- `{session}` - Session identifier  
- `{task}` - Task name
- `{acquisition}` - Acquisition parameters

You can also use custom variable names, but filtering will only work with the standard entities listed above.

### Command-Line Usage

```bash
# Basic glob reader usage
python src/cli.py \
    --reader glob \
    --data-root /path/to/data \
    --glob-pattern "sub-{subject}/ses-{session}/eeg/sub-{subject}_task-{task}_eeg.vhdr"

# With filtering
python src/cli.py \
    --reader glob \
    --data-root /path/to/data \
    --glob-pattern "sub-{subject}/ses-{session}/eeg/sub-{subject}_task-{task}_eeg.vhdr" \
    --subjects 01 02 \
    --tasks rest
    
# Custom structure example
python src/cli.py \
    --reader glob \
    --data-root /home/user/eeg_data \
    --glob-pattern "participants/{subject}/recordings/session_{session}/{task}.vhdr" \
    --subjects 001 002 \
    --tasks baseline memory
```

### Programmatic Usage

```python
from meegflow import MEEGFlowPipeline
from readers import GlobReader

# Create a glob reader with your custom pattern
reader = GlobReader(
    data_root='/path/to/data',
    pattern='sub-{subject}/ses-{session}/eeg/sub-{subject}_task-{task}_eeg.vhdr'
)

# Initialize the pipeline with the glob reader
pipeline = MEEGFlowPipeline(
    reader=reader,
    config=config
)

# Run the pipeline
results = pipeline.run_pipeline(
    subjects=['01', '02'],
    tasks='rest'
)
```

## Pattern Examples

### Example 1: BIDS-like structure
```
Pattern: "sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr"

Matches:
  sub-01/ses-01/eeg/sub-01_ses-01_task-rest_eeg.vhdr
  sub-02/ses-02/eeg/sub-02_ses-02_task-memory_eeg.vhdr

Extracts:
  subject: 01, 02
  session: 01, 02
  task: rest, memory
```

### Example 2: Simple subject/task structure
```
Pattern: "data/{subject}/{task}.vhdr"

Matches:
  data/P001/baseline.vhdr
  data/P002/memory.vhdr

Extracts:
  subject: P001, P002
  task: baseline, memory
```

### Example 3: Date-based structure
```
Pattern: "recordings/{date}/participant_{subject}/{task}_recording.vhdr"

Matches:
  recordings/2024-01-15/participant_001/rest_recording.vhdr
  recordings/2024-01-16/participant_002/memory_recording.vhdr

Extracts:
  date: 2024-01-15, 2024-01-16
  subject: 001, 002
  task: rest, memory

Note: date is extracted but not used for filtering (only standard entities)
```

### Example 4: Multiple files per recording
```
Pattern: "studies/{study}/sub-{subject}/*_task-{task}_eeg.vhdr"

Matches:
  studies/study1/sub-01/run-1_task-rest_eeg.vhdr
  studies/study1/sub-01/run-2_task-rest_eeg.vhdr

Extracts and groups:
  All files for sub-01, task-rest are processed together
```

## Variable Repetition

If a variable appears multiple times in the pattern, the glob reader ensures consistency:

```python
# Pattern with repeated variables
pattern = "sub-{subject}/sub-{subject}_task-{task}_eeg.vhdr"

# Only matches if both instances of {subject} contain the same value
# Matches: sub-01/sub-01_task-rest_eeg.vhdr ✓
# Doesn't match: sub-01/sub-02_task-rest_eeg.vhdr ✗
```

## Choosing the Right Reader

### Use BIDSReader when:
- Your data is already in BIDS format
- You want automatic BIDS validation
- You need integration with other BIDS tools
- You want to leverage MNE-BIDS features

### Use GlobReader when:
- Your data has a custom directory structure
- You're migrating legacy data
- You have constraints that prevent BIDS conversion
- You need flexible pattern matching

## Migration from Legacy Code

If you have existing code using `MEEGFlowPipeline`, you need to update it to pass a reader:

```python
# Old code (no longer works)
# pipeline = MEEGFlowPipeline(bids_root='/path/to/bids', config=config)

# New code - create a reader first
from readers import BIDSReader
reader = BIDSReader('/path/to/bids')
pipeline = MEEGFlowPipeline(reader=reader, config=config)
results = pipeline.run_pipeline(subjects=['01'], tasks='rest')
```

To use glob reader instead, simply create a GlobReader and pass it:

```python
from readers import GlobReader

reader = GlobReader('/path/to/data', 'your/pattern/here/{subject}_{task}.vhdr')
pipeline = MEEGFlowPipeline(
    reader=reader,
    config=config
)
results = pipeline.run_pipeline(subjects=['01'], tasks='rest')
```

## Troubleshooting

### No files found

If the glob reader doesn't find files:

1. Check the pattern matches your actual directory structure
2. Use absolute paths or ensure you're in the correct working directory
3. Verify variable names in the pattern match the entities you're filtering by
4. Check file extensions match (default is `.vhdr`)

Example debugging:

```python
reader = GlobReader('/path/to/data', 'sub-{subject}/*_task-{task}_eeg.vhdr')

# Check what the glob pattern looks like
print(f"Glob pattern: {reader.glob_pattern}")
# Output: sub-*/*_task-*_eeg.vhdr

# Check what variables are extracted
print(f"Variables: {reader.variable_names}")
# Output: ['subject', 'task']

# Find all recordings (no filtering)
recordings = reader.find_recordings()
print(f"Found {len(recordings)} recordings")

# Check what was extracted from the first recording
if recordings:
    print(f"First recording metadata: {recordings[0]['metadata']}")
```

### Variables not recognized for filtering

Only these standard entity names are recognized for filtering:
- `subject`
- `session`
- `task`
- `acquisition`

Custom variable names will be extracted but won't filter results. Rename your variables to match these standard names if you need filtering.
