# Python API

MEEGFlow can also be used programmatically, which is useful for scripting, notebooks, or integrating the pipeline into larger workflows.

## Basic usage

```python
import yaml
from meegflow import MEEGFlowPipeline
from meegflow.readers import BIDSReader

# Load config
with open("configs/config_example.yaml") as f:
    config = yaml.safe_load(f)

# Create a reader
reader = BIDSReader("/data/my_study")

# Build and run the pipeline
pipeline = MEEGFlowPipeline(reader=reader, config=config)
pipeline.run_pipeline(subjects=["01", "02"], tasks=["rest"])
```

By default recordings are processed sequentially, in this same process. To
process them in parallel via Dask instead, add an `execution` block to
`config` — see [Parallel Execution](parallel-execution.md).

## Using a GlobReader

For datasets that don't follow strict BIDS naming you can use `GlobReader` with a glob pattern that contains `{subject}`, `{session}`, `{task}`, etc. placeholders:

```python
from meegflow.readers import GlobReader

reader = GlobReader(
    root="/data/my_study",
    pattern="sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr",
)

pipeline = MEEGFlowPipeline(reader=reader, config=config)
pipeline.run_pipeline()
```

## Custom steps

Place a Python file anywhere and pass its directory to the pipeline. The file must contain functions with the signature `(data, step_config) -> data`:

```python
# my_steps.py
def my_custom_filter(data, step_config):
    cutoff = step_config.get("cutoff", 1.0)
    data["raw"].filter(l_freq=cutoff, h_freq=None)
    return data
```

```yaml
# config.yaml
pipeline:
  - name: my_custom_filter
    cutoff: 2.0
```

```python
pipeline = MEEGFlowPipeline(
    reader=reader,
    config=config,
    custom_steps_folder="./my_steps.py",
)
```
