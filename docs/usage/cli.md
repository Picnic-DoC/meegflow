# Command-Line Interface

After installation, the `meegflow` command is available on your PATH.

## Synopsis

```
meegflow --bids-root PATH --config PATH [OPTIONS]
```

## Options

| Option | Type | Description |
|--------|------|-------------|
| `--bids-root` | path | Root directory of the BIDS dataset (**required**) |
| `--config` | path | Path to the YAML pipeline configuration file (**required**) |
| `--subjects` | list | Subject ID(s) to process (default: all) |
| `--sessions` | list | Session ID(s) to process (default: all) |
| `--tasks` | list | Task name(s) to process (default: all) |
| `--acquisitions` | list | Acquisition label(s) to process (default: all) |
| `--runs` | list | Run number(s) to process (default: all) |
| `--extension` | str | File extension filter (default: `.fif` when the datatype is `meg`, `.vhdr` otherwise) |
| `--datatype` | str | BIDS datatype to process: `eeg`, `meg`, `ieeg` or `nirs` (BIDS reader only; defaults to the config's top-level `datatype`, otherwise detected from the dataset) |
| `--log-file` | path | Write MNE log to this file (appended) |
| `--log-level` | str | MNE log level (`DEBUG`, `INFO`, `WARNING`, …). Default `INFO` |

## Examples

### Process all subjects

```bash
meegflow \
    --bids-root /data/my_study \
    --config config.yaml
```

### Process specific subjects and tasks

```bash
meegflow \
    --bids-root /data/my_study \
    --config config.yaml \
    --subjects 01 02 03 \
    --tasks rest
```

### Save log to file

```bash
meegflow \
    --bids-root /data/my_study \
    --config config.yaml \
    --log-file logs/preproc.log \
    --log-level DEBUG
```
