# GitHub Copilot / AI agent instructions for nice-preprocessing

Quick orientation (what matters):

- **Project purpose:** configuration-driven EEG preprocessing built on MNE / MNE-BIDS. Main entry points are the CLI (`src/cli.py`) and the pipeline class (`src/eeg_preprocessing_pipeline.py`).
- **Config-first design:** preprocessing pipelines are specified in YAML files as a `pipeline:` list of steps (each step is a dict with `name` + parameters). The repository ships no config files; see the examples in `README.md` and `docs/usage/examples.md`.
- **Console script:** installing the package (`pip install -e .`) registers `eeg-preprocess` (see `setup.py` entry_points). The CLI also has `python src/cli.py` usage (see `README.md` examples).

How the pipeline is structured (big picture):

- `EEGPreprocessingPipeline` (in `src/eeg_preprocessing_pipeline.py`) drives processing:
  - It maps step names to methods in `self.step_functions` (each `_step_<name>(data, step_config)` returns updated `data`).
  - The pipeline expects a mutable `data` dict carrying items like `'raw'`, `'epochs'`, `'events'`, `'preprocessing_steps'` and writeable outputs such as `json_report` / `html_report`.
  - New steps: add an entry to `step_functions` and implement `_step_<name>(data, step_config)` following existing patterns.

Important patterns & conventions (be specific):

- Config shape: `pipeline: - name: load_data - name: bandpass_filter l_freq: 0.5 h_freq: 40.0 ...` — the code strips `name` and passes the rest as `step_config`.
- Step function signature: always accept `(data: Dict, step_config: Dict)` and should append a descriptive dict to `data['preprocessing_steps']` summarizing parameters and effects.
- Input/Output instances: steps operate on named instances `'raw'` or `'epochs'`. Many steps check `if '<instance>' not in data: raise ValueError(...)` — preserve this behavior.
- Picks handling: `_get_picks(info, picks_params)` expects either channel-type list (e.g., `['eeg']`) or None and returns MNE picks. Use it rather than manually creating picks.
- Error handling: individual recording failures are logged and collected in results; however `_process_single_recording` currently re-raises exceptions after recording-level handling — be cautious when changing behavior if you want fully resilient batch runs.

Integration points & external deps:

- Uses `mne`, `mne-bids` (BIDSPath + `read_raw_bids`), `rich` for progress bars, and `matplotlib` for reports. See `requirements.txt`.
- HTML report utilities are in `src/report.py` (helpers: `collect_bad_channels_from_steps`, `create_bad_channels_topoplot`, `create_preprocessing_steps_table`). Use these helpers to keep report logic consistent.

Developer workflows (how I usually run & debug):

- Install dev editable environment and dependencies:
  ```bash
  pip install -r requirements.txt
  pip install -e .
  ```
- Run pipeline locally (example):
  ```bash
  python src/cli.py --bids-root /path/to/bids --subjects 01 --tasks rest --config config.yaml
  # or after install
  eeg-preprocess --bids-root /path/to/bids --tasks rest --config config.yaml
  ```
- Run tests: `pytest tests/` (project includes unit/integration tests under `tests/`).

How to add or modify a preprocessing step (concrete example):

1. Add method name to `self.step_functions` in `EEGPreprocessingPipeline.__init__` e.g. `'my_step': self._step_my_step`.
2. Implement `_step_my_step(self, data, step_config)` in `src/eeg_preprocessing_pipeline.py`.
   - Validate required instances (`'raw'` or `'epochs'`) and raise a clear ValueError if missing.
   - Mutate `data` in place (e.g., set `data['epochs'] = ...`) and append a summary dict into `data['preprocessing_steps']`.
   - Return `data` at the end.
3. If helpful, add an example configuration to `docs/usage/examples.md` and update `README.md` usage examples.

Files to inspect for examples and conventions:

- `src/eeg_preprocessing_pipeline.py` — core pipeline and all existing `_step_*` implementations
- `src/cli.py` — CLI argument handling, logging setup, JSON result file location
- `src/report.py` — helpers used to build HTML reports
- `docs/usage/examples.md` — example YAML configurations
- `README.md` — full usage and output structure

Gotchas and discovered behaviors to watch for:

- Default file extension: CLI default is `--extension .vhdr` and pipeline uses that in BIDSPath matching. If your dataset uses a different extension, pass it explicitly.
- `save_clean_instance` uses `BIDSPath(..., suffix='epo', extension='.fif')` and writes into `derivatives/nice_preprocessing/<instance>/` — verify the `datatype` and suffix when adapting saves.
- Pipeline validation: `EEGPreprocessingPipeline.__init__` will raise if a config contains unknown step names — keep `step_functions` in sync with configs.

If you want me to expand examples (e.g., a minimal step template, or a checklist for adding tests), tell me which you'd prefer and I will add it.
