#!/usr/bin/env python3
"""
MEEGFlow: MEEG Preprocessing Pipeline using MNE-BIDS.

This module provides a modular, configuration-driven MEEG preprocessing pipeline.
Each preprocessing step is a separate function in the ``meegflow.steps`` package,
registered by name and dispatched against a :class:`~meegflow.context.PipelineContext`.
Steps can be customized and combined through a YAML configuration file, and custom
steps (loaded from ``custom_steps_folder``) plug into the same registry.

Main Components
---------------
- MEEGFlowPipeline: Core pipeline class
  - Processes MEEG data from BIDS datasets
  - Executes configurable preprocessing steps
  - Generates JSON and HTML reports
  - Supports batch processing of multiple subjects/sessions

Configuration
-------------
The pipeline is driven by YAML configuration files that specify:
- List of preprocessing steps to execute
- Parameters for each step
- Order of execution

Available Steps
---------------
Data I/O and Setup:
  - set_montage: Set electrode positions
  - drop_unused_channels: Remove specific channels

Filtering:
  - bandpass_filter: Apply high-pass and low-pass filters
  - notch_filter: Remove line noise

Preprocessing:
  - resample: Change sampling frequency
  - reference: Apply re-referencing
  - ica: ICA-based artifact removal

Bad Channel Detection:
  - find_flat_channels: Detect flat/disconnected channels
  - find_bads_channels_threshold: Threshold-based bad channel detection
  - find_bads_channels_variance: Variance-based detection
  - find_bads_channels_high_frequency: High-frequency noise detection

Bad Channel Handling:
  - interpolate_bad_channels: Repair bad channels via interpolation
  - drop_bad_channels: Remove bad channels permanently

Epoching:
  - find_events: Extract events from data
  - epoch: Create epochs around events
  - chunk_in_epoch: Create fixed-length epochs
  - find_bads_epochs_threshold: Detect and remove bad epochs

Output:
  - save_clean_instance: Save preprocessed data to .fif
  - generate_json_report: Create JSON report
  - generate_html_report: Create interactive HTML report

Utilities:
  - call_module: Dynamically call any importable function or object method.
    Supports positional args, data__ references to pipeline objects, method
    calls via ``target``, and multi-value unpacking via ``unpack_as``.
    See configuration reference for full details.

Usage Example
-------------
```python
from meegflow import MEEGFlowPipeline
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize and run pipeline
pipeline = MEEGFlowPipeline(
    bids_root='/path/to/bids',
    config=config
)
results = pipeline.run_pipeline(
    subjects=['01', '02'],
    tasks='rest'
)
```

See README.md for detailed documentation and examples.
"""


from __future__ import annotations

import os
os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path
from typing import Union, Dict, Any, List, Callable, TYPE_CHECKING
from mne.utils import logger
from mne_bids import BIDSPath
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import importlib.util
import sys
import inspect
from .context import PipelineContext
from .steps import STEP_REGISTRY

if TYPE_CHECKING:
    from .readers import DatasetReader


class MEEGFlowPipeline:
    def __init__(
        self, 
        reader: DatasetReader,
        output_root: Union[str, Path] = None, 
        config: Dict[str, Any] = None
    ):
        """Initialize MEEGFlow preprocessing pipeline.
        
        Parameters
        ----------
        reader : DatasetReader
            Reader instance for discovering data files. Use BIDSReader for BIDS datasets
            or GlobReader for custom directory structures.
        output_root : str or Path, optional
            Path to output derivatives root. If not provided, defaults to
            {dataset_root}/derivatives/meegflow
        config : dict, optional
            Configuration dictionary containing pipeline steps and parameters
        """
        self.config = config or {}
        self.output_root = Path(output_root) if output_root else None
        self.reader = reader

        # Built-in steps come from the registry (populated by importing the
        # steps package); custom steps may add to or override them by name.
        self.step_functions = dict(STEP_REGISTRY)

        # Load custom steps if folder is specified in config
        custom_steps_folder = self.config.get('custom_steps_folder')
        if custom_steps_folder:
            custom_steps = self._load_custom_steps(custom_steps_folder)
            self.step_functions.update(custom_steps)
            logger.info(f"Loaded {len(custom_steps)} custom step(s): {list(custom_steps.keys())}")

        # Validate pipeline steps if provided in config
        pipeline_cfg = self.config.get('pipeline', [])
        unknown = [s.get('name') for s in pipeline_cfg if s.get('name') not in self.step_functions]
        if unknown:
            raise ValueError(f"Unknown pipeline steps in config: {unknown}")

    @property
    def dataset_root(self) -> Path:
        """Get the dataset root path from the reader."""
        return self.reader.root

    def run_step(
        self,
        name: str,
        data: Union[Dict[str, Any], "PipelineContext"],
        config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Execute a single named step against a data mapping.

        Wraps ``data`` in a :class:`~meegflow.context.PipelineContext` (unless it
        already is one), dispatches the registered step, and returns the updated
        data mapping. Useful for running or testing one step in isolation.

        Parameters
        ----------
        name : str
            Registered step name (built-in or custom).
        data : dict or PipelineContext
            The shared data bag the step reads from / writes to.
        config : dict, optional
            Step configuration (everything except the ``name`` key).

        Returns
        -------
        dict
            The updated data mapping.
        """
        if name not in self.step_functions:
            raise ValueError(f"Unknown step '{name}'")

        if isinstance(data, PipelineContext):
            ctx = data
        else:
            ctx = PipelineContext(
                data,
                reader=self.reader,
                output_root=self.output_root,
                config=self.config,
            )

        result = self.step_functions[name](ctx, config or {})
        if isinstance(result, PipelineContext):
            ctx = result
        elif isinstance(result, dict):
            ctx.data = result
        return ctx.data

    def _load_custom_steps(self, custom_steps_folder: Union[str, Path]) -> Dict[str, Callable]:
        """
        Load custom preprocessing steps from Python files in the specified folder.
        
        This method discovers .py files in the custom_steps_folder and imports functions
        that follow the step function signature: func(data: Dict, step_config: Dict) -> Dict
        
        The function name will be used as the step name in the pipeline configuration.
        Custom steps can override built-in steps by using the same name.
        
        Parameters
        ----------
        custom_steps_folder : str or Path
            Path to folder containing Python files with custom step functions.
            
        Returns
        -------
        custom_steps : dict
            Dictionary mapping step names to their functions.
            
        Notes
        -----
        Custom step functions must:
        - Accept two parameters: data (Dict) and step_config (Dict)
        - Return a Dict (the updated data dictionary)
        - Be defined at module level (not inside classes)
        
        Example custom step file (my_steps.py):
        ```python
        def my_custom_filter(data, step_config):
            '''Apply custom filtering to raw data.'''
            if 'raw' not in data:
                raise ValueError("my_custom_filter requires 'raw' in data")
            
            # Get parameters from step_config
            cutoff_freq = step_config.get('cutoff_freq', 30.0)
            
            # Apply custom processing
            data['raw'].filter(h_freq=cutoff_freq, l_freq=None)
            
            # Record the step
            data['preprocessing_steps'].append({
                'step': 'my_custom_filter',
                'cutoff_freq': cutoff_freq
            })
            
            return data
        ```
        """
        custom_steps_folder = Path(custom_steps_folder)
        
        if not custom_steps_folder.exists():
            raise ValueError(f"Custom steps folder does not exist: {custom_steps_folder}")
        
        if not custom_steps_folder.is_dir():
            raise ValueError(f"Custom steps folder is not a directory: {custom_steps_folder}")
        
        custom_steps = {}
        python_files = list(custom_steps_folder.glob("*.py"))
        
        logger.info(f"Searching for custom steps in: {custom_steps_folder}")
        logger.info(f"Found {len(python_files)} Python file(s)")
        
        for py_file in python_files:
            # Skip __init__.py and files starting with underscore
            if py_file.name.startswith('_'):
                logger.debug(f"Skipping {py_file.name}")
                continue
                
            try:
                # Create a unique module name to avoid conflicts
                module_name = f"custom_steps.{py_file.stem}"
                
                # Load the module
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    logger.warning(f"Could not load module spec for {py_file}")
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # Find all functions in the module that match the step signature
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    # Skip private functions
                    if name.startswith('_'):
                        continue
                    
                    # Check function signature
                    sig = inspect.signature(obj)
                    params = list(sig.parameters.keys())
                    
                    # Step functions should accept exactly 2 parameters: data and step_config
                    if len(params) == 2:
                        custom_steps[name] = obj
                        logger.info(f"Loaded custom step '{name}' from {py_file.name}")
                    else:
                        logger.debug(f"Skipping function '{name}' in {py_file.name} - "
                                   f"expected 2 parameters, found {len(params)}")
                        
            except Exception as e:
                logger.error(f"Error loading custom steps from {py_file}: {e}")
                # Continue loading other files even if one fails
                continue
        
        if not custom_steps:
            logger.warning(f"No valid custom steps found in {custom_steps_folder}")
        
        return custom_steps


    def _get_pipeline_steps(self) -> List[Dict[str, Any]]:
        """Retrieve the list of pipeline steps from the configuration."""
        pipeline_steps = self.config.get('pipeline', [])

        if not pipeline_steps:
            raise ValueError(
                "No pipeline steps provided in configuration. "
                "Please specify a 'pipeline' list in your config file with at least one preprocessing step."
            )
    
        return pipeline_steps



    























    def _process_single_recording(
        self, 
        paths: List[Union[BIDSPath, Path]], 
        metadata: Dict[str, Any],
        progress: Progress = None,
        io_backend: str = 'read_raw_bids',
        task_id: int = None
    ) -> Dict[str, Any]:
        """Process a single recording using the configured pipeline steps.
        
        Parameters
        ----------
        paths : list of BIDSPath or Path
            List of file paths to process together
        metadata : dict
            Metadata dictionary with keys like 'subject', 'task', 'session', 'acquisition'
        progress : Progress, optional
            Rich progress bar instance
        task_id : int, optional
            Progress task ID for updating progress
            
        Returns
        -------
        results : dict
            Dictionary containing processing results
        """
        # Initialize data dictionary with metadata
        data = {
            'subject': metadata.get('subject'),
            'task': metadata.get('task'),
            'session': metadata.get('session'),
            'acquisition': metadata.get('acquisition'),
            'preprocessing_steps': []
        }

        # Read data files
        logger.info(f"Reading data from:")
        for path in paths:
            logger.info(f"  - {path}")

        # Read all files (loading is delegated to the reader)
        data['all_raw'] = self.reader.read(paths, io_backend=io_backend)

        # Wrap the shared data bag in a context exposing the step services.
        ctx = PipelineContext(
            data,
            reader=self.reader,
            output_root=self.output_root,
            config=self.config,
        )

        # Get pipeline steps from config
        pipeline_steps = self._get_pipeline_steps()

        # Execute each step in order
        for step_idx, step in enumerate(pipeline_steps):
            step_name = step.get('name')
            if step_name not in self.step_functions:
                raise ValueError(f"Unknown step '{step_name}' in pipeline execution")

            # Update progress for this step
            if progress and task_id is not None:
                progress.update(task_id, description=f"[cyan]Step: {step_name}", completed=step_idx)

            logger.info(f"Executing step: {step_name}")

            # Execute the step with its configuration
            step_config = {k: v for k, v in step.items() if k != 'name'}
            result = self.step_functions[step_name](ctx, step_config)
            # Steps mutate the context in place; for backwards compatibility a
            # step may also return the context or a plain data mapping.
            if isinstance(result, PipelineContext):
                ctx = result
            elif isinstance(result, dict):
                ctx.data = result

        data = ctx.data

        # Mark as complete
        if progress and task_id is not None:
            progress.update(task_id, completed=len(pipeline_steps))

        # Prepare results
        results = {
            'subject': data.get('subject'),
            'task': data.get('task'),
            'session': data.get('session'),
            'acquisition': data.get('acquisition'),
            'raw_files': [str(p) for p in paths],
        }

        # Copy relevant output information to results
        for key in ['raw_file', 'epochs_file', 'json_report', 'html_report', 'n_epochs', 'preprocessing_steps']:
            if key in data:
                results[key] = data[key]

        logger.info(f"Successfully processed {data.get('subject')} - {data.get('session')} - {data.get('task')} - {data.get('acquisition')}")
        return results

    def run_pipeline(
        self,
        subjects: Union[str, List[str]] = None,
        sessions: Union[str, List[str]] = None,
        tasks: Union[str, List[str]] = None,
        acquisitions: Union[str, List[str]] = None,
        runs: Union[str, List[str]] = None,
        extension: str = '.vhdr',
        io_backend: str = 'read_raw_bids'
    ) -> Dict[str, Any]:
        """Run the pipeline using the configured reader to find files.

        Parameters
        ----------
        subjects : str | list of str | None
            Subject ID(s) to process. None matches all subjects.
        sessions : str | list of str | None
            Session ID(s) to process. None matches all sessions.
        tasks : str | list of str | None
            Task(s) to process. None matches all tasks.
        acquisitions : str | list of str | None
            Acquisition parameter(s). None matches all acquisitions.
        runs : str | list of str | None
            Run ID(s) to process. None matches all runs.
        extension : str
            File extension to match (default: ``'.vhdr'``).
        io_backend : str
            MNE IO function used to read each file (default:
            ``'read_raw_bids'``). Any function name resolvable via
            ``mne.io`` can be supplied (e.g. ``'read_raw_eeglab'``).

        Returns
        -------
        all_results : dict
            Dictionary mapping recording name -> result dict. Each result
            contains the keys set by whichever output steps ran (e.g.
            ``'raw_file'``, ``'epochs_file'``, ``'json_report'``,
            ``'html_report'``), or an ``'error'`` key with the exception if
            processing failed.
        """
        recordings = self.reader.find_recordings(
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            acquisitions=acquisitions,
            runs=runs,
            extension=extension
        )
        
        logger.info(f"Found {len(recordings)} recording(s) to process")

        all_results = {}

        # Create progress bars for matched paths and preprocessing steps
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:

            # Overall progress for all recordings
            overall_task = progress.add_task(
                "[green]Processing recordings", 
                total=len(recordings)
            )

            for i, recording in enumerate(recordings):
                # Extract metadata and paths from the recording
                paths = recording['paths']
                metadata = recording['metadata']
                recording_name = recording['recording_name']

                # Get pipeline steps for this recording's progress bar
                pipeline_steps = self._get_pipeline_steps()
                
                # Create a task for the current recording's steps
                step_task_id = progress.add_task(
                    f"[cyan]{recording_name}", 
                    total=len(pipeline_steps)
                )

                try:
                    results = self._process_single_recording(
                        paths=paths,
                        metadata=metadata,
                        progress=progress,
                        io_backend=io_backend,
                        task_id=step_task_id
                    )

                    # Use subject from metadata if available, otherwise use first available key
                    subject_key = metadata.get('subject', list(metadata.values())[0] if metadata else 'unknown')
                    all_results.setdefault(subject_key, []).append(results)
                    logger.info(f"Successfully completed {recording_name}")
                except Exception as exc:
                    # Do not stop the whole batch if one subject fails; capture the error
                    logger.error(f"Error processing {recording_name}: {str(exc)}")
                    subject_key = metadata.get('subject', list(metadata.values())[0] if metadata else 'unknown')
                    all_results.setdefault(subject_key, []).append({'error': str(exc)})
                    # Continue processing the remaining recordings; the failure
                    # is captured in all_results and summarised by the caller.
                finally:
                    # Remove the step task after this recording is done
                    progress.remove_task(step_task_id)
                
                # Update overall progress
                progress.update(overall_task, completed=i+1)

        logger.info(f"Pipeline completed.")
        return all_results