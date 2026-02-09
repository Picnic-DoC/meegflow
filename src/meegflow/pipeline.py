#!/usr/bin/env python3
"""
MEEGFlow: MEEG Preprocessing Pipeline using MNE-BIDS.

This module provides a modular, configuration-driven MEEG preprocessing pipeline.
Each preprocessing step is a separate method that can be customized and combined
through a YAML configuration file.

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

from itertools import product
from pathlib import Path
from typing import Union, Dict, Any, List, Callable, TYPE_CHECKING
import json
import mne
from mne.utils import logger
import numpy as np
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from . import adaptive_reject
from collections import defaultdict
from .utils import NpEncoder
import matplotlib.pyplot as plt
import importlib.util
import sys
import inspect
from .readers import BIDSReader

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

        # Map step names to their corresponding methods
        self.step_functions = {
            'strip_recording': self._step_strip_recording,
            'concatenate_recordings': self._step_concatenate_recordings,
            'copy_instance': self._step_copy_instance,
            'set_montage': self._step_set_montage,
            'drop_unused_channels': self._step_drop_unused_channels,
            'bandpass_filter': self._step_bandpass_filter,
            'notch_filter': self._step_notch_filter,
            'resample': self._step_resample,
            'reference': self._step_reference,
            'interpolate_bad_channels': self._step_interpolate_bad_channels,
            'drop_bad_channels': self._step_drop_bad_channels,
            'ica': self._step_ica,
            'find_events': self._step_find_events,
            'epoch': self._step_epoch,
            'chunk_in_epoch': self._step_chunk_in_epoch,
            'find_flat_channels': self._step_find_flat_channels,
            'find_bads_channels_threshold': self._step_find_bads_channels_threshold,
            'find_bads_channels_variance': self._step_find_bads_channels_variance,
            'find_bads_channels_high_frequency': self._step_find_bads_channels_high_frequency,
            'find_bads_epochs_threshold': self._step_find_bads_epochs_threshold,
            'save_clean_instance': self._step_save_clean_instance,
            'generate_json_report': self._step_generate_json_report,
            'generate_html_report': self._step_generate_html_report,
        }

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
        """Get the dataset root path from the reader.
        
        Returns the reader's root directory, which may be bids_root or data_root
        depending on the reader type.
        """
        if hasattr(self.reader, 'bids_root'):
            return self.reader.bids_root
        elif hasattr(self.reader, 'data_root'):
            return self.reader.data_root
        else:
            raise AttributeError("Reader does not have a bids_root or data_root attribute")
    
    def _get_derivatives_root(self, subdir: str = "") -> Path:
        """Get the derivatives root directory.
        
        Parameters
        ----------
        subdir : str, optional
            Subdirectory within derivatives/meegflow
            
        Returns
        -------
        Path
            Path to derivatives directory
        """
        if self.output_root:
            base = self.output_root
        else:
            base = self.dataset_root / "derivatives" / "meegflow"
        
        if subdir:
            return base / subdir
        return base

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

    def _build_include_patterns(
        self,
        subjects: List[str] = None,
        sessions: List[str] = None
    ) -> Union[str, List[str]]:
        """Build include_match patterns to narrow BIDS entity search.
        
        Parameters
        ----------
        subjects : list of str, optional
            Known subject values to narrow the search.
        sessions : list of str, optional
            Known session values to narrow the search. If sessions were discovered
            (not explicitly provided), patterns will include both session and 
            non-session directories to handle subjects with and without sessions.
        
        Returns
        -------
        str or list of str
            Pattern(s) to use with get_entity_vals include_match parameter.
        """
        if subjects:
            subjects = [s if s is not None else '*' for s in subjects]
        if sessions:
            sessions = [s if s is not None else '*' for s in sessions]
        
        # If we have both subjects and sessions, create specific patterns
        if subjects and sessions:
            patterns = []
            # Add patterns for subjects with sessions
            for sub in subjects:
                for ses in sessions:
                    patterns.append(f'sub-{sub}/ses-{ses}/')
            # Also add patterns without sessions to catch subjects that don't use sessions
            # This is important because get_entity_vals only returns sessions that exist,
            # so we need to also search for files without the session entity
            for sub in subjects:
                patterns.append(f'sub-{sub}/')
            return patterns
        
        # If we only have subjects, create subject-specific patterns
        if subjects:
            return [f'sub-{sub}/' for sub in subjects]
        
        # If we only have sessions, we still need to search all subjects
        # but can narrow to specific sessions
        if sessions:
            patterns = []
            for ses in sessions:
                patterns.append(f'sub-*/ses-{ses}/')
            return patterns
        
        # Default: search all subject directories
        return 'sub-*/'

    def _get_entity_values(
        self, 
        entity_key: str, 
        entity_value: any,
        subjects: List[str] = None,
        sessions: List[str] = None
    ) -> List[Union[str, None]]:
        """Get all unique values for a given BIDS entity in the dataset.
        
        Parameters
        ----------
        entity_key : str
            The BIDS entity key (e.g., 'subject', 'task', 'session', 'acquisition').
        entity_value : str | list of str | None
            The entity value(s) to process. If None, discovers all existing values
            from the BIDS dataset. If a string, returns it as a single-element list.
            If a list, returns it as-is.
        subjects : list of str, optional
            Known subject values to narrow the search. Only used when entity_value is None.
        sessions : list of str, optional
            Known session values to narrow the search. Only used when entity_value is None.
        
        Returns
        -------
        list of str or [None]
            List of entity values to process. Returns [None] if entity_value is None
            and no values are found in the dataset.
        """
        if isinstance(entity_value, str):
            return [entity_value]
    
        if isinstance(entity_value, list):
            return entity_value

        if entity_value is None:
            # Build include_match pattern based on known entity values to narrow search
            include_patterns = self._build_include_patterns(subjects, sessions)
            
            # Use get_entity_vals to find all existing values for this entity
            all_values = get_entity_vals(
                root=self.dataset_root,
                entity_key=entity_key,
                include_match=include_patterns
            )
            # Return the list of values, or [None] if no values found
            return list(all_values) if all_values else [None]

        raise ValueError(f"Invalid type for entity '{entity_key}': {type(entity_value)}")

    def _find_events_from_raw(self, raw, get_events_from='annotations', shortest_event=1, event_id='auto', stim_channel=None):
        """Extract events from a Raw object using annotations or a stim channel.

        Args:
            raw: MNE Raw object to extract events from.
            get_events_from: Source of events — ``'annotations'`` (default) or
                ``'stim_channel'``.
            shortest_event: Minimum number of samples for an event. Used only
                when ``get_events_from='stim_channel'``. Default 1.
            event_id: Event ID mapping passed to
                ``mne.events_from_annotations``. Default ``'auto'``.
            stim_channel: Stimulus channel name. Required when
                ``get_events_from='stim_channel'``.

        Returns:
            Tuple of ``(events, event_id_map)`` where ``events`` is a NumPy
            array of shape ``(n_events, 3)`` and ``event_id_map`` is a dict
            mapping event names to integer codes (``None`` for stim-channel
            mode).

        Raises:
            ValueError: If ``get_events_from`` is not a recognised method.
        """
        if get_events_from == 'stim_channel':
            events = mne.find_events(
                raw,
                shortest_event=shortest_event,
                stim_channel=stim_channel,
                verbose=False
            )
            return events, None
        
        if get_events_from == 'annotations':
            events, found_event_id = mne.events_from_annotations(raw, event_id=event_id)
            return events, found_event_id

        raise ValueError(f"Invalid get_events_from method: {get_events_from}")

    def _get_pipeline_steps(self) -> List[Dict[str, Any]]:
        """Retrieve the list of pipeline steps from the configuration."""
        pipeline_steps = self.config.get('pipeline', [])

        if not pipeline_steps:
            raise ValueError(
                "No pipeline steps provided in configuration. "
                "Please specify a 'pipeline' list in your config file with at least one preprocessing step."
            )
    
        return pipeline_steps

    def _apply_excluded_channels(self, info: mne.Info, picks: List[int], excluded_channels: List[str] = None) -> List[int]:
        """
        Auxiliary function to exclude specific channels from picks.
        
        This function allows excluding channels (e.g., reference channels like 'Cz') 
        from analysis steps where it makes sense, to avoid reference problems.
        
        Parameters
        ----------
        info : mne.Info
            MNE info object containing channel information
        picks : list of int
            Channel indices to filter
        excluded_channels : list of str, optional
            List of channel names to exclude from picks
            
        Returns
        -------
        picks : list of int
            Filtered channel indices with excluded channels removed
        """
        if excluded_channels is None or len(excluded_channels) == 0:
            return picks
            
        # Get channel names for the picks
        ch_names = [info['ch_names'][pick] for pick in picks]
        
        # Filter out excluded channels
        filtered_picks = [pick for pick, ch_name in zip(picks, ch_names) 
                         if ch_name not in excluded_channels]
        
        logger.info(f"Excluding channels: {excluded_channels}. "
                   f"Reduced from {len(picks)} to {len(filtered_picks)} channels.")
        
        return filtered_picks

    def _get_picks(self, info: mne.Info, picks_params: Any, excluded_channels: List[str] = None) -> List[int]:
        """
        Get channel picks with optional exclusion of specific channels.
        
        Parameters
        ----------
        info : mne.Info
            MNE info object containing channel information
        picks_params : list, tuple, or None
            Channel type specification (e.g., ['eeg'], ['eeg', 'eog'])
        excluded_channels : list of str, optional
            List of channel names to exclude from picks
            
        Returns
        -------
        picks : list of int
            Channel indices, excluding 'bads' and any specified excluded_channels
        """
        # Compute picks if provided, otherwise return all EEG channels
        if isinstance(picks_params, (list, tuple)):
            picks = mne.pick_types(
                info,
                exclude='bads',
                **{ch_type: True for ch_type in picks_params}
            )
        else:
            picks = mne.pick_types(
                info,
                exclude='bads',
                eeg=True,
                eog=False,
                meg=False
            )
        
        # Apply excluded_channels filter
        picks = self._apply_excluded_channels(info, picks, excluded_channels)
        
        return picks

    def _step_strip_recording(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        
        instance = step_config.get('instance', 'raw')
        start_padding = step_config.get('start_padding', 1)
        end_padding = step_config.get('end_padding', 1)
        get_events_from = step_config.get('get_events_from', 'annotations')
        shortest_event = step_config.get('shortest_event', 1)
        event_id = step_config.get('event_id', 'auto')
        

        if instance not in data:
            raise ValueError(f"strip recordings step requires '{instance}' to be present in data (either 'all_raw', 'raw')")

        # TODO: improve this and make it general to all corresponding steps        
        all_instances = data[instance]
        if not isinstance(all_instances, list):
            all_instances = [all_instances]

        for i, inst in enumerate(all_instances):
            events, _ = self._find_events_from_raw(
                inst,
                get_events_from=get_events_from,
                shortest_event=shortest_event,
                event_id=event_id
            )
            
            start = inst.times[events[0,0]] - start_padding
            end = inst.times[events[-1,0]] + end_padding

            start = max(0, start)
            end = min(inst.times[-1], end)
            
            inst.crop(start, end)
            
            data['preprocessing_steps'].append({
                'step': 'strip_recording',
                'instance': f'{instance}-{i}',
                'start': start,
                'end': end
            })
        
        return data
    
    def _step_concatenate_recordings(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        if 'all_raw' not in data:
            raise ValueError("notch_filter requires 'all_raw' in data")

        if len(data['all_raw']) > 1:
            data['raw'] = mne.concatenate_raws(data['all_raw'])
        else:
            data['raw'] = data['all_raw'][0]

        data['preprocessing_steps'].append({
            'step': 'concatenate_recordings',
        })
        
        return data

    def _step_copy_instance(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        from_instance = step_config.get('from_instance', 'raw')
        to_instance = step_config.get('to_instance', 'raw_cleaned')

        if from_instance not in data:
            raise ValueError(f"copy_instance step requires '{from_instance}' to be in data")

        data[to_instance] = data[from_instance].copy()
        data['preprocessing_steps'].append({
            'step': 'copy_instance',
            'from_instance': from_instance,
            'to_instance': to_instance
        })

        return data

    def _step_set_montage(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set channel montage for MEEG data.
        
        Useful when data lacks electrode position information. Sets standard electrode
        positions based on the specified montage.
        
        Parameters (via step_config)
        -----------------------------
        montage : str, optional
            Name of standard montage to use (default: 'standard_1020').
            Examples: 'standard_1020', 'standard_1005', 'biosemi64', etc.
            See MNE documentation for available montages.
        
        Updates
        -------
        data['raw'] : mne.io.Raw
            Electrode positions are set based on the montage
        data['preprocessing_steps'] : list
            Appends step information
        
        Returns
        -------
        data : dict
            Updated data dictionary with montage set
        """
        if 'raw' not in data:
            raise ValueError("set_montage requires 'raw' in data")

        montage_name = step_config.get('montage', 'standard_1020')

        montage = mne.channels.make_standard_montage(montage_name)
        data['raw'].set_montage(montage, on_missing="ignore")

        data['preprocessing_steps'].append({
            'step': 'set_montage',
            'montage': montage_name
        })
        return data

    def _step_drop_unused_channels(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Drop unused channels from the data.
        
        This step explicitly removes specified channels from the dataset.
        Different from drop_bad_channels, this step drops channels by name
        regardless of whether they are marked as bad.
        
        Parameters (via step_config)
        -----------------------------
        channels_to_drop : list of str
            List of channel names to drop from the data
        instance : str, optional
            Which data instance to drop channels from - 'raw' or 'epochs' (default: 'raw')
        
        Updates
        -------
        data[instance] : mne.io.Raw or mne.Epochs
            Specified channels are removed from the data
        data['preprocessing_steps'] : list
            Appends step information including list of dropped channels
        
        Returns
        -------
        data : dict
            Updated data dictionary with specified channels removed
        """
        channels_to_drop = step_config.get('channels_to_drop', [])
        instance = step_config.get('instance', 'raw') 

        if instance not in data:
            raise ValueError(f"drop_unused_channels step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

        data[instance].drop_channels(channels_to_drop)

        data['preprocessing_steps'].append({
            'step': 'drop_unused_channels',
            'instance': instance,
            'channels_dropped': channels_to_drop
        })

        return data

    def _step_bandpass_filter(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply bandpass filtering.
        
        Applies both high-pass and low-pass filters using IIR Butterworth filters.
        
        Parameters (via step_config)
        -----------------------------
        l_freq : float, optional
            High-pass filter frequency in Hz (default: 0.5)
        h_freq : float, optional
            Low-pass filter frequency in Hz (default: 45.0)
        l_freq_order : int, optional
            Filter order for high-pass filter (default: 6)
        h_freq_order : int, optional
            Filter order for low-pass filter (default: 8)
        picks : list, optional
            Channel types to filter (e.g., ['eeg']). If None, defaults to MEEG channels.
        excluded_channels : list of str, optional
            Channel names to exclude from filtering (e.g., reference channels)
        n_jobs : int, optional
            Number of parallel jobs (default: 1)
        
        Updates
        -------
        data['raw'] : mne.io.Raw
            Filters applied in-place
        data['preprocessing_steps'] : list
            Appends step information for both high-pass and low-pass filters
        
        Returns
        -------
        data : dict
            Updated data dictionary with filtered raw data
        """
        if 'raw' not in data:
            raise ValueError("bandpass_filter requires 'raw' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        l_freq = step_config.get('l_freq', 0.5)
        l_freq_order = step_config.get('l_freq_order', 6)
        h_freq = step_config.get('h_freq', 45.0)
        h_freq_order = step_config.get('h_freq_order', 8)
        n_jobs = step_config.get('n_jobs', 1)

        # Compute picks if provided, otherwise None (all channels)
        picks = self._get_picks(data['raw'].info, picks_params, excluded_channels)

        # Apply filtering in 2 steps: high-pass and low-pass
        high_pass_filter_params = dict(
            method='iir',
            l_trans_bandwidth=0.1,
            iir_params=dict(ftype='butter', order=l_freq_order),
            l_freq=l_freq,
            h_freq=None,
            n_jobs=n_jobs
        )
        data['raw'].filter(
            picks=picks,
            **high_pass_filter_params
        )

        low_pass_filter_params = dict(
            method='iir',
            h_trans_bandwidth=0.1,
            iir_params=dict(ftype='butter', order=h_freq_order),
            l_freq=None,
            h_freq=h_freq,
            n_jobs=n_jobs
        )
        data['raw'].filter(
            picks=picks,
            **low_pass_filter_params
        )

        # Store info for reporting
        data['preprocessing_steps'].extend([
            {
                'step': 'high_pass_filter',
                'picks': picks_params,
                'excluded_channels': excluded_channels,
                'params': high_pass_filter_params
            },
            {
                'step': 'low_pass_filter',
                'picks': picks_params,
                'excluded_channels': excluded_channels,
                'params': low_pass_filter_params
            }
        ])

        return data

    def _step_notch_filter(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply notch filtering to remove line noise.
        
        Removes power line interference at specified frequencies (e.g., 50 Hz or 60 Hz
        and their harmonics).
        
        Parameters (via step_config)
        -----------------------------
        freqs : list of float
            Frequencies to notch filter in Hz (e.g., [50.0, 100.0])
        notch_widths : float or list, optional
            Width of notch filters. If None, uses MNE default.
        method : str, optional
            Filtering method (default: 'fft')
        picks : list, optional
            Channel types to filter. If None, defaults to MEEG channels.
        excluded_channels : list of str, optional
            Channel names to exclude from filtering
        n_jobs : int, optional
            Number of parallel jobs (default: 1)
        
        Updates
        -------
        data['raw'] : mne.io.Raw
            Notch filters applied in-place
        data['preprocessing_steps'] : list
            Appends step information
        
        Returns
        -------
        data : dict
            Updated data dictionary with notch-filtered raw data
        """
        if 'raw' not in data:
            raise ValueError("notch_filter requires 'raw' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        freqs = step_config.get('freqs', [50.0, 100.0])
        notch_widths = step_config.get('notch_widths', None)
        method = step_config.get('method', 'fft')
        n_jobs = step_config.get('n_jobs', 1)

        # Compute picks if provided
        picks = self._get_picks(data['raw'].info, picks_params, excluded_channels)

        data['raw'].notch_filter(
            freqs=freqs,
            method=method,
            notch_widths=notch_widths,
            picks=picks,
            n_jobs=n_jobs
        )

        # Store info for reporting
        data['preprocessing_steps'].append({
            'step': 'notch_filter',
            'picks': picks_params,
            'excluded_channels': excluded_channels,
            'freqs': freqs,
            'method': method,
            'notch_widths': notch_widths
        })

        return data

    def _step_resample(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Resample a Raw or Epochs instance to a new sampling frequency.

        Args:
            data: Pipeline data dict. Must contain the key named by
                ``step_config['instance']`` (default ``'raw'``).
            step_config: Step parameters:
                - ``instance`` (str): Key in ``data`` to resample. Default ``'raw'``.
                - ``sfreq`` (float): Target sampling frequency. Default 250.
                - ``npad`` (str|int): Padding mode passed to MNE. Default ``'auto'``.
                - ``n_jobs`` (int): Parallel jobs. Default 1.
                - ``resample_events`` (bool): Also resample the events array.
                  Default ``False``.

        Returns:
            Updated data dict with the instance resampled in-place.

        Raises:
            ValueError: If the requested instance is not present in ``data``.
        """
        instance = step_config.get('instance', 'raw')
        
        if instance not in data:
            raise ValueError(f"resample requires '{instance}' in data")

        resample_events = step_config.get('resample_events', False)
        sfreq = step_config.get('sfreq', 250)
        npad = step_config.get('npad', 'auto')
        n_jobs = step_config.get('n_jobs', 1)

        data[instance].resample(
            sfreq=sfreq,
            npad=npad,
            n_jobs=n_jobs
        )

        if resample_events and 'events' in data:
            mne.events.resample_events(
                data['events'],
                data['events_sfreq'],
                sfreq
            ) 

        # Store info for reporting
        data['preprocessing_steps'].append({
            'step': 'resample',
            'instance': instance,
            'resample_events': resample_events,
            'sfreq': sfreq,
            'npad': npad,
        })

        return data

    def _step_reference(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply EEG re-referencing to a Raw or Epochs instance.

        Args:
            data: Pipeline data dict. Must contain the key named by
                ``step_config['instance']`` (default ``'epochs'``).
            step_config: Step parameters:
                - ``instance`` (str): Key in ``data`` to re-reference.
                  Default ``'epochs'``.
                - ``ref_channels`` (str|list): Reference channel(s) passed to
                  ``mne.set_eeg_reference``. Default ``'average'``.

        Returns:
            Updated data dict with the instance re-referenced in-place.

        Raises:
            ValueError: If the requested instance is not present in ``data``.
        """
        ref_channels = step_config.get('ref_channels', 'average')
        instance = step_config.get('instance', 'epochs')

        if instance not in data:
            raise ValueError(f"reference step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

        mne.set_eeg_reference(
            inst=data[instance],
            ref_channels=ref_channels,
        )

        data['preprocessing_steps'].append({
            'step': 'reference',
            'ref_channels': ref_channels
        })

        return data

    def _step_interpolate_bad_channels(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpolate bad channels using spherical spline interpolation.
        
        Repairs bad channels by interpolating their values from neighboring channels.
        After interpolation, the channels are removed from the info['bads'] list.
        
        Parameters (via step_config)
        -----------------------------
        instance : str, optional
            Which data instance to interpolate - 'raw' or 'epochs' (default: 'epochs')
        excluded_channels : list of str, optional
            Channel names to exclude from interpolation even if marked as bad.
            These channels will remain in info['bads'] after interpolation.
        
        Updates
        -------
        data[instance] : mne.io.Raw or mne.Epochs
            Bad channels (except excluded ones) are interpolated and removed from info['bads']
        data['preprocessing_steps'] : list
            Appends step information
        
        Returns
        -------
        data : dict
            Updated data dictionary with bad channels interpolated
        """
        instance = step_config.get('instance', 'epochs')
        excluded_channels = step_config.get('excluded_channels', [])

        if instance not in data:
            raise ValueError(f"interpolate_bad_channels step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

        data[instance].interpolate_bads(
            reset_bads=True,
            exclude=excluded_channels
        )

        data['preprocessing_steps'].append({
            'step': 'interpolate_bad_channels',
            'excluded_channels': excluded_channels,
            'instance': instance
        })

        return data

    def _step_drop_bad_channels(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Drop bad channels without interpolation.
        
        This step removes channels marked as bad from the data instead of interpolating them.
        Useful when you want to permanently remove problematic channels from the dataset.
        
        Parameters (via step_config)
        -----------------------------
        instance : str, optional
            Which data instance to drop channels from - 'raw' or 'epochs' (default: 'epochs')
        excluded_channels : list of str, optional
            List of channel names to exclude from dropping even if marked as bad.
            These channels will remain in the data even if they are in info['bads'].
        
        Updates
        -------
        data[instance] : mne.io.Raw or mne.Epochs
            Channels marked as bad (except excluded ones) are removed from the data
        data['preprocessing_steps'] : list
            Appends step information including list of dropped channels
        
        Returns
        -------
        data : dict
            Updated data dictionary with bad channels removed
        """
        instance = step_config.get('instance', 'epochs')
        excluded_channels = step_config.get('excluded_channels', None)

        if instance not in data:
            raise ValueError(f"drop_bad_channels step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

        # Get the list of bad channels before dropping
        bad_channels = list(data[instance].info['bads'])
        
        # Filter out excluded channels if specified
        if excluded_channels:
            channels_to_drop = [ch for ch in bad_channels if ch not in excluded_channels]
            excluded_bads = [ch for ch in bad_channels if ch in excluded_channels]
            if excluded_bads:
                logger.info(f"Excluding {len(excluded_bads)} bad channels from dropping: {excluded_bads}")
        else:
            channels_to_drop = bad_channels
        
        if channels_to_drop:
            # Drop the bad channels
            data[instance].drop_channels(channels_to_drop)
            logger.info(f"Dropped {len(channels_to_drop)} bad channels: {channels_to_drop}")
        else:
            logger.info("No bad channels to drop")

        data['preprocessing_steps'].append({
            'step': 'drop_bad_channels',
            'instance': instance,
            'excluded_channels': excluded_channels,
            'dropped_channels': channels_to_drop,
            'n_bad_channels': len(channels_to_drop)
        })

        return data

    def _step_ica(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Independent Component Analysis (ICA) for artifact removal.
        
        Decomposes the signal into independent components and can automatically
        detect and remove EOG and ECG artifacts.
        
        Parameters (via step_config)
        -----------------------------
        n_components : int, optional
            Number of ICA components (default: 20)
        method : str, optional
            ICA method: 'fastica', 'infomax', or 'picard' (default: 'fastica')
        random_state : int, optional
            Random state for reproducibility (default: 97)
        picks : list, optional
            Channel types to include in ICA. If None, defaults to MEEG channels.
        excluded_channels : list of str, optional
            Channel names to exclude from ICA decomposition
        find_eog : bool, optional
            Automatically find and exclude EOG artifacts (default: False)
        find_ecg : bool, optional
            Automatically find and exclude ECG artifacts (default: False)
        selected_indices : list of int, optional
            Manually specify component indices to exclude
        apply : bool, optional
            Apply ICA to remove artifacts (default: True)

        Updates
        -------
        data['ica'] : mne.preprocessing.ICA
            Fitted ICA object (stored for optional visualization)
        data['raw'] : mne.io.Raw
            If apply=True, artifacts are removed from raw data
        data['preprocessing_steps'] : list
            Appends step information including excluded components

        Returns
        -------
        data : dict
            Updated data dictionary with ICA applied
        """
        if 'raw' not in data:
            raise ValueError("ica step requires 'raw' in data")

        n_components = step_config.get('n_components', 20)
        random_state = step_config.get('random_state', 97)
        method = step_config.get('method', 'fastica')
        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        ica_l_freq = step_config.get('ica_fit_l_freq', 1.0)
        ica_h_freq = step_config.get('ica_fit_h_freq', None)
        eog_measure = step_config.get('eog_measure', 'correlation')
        eog_threshold = step_config.get('eog_threshold', 'auto')
        eog_channels = step_config.get('eog_channels', None)
        eog_l_freq = step_config.get('eog_l_freq', 1.0)
        eog_h_freq = step_config.get('eog_h_freq', 10.0)
        ecg_measure = step_config.get('ecg_measure', 'correlation')
        ecg_threshold = step_config.get('ecg_threshold', 'auto')
        ecg_channels = step_config.get('ecg_channels', None)
        ecg_l_freq = step_config.get('ecg_l_freq', 1.0)
        ecg_h_freq = step_config.get('ecg_h_freq', 10.0)
        selected_indices = step_config.get('selected_indices', None)
        apply = step_config.get('apply', True)

        raw = data['raw'].copy().filter(l_freq=ica_l_freq, h_freq=ica_h_freq)

        # --- Fit ICA on MEEG only (your _get_picks already defaults to eeg=True, eog=False) ---
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=random_state,
            method=method,
            max_iter='auto'
        )

        # Compute picks if provided
        picks = self._get_picks(raw.info, picks_params, excluded_channels)

        # Fit ICA
        ica.fit(raw, picks=picks)

        excluded_components = defaultdict(list)
        eog_detection_report = None
        ecg_detection_report = None

        # EOG
        if step_config.get('find_eog', False):

            if eog_channels is None:
                eog_channels = mne.pick_types(
                    raw.info,
                    eog=True,
                    exclude='bads'
                )

            if isinstance(eog_channels, str):
                eog_channels = [eog_channels]

            if eog_channels is None or len(eog_channels) == 0:
                raise ValueError("No eog_channels on instance and no channel selected in the config. Can't perform automatic EOG ICA without EOG channels.")

            present_eog = [ch for ch in eog_channels if ch in raw.ch_names]
            if len(present_eog) == 0:
                raise ValueError('All EOG channels from config are not in the instance.')
            
            if len(present_eog) < len(eog_channels):
                non_existent_eog = [ch for ch in eog_channels if ch not in raw.ch_names]
                logger.warning(f'The following selected EOG channels are not in the instance: {non_existent_eog}')

            eog_indices = []
            eog_scores = []
            for ch_name in present_eog:
                cur_eog_indices, cur_eog_scores = ica.find_bads_eog(
                    raw,
                    ch_name=ch_name,
                    measure=eog_measure,
                    l_freq=eog_l_freq,
                    h_freq=eog_h_freq,
                    threshold=eog_threshold
                )

                eog_indices.extend(cur_eog_indices)
                eog_scores.append(
                    cur_eog_scores.tolist()
                    if isinstance(cur_eog_scores, np.ndarray)
                    else cur_eog_scores
                )

            eog_indices = list(set(eog_indices))  # Unique indices

            for idx in eog_indices:
                excluded_components[idx].append('eog')

            eog_detection_report = {
                'eog_channels_requested': eog_channels,
                'eog_channels_present': present_eog,
                'eog_l_freq': eog_l_freq,
                'eog_h_freq': eog_h_freq,
                'eog_measure': eog_measure,
                'eog_threshold': eog_threshold,
                'eog_excluded_components': eog_indices,
                'eog_scores': eog_scores,
            }

        # ECG
        if step_config.get('find_ecg', False):

            if ecg_channels is None:
                ecg_channels = mne.pick_types(
                    raw.info,
                    ecg=True,
                    exclude='bads'
                )

            if isinstance(ecg_channels, str):
                ecg_channels = [ecg_channels]

            if ecg_channels is None or len(ecg_channels) == 0:
                raise ValueError("No ecg_channels on instance and no channel selected in the config. Can't perform automatic ECG ICA without ECG channels.")

            present_ecg = [ch for ch in ecg_channels if ch in raw.ch_names]
            if len(present_ecg) == 0:
                raise ValueError('All ECG channels from config are not in the instance.')
            
            if len(present_ecg) < len(ecg_channels):
                non_existent_dropped_ecg = [ch for ch in ecg_channels if ch not in raw.ch_names]
                logger.warning(f'The following selected ECG channels are not in the instance: {non_existent_dropped_ecg}')

            ecg_indices = []
            ecg_scores = []
            for ch_name in present_ecg:
                cur_ecg_indices, cur_ecg_scores = ica.find_bads_ecg(
                    raw,
                    ch_name=ch_name,
                    measure=ecg_measure,
                    l_freq=ecg_l_freq,
                    h_freq=ecg_h_freq,
                    threshold=ecg_threshold
                )

                ecg_indices.extend(cur_ecg_indices)
                ecg_scores.append(
                    cur_ecg_scores.tolist()
                    if isinstance(cur_ecg_scores, np.ndarray)
                    else cur_ecg_scores
                )
            
            ecg_indices = list(set(ecg_indices))  # Unique indices

            for idx in ecg_indices:
                excluded_components[idx].append('ecg')
            
            ecg_detection_report = {
                'ecg_channels_requested': ecg_channels,
                'ecg_channels_present': present_ecg,
                'ecg_l_freq': ecg_l_freq,
                'ecg_h_freq': ecg_h_freq,
                'ecg_measure': ecg_measure,
                'ecg_threshold': ecg_threshold,
                'ecg_excluded_components': ecg_indices,
                'ecg_scores': ecg_scores,
            }

        # Manual selection optional
        if selected_indices is not None:
            for idx in selected_indices:
                excluded_components[idx].append('selected')

        ica.exclude = sorted(excluded_components.keys())

        # Apply ICA to remove artifacts if requested
        if apply:
            ica.apply(data['raw'])
        
        data['ica'] = ica

        data['preprocessing_steps'].append({
            'step': 'ica',
            'n_components': n_components,
            'random_state': random_state,
            'method': method,
            'picks': picks_params,
            'excluded_channels': excluded_channels,
            'ica_l_freq': ica_l_freq,
            'ica_h_freq': ica_h_freq,
            'eog_detection': eog_detection_report or {},
            'ecg_detection': ecg_detection_report or {},
            'excluded_components': ica.exclude,
            'apply': apply,
        })

        return data

    def _step_find_events(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Find events in the data."""
        if 'raw' not in data:
            raise ValueError("find_events requires 'raw' in data")

        get_events_from = step_config.get('get_events_from', 'annotations')
        shortest_event = step_config.get('shortest_event', 1)
        event_id = step_config.get('event_id', 'auto')
        stim_channel = step_config.get('stim_channel', None)
        
        data['events'], found_event_id = self._find_events_from_raw(
            data['raw'],
            get_events_from=get_events_from,
            shortest_event=shortest_event,
            event_id=event_id,
            stim_channel=stim_channel
        )
        data['event_id'] = found_event_id
        data['events_sfreq'] = data['raw'].info['sfreq']

        data['preprocessing_steps'].append({
            'step': 'find_events',
            'found_event_id': found_event_id,
            'found_events': data['events'].tolist(),
            'n_events': data['events'].shape[0]
        })

        return data

    def _step_epoch(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create event-locked epochs from raw data.

        Args:
            data: Pipeline data dict. Must contain ``'raw'`` and ``'events'``.
            step_config: Step parameters:
                - ``event_id`` (dict|str): Event IDs to epoch around. Falls
                  back to ``data['event_id']`` if not provided.
                - ``tmin`` (float): Epoch start time relative to event.
                  Default -0.2 s.
                - ``tmax`` (float): Epoch end time relative to event.
                  Default 0.5 s.
                - ``baseline`` (tuple): Baseline correction window.
                  Default ``(None, 0.0)``.
                - ``reject`` (dict|None): Peak-to-peak rejection thresholds.
                  Default ``None``.

        Returns:
            Updated data dict with ``data['epochs']`` set to the new
            ``mne.Epochs`` object.

        Raises:
            ValueError: If ``'raw'`` or ``'events'`` are not in ``data``.
        """
        if data.get('raw', None) is None or data.get('events', None) is None:
            raise ValueError("epoch step requires both 'raw' and 'events' in data")

        event_id = step_config.get('event_id', None) or data.get('event_id', 'NOT FOUND')
        tmin = step_config.get('tmin', -0.2)
        tmax = step_config.get('tmax', 0.5)
        baseline = step_config.get('baseline', (None, 0.0))
        reject = step_config.get('reject', None)

        data['epochs'] = mne.Epochs(
            data['raw'],
            events=data['events'],
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=reject,
            preload=True
        )

        data['preprocessing_steps'].append({
            'step': 'epoch',
            'event_id': event_id,
            'tmin': tmin,
            'tmax': tmax,
            'baseline': baseline,
            'reject': reject,
            'n_epochs': len(data['epochs'])
        })

        return data

    def _step_chunk_in_epoch(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Segment raw data into fixed-length epochs (no events required).

        Args:
            data: Pipeline data dict. Must contain ``'raw'``.
            step_config: Step parameters:
                - ``duration`` (float): Length of each epoch in seconds.
                  Default 1.

        Returns:
            Updated data dict with ``data['epochs']`` set to the new
            fixed-length ``mne.Epochs`` object.

        Raises:
            ValueError: If ``'raw'`` is not in ``data``.
        """
        if data.get('raw', None) is None:
            raise ValueError("epoch step requires 'raw' in data")

        duration = step_config.get('duration', 1)

        data['epochs'] = mne.make_fixed_length_epochs(data['raw'], duration=duration, preload=True)

        data['preprocessing_steps'].append({
            'step': 'epoch',
            'type': 'fixed_length_epochs',
            'duration': duration,
        })

        return data

    def _step_find_flat_channels(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find flat channels based on variance threshold.
        
        Flat channels often indicate disconnected electrodes or other hardware issues.
        Channels with variance below the threshold are marked as bad.
        
        Parameters (via step_config)
        -----------------------------
        picks : list, optional
            Channel types to analyze (default: all MEEG channels)
        excluded_channels : list, optional
            Channel names to exclude from analysis (e.g., reference channels)
        threshold : float, optional
            Variance threshold below which channels are considered flat
            (default: 1e-12)
        
        Updates
        -------
        data['raw'].info['bads'] : list
            Adds detected flat channels (without duplicates)
        data['preprocessing_steps'] : list
            Appends step information including detected bad channels
        
        Returns
        -------
        data : dict
            Updated data dictionary with flat channels marked as bad
        """
        if 'raw' not in data:
            raise ValueError("find_flat_channels requires 'raw' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        threshold = step_config.get('threshold', 1e-12)
        
        # Get picks with exclusions
        picks = self._get_picks(data['raw'].info, picks_params, excluded_channels)

        # Get data only for selected picks
        raw_data = data['raw'].get_data(picks=picks)
        variances = raw_data.var(axis=1)
        flat_idx = np.where(variances < threshold)[0]
        # Map back to channel names using picks
        flat_chs = [data['raw'].ch_names[picks[i]] for i in flat_idx]
        
        if flat_chs:
            data['raw'].info['bads'].extend([ch for ch in flat_chs if ch not in data['raw'].info['bads']])
        
        data['preprocessing_steps'].append({
            'step': 'find_flat_channels',
            'instance': 'raw',
            'picks': picks_params,
            'excluded_channels': excluded_channels,
            'apply_on': ['raw'],
            'threshold': threshold,
            'bad_channels': flat_chs,
            'n_bad_channels': len(flat_chs)
        })

        return data

    def _step_find_bads_channels_threshold(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Find bad channels that exceed amplitude thresholds across epochs.

        Delegates to :func:`adaptive_reject.find_bads_channels_threshold` and
        marks detected channels as bad in all instances listed in
        ``step_config['apply_on']``.

        Args:
            data: Pipeline data dict. Must contain ``'epochs'``.
            step_config: Step parameters:
                - ``picks`` (list|None): Channel types to consider. Default all.
                - ``excluded_channels`` (list|None): Channels to skip.
                - ``reject`` (dict): Amplitude thresholds per channel type,
                  e.g. ``{'eeg': 100e-6}``. Default ``{'eeg': 100e-6}``.
                - ``n_epochs_bad_ch`` (float): Fraction of epochs in which a
                  channel must exceed the threshold to be marked bad. Default 0.5.
                - ``apply_on`` (list): Instances to mark bad channels on.
                  Default ``['epochs']``.

        Returns:
            Updated data dict with bad channels added to ``info['bads']``.

        Raises:
            ValueError: If ``'epochs'`` or any ``apply_on`` instance is absent.
        """
        if 'epochs' not in data:
            raise ValueError("find_bads_channels_threshold requires 'epochs' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        reject = step_config.get('reject', {'eeg': 100e-6})
        n_epochs_bad_ch = step_config.get('n_epochs_bad_ch', 0.5)
        apply_on = step_config.get('apply_on', ['epochs'])

        if not isinstance(apply_on, list):
            apply_on = [apply_on]

        if any(inst not in data for inst in apply_on):
            raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

        picks = self._get_picks(data['epochs'].info, picks_params, excluded_channels)

        bad_chs = adaptive_reject.find_bads_channels_threshold(
            data['epochs'], picks, reject, n_epochs_bad_ch
        )

        if bad_chs:
            for instance_to_apply in apply_on:
                data[instance_to_apply].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_to_apply].info['bads']])

        data['preprocessing_steps'].append({
            'step': 'find_bads_channels_threshold',
            'picks': picks_params,
            'excluded_channels': excluded_channels,
            'apply_on': apply_on,
            'reject': reject,
            'n_epochs_bad_ch': n_epochs_bad_ch,
            'bad_channels': bad_chs,
            'n_bad_channels': len(bad_chs)
        })

        return data

    def _step_find_bads_channels_variance(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Find bad channels with abnormal variance using z-score outlier detection.

        Delegates to :func:`adaptive_reject.find_bads_channels_variance` and
        marks detected channels as bad in all instances listed in
        ``step_config['apply_on']``.

        Args:
            data: Pipeline data dict. Must contain the key specified by
                ``step_config['instance']`` (default ``'epochs'``).
            step_config: Step parameters:
                - ``instance`` (str): Data key to analyse. Default ``'epochs'``.
                - ``picks`` (list|None): Channel types to consider. Default all.
                - ``excluded_channels`` (list|None): Channels to skip.
                - ``zscore_thresh`` (float): Z-score threshold. Default 4.
                - ``max_iter`` (int): Maximum outlier-removal iterations.
                  Default 2.
                - ``apply_on`` (list): Instances to mark bad channels on.
                  Defaults to ``[instance]``.

        Returns:
            Updated data dict with bad channels added to ``info['bads']``.

        Raises:
            ValueError: If the requested instance or any ``apply_on`` key is absent.
        """
        # Check which instance to use
        instance = step_config.get('instance', 'epochs')
        if instance not in data:
            raise ValueError(f"find_bads_channels_variance requires '{instance}' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        zscore_thresh = step_config.get('zscore_thresh', 4)
        max_iter = step_config.get('max_iter', 2)
        apply_on = step_config.get('apply_on', [instance])

        if not isinstance(apply_on, list):
            apply_on = [apply_on]

        if any(inst not in data for inst in apply_on):
            raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

        picks = self._get_picks(data[instance].info, picks_params, excluded_channels)

        bad_chs = adaptive_reject.find_bads_channels_variance(
            data[instance], picks, zscore_thresh, max_iter
        )

        # Mark channels as bad
        if bad_chs:
            for instance_to_apply in apply_on:
                data[instance_to_apply].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_to_apply].info['bads']])

        data['preprocessing_steps'].append({
            'step': 'find_bads_channels_variance',
            'instance': instance,
            'picks': picks_params,
            'excluded_channels': excluded_channels,
            'apply_on': apply_on,
            'zscore_thresh': zscore_thresh,
            'max_iter': max_iter,
            'bad_channels': bad_chs,
            'n_bad_channels': len(bad_chs)
        })

        return data

    def _step_find_bads_channels_high_frequency(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Find bad channels with excessive high-frequency noise.

        Applies a 25 Hz high-pass filter and identifies channels whose filtered
        variance is a z-score outlier. Delegates to
        :func:`adaptive_reject.find_bads_channels_high_frequency`.

        Args:
            data: Pipeline data dict. Must contain the key specified by
                ``step_config['instance']`` (default ``'epochs'``).
            step_config: Step parameters:
                - ``instance`` (str): Data key to analyse. Default ``'epochs'``.
                - ``picks`` (list|None): Channel types to consider. Default all.
                - ``excluded_channels`` (list|None): Channels to skip.
                - ``zscore_thresh`` (float): Z-score threshold. Default 4.
                - ``max_iter`` (int): Maximum outlier-removal iterations.
                  Default 2.
                - ``apply_on`` (list): Instances to mark bad channels on.
                  Defaults to ``[instance]``.

        Returns:
            Updated data dict with bad channels added to ``info['bads']``.

        Raises:
            ValueError: If the requested instance or any ``apply_on`` key is absent.
        """
        # Check which instance to use
        instance = step_config.get('instance', 'epochs')
        if instance not in data:
            raise ValueError(f"find_bads_channels_high_frequency requires '{instance}' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        zscore_thresh = step_config.get('zscore_thresh', 4)
        max_iter = step_config.get('max_iter', 2)
        apply_on = step_config.get('apply_on', [instance])

        if not isinstance(apply_on, list):
            apply_on = [apply_on]
        
        if any(inst not in data for inst in apply_on):
            raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

        picks = self._get_picks(data[instance].info, picks_params, excluded_channels)

        bad_chs = adaptive_reject.find_bads_channels_high_frequency(
            data[instance], picks, zscore_thresh, max_iter
        )

        # Mark channels as bad
        if bad_chs:
            for instance_to_apply in apply_on:
                data[instance_to_apply].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_to_apply].info['bads']])

        data['preprocessing_steps'].append({
            'step': 'find_bads_channels_high_frequency',
            'instance': instance,
            'picks': picks_params,
            'excluded_channels': excluded_channels,
            'apply_on': apply_on,
            'zscore_thresh': zscore_thresh,
            'max_iter': max_iter,
            'bad_channels': bad_chs,
            'n_bad_channels': len(bad_chs)
        })

        return data

    def _step_find_bads_epochs_threshold(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Drop epochs in which too many channels exceed amplitude thresholds.

        Delegates to :func:`adaptive_reject.find_bads_epochs_threshold` and
        drops the identified bad epochs in-place.

        Args:
            data: Pipeline data dict. Must contain ``'epochs'``.
            step_config: Step parameters:
                - ``picks`` (list|None): Channel types to consider. Default all.
                - ``excluded_channels`` (list|None): Channels to skip.
                - ``reject`` (dict): Amplitude thresholds per channel type.
                  Default ``{'eeg': 100e-6}``.
                - ``n_channels_bad_epoch`` (float): Fraction of channels that
                  must exceed the threshold for an epoch to be dropped.
                  Default 0.1.

        Returns:
            Updated data dict with bad epochs dropped from ``data['epochs']``.

        Raises:
            ValueError: If ``'epochs'`` is not in ``data``.
        """
        if 'epochs' not in data:
            raise ValueError("find_bads_epochs_threshold requires 'epochs' in data")

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        reject = step_config.get('reject', {'eeg': 100e-6})
        n_channels_bad_epoch = step_config.get('n_channels_bad_epoch', 0.1)

        picks = self._get_picks(data['epochs'].info, picks_params, excluded_channels)

        bad_epochs = adaptive_reject.find_bads_epochs_threshold(
            data['epochs'], picks, reject, n_channels_bad_epoch
        )

        # Drop bad epochs
        if len(bad_epochs) > 0:
            data['epochs'].drop(bad_epochs, reason='ADAPTIVE AUTOREJECT')

        data['preprocessing_steps'].append({
            'step': 'find_bads_epochs_threshold',
            'picks': picks_params,
            'excluded_channels': excluded_channels,
            'apply_on': ['epochs'], # only for compatibility with others reject steps
            'reject': reject,
            'n_channels_bad_epoch': n_channels_bad_epoch,
            'bad_epochs': bad_epochs.tolist() if hasattr(bad_epochs, 'tolist') else list(bad_epochs),
            'n_bad_epochs': len(bad_epochs),
            'n_epochs_remaining': len(data['epochs'])
        })

        return data

    def _step_save_clean_instance(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Save a preprocessed Raw or Epochs instance to the BIDS derivatives tree.

        The output path follows BIDS conventions under the pipeline's
        derivatives root, with ``processing=clean`` and ``description=cleaned``.

        Args:
            data: Pipeline data dict. Must contain the key named by
                ``step_config['instance']`` (default ``'epochs'``).
            step_config: Step parameters:
                - ``instance`` (str): Key in ``data`` to save (``'raw'`` or
                  ``'epochs'``). Default ``'epochs'``.
                - ``overwrite`` (bool): Overwrite existing file. Default ``True``.

        Returns:
            Updated data dict with ``data['{instance}_file']`` set to the saved
            path string.

        Raises:
            ValueError: If the requested instance is not in ``data``.
        """
        instance = step_config.get('instance', 'epochs')
        overwrite = step_config.get('overwrite', True)
        processing = step_config.get('processing', None)
        description = step_config.get('description', None)
        datatype = step_config.get('datatype', None)
        suffix = step_config.get('suffix', None)
        extension = step_config.get('suffix', None)

        if instance not in data:
            raise ValueError(f"save_clean_instances step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

        # Compute default suffixes if not specified otherwise
        if suffix is None:
            if instance == 'epochs':
                suffix = 'epo'
            elif instance == 'raw':
                suffix = 'eeg'

        # Compute extension if not expecified
        if extension is None and instance in ['epochs', 'raw']:
            suffix = '.fif'

        # Derivatives root for this pipeline
        deriv_root = self._get_derivatives_root(instance)

        bids_path = BIDSPath(
            subject=data['subject'],
            task=data['task'],
            session=data.get('session', None),
            acquisition=data.get('acquisition', None),
            datatype=datatype,
            root=deriv_root,
            suffix=suffix,
            extension=extension,
            processing=processing,
            description=description,
            check=False,
        )

        # Ensure directory exists
        bids_path.mkdir(exist_ok=True)

        # Save instance
        data[instance].save(bids_path.fpath, overwrite=overwrite)

        # Store paths
        data[f'{instance}_file'] = str(bids_path)

        return data

    def _step_generate_json_report(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Write a JSON report of the preprocessing steps to the BIDS derivatives tree.

        The report includes subject/task/session metadata, raw data properties,
        and the full list of preprocessing steps recorded in
        ``data['preprocessing_steps']``.

        Args:
            data: Pipeline data dict containing at minimum ``'subject'``,
                ``'task'``, and ``'preprocessing_steps'``.
            step_config: Currently unused; reserved for future options.

        Returns:
            Updated data dict with ``data['json_report']`` set to the output
            path string.
        """

        # JSON report
        report = {
            'subject': data['subject'],
            'task': data['task'],
            'session': data.get('session', None),
            'acquisition': data.get('acquisition', None),
            'preprocessing_steps': data.get('preprocessing_steps', []),
        }

        if 'raw' in data:
            report['raw'] = dict(
                n_channels=data['raw'].info.get('nchan'),
                sfreq=data['raw'].info.get('sfreq'),
                n_times=data['raw'].n_times
            )

        # Derivatives root for this pipeline
        deriv_root = self._get_derivatives_root("reports")

        bids_path = BIDSPath(
            subject=data['subject'],
            task=data['task'],
            session=data.get('session', None),
            acquisition=data.get('acquisition', None),
            datatype="eeg",
            root=deriv_root,
            suffix="report",
            extension=".json",
            processing="clean",
            description="cleaned",
            check=False,
        )

        # Ensure directory exists
        bids_path.mkdir(exist_ok=True)

        with open(bids_path.fpath, 'w') as f:
            json.dump(report, f, indent=2, cls=NpEncoder)

        data['json_report'] = str(bids_path)
        return data

    def _step_generate_html_report(self, data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an interactive HTML report of the preprocessing results.

        Produces a self-contained HTML file in the BIDS derivatives tree that
        includes a bad-channels topoplot, a preprocessing-steps table, and
        optional raw/ICA comparison plots.

        Args:
            data: Pipeline data dict containing at minimum ``'subject'``,
                ``'task'``, and ``'preprocessing_steps'``.
            step_config: Step parameters:
                - ``picks`` (list|None): Channel types for topoplot. Default all.
                - ``excluded_channels`` (list|None): Channels to skip.
                - ``outlines`` (str): Topoplot outline style. Default ``'head'``.
                - ``compare_instances`` (list): Pairs of instance keys to
                  compare in the report. Default ``[]``.
                - ``plot_raw_kwargs`` (dict): Extra kwargs for raw plots.
                - ``plot_ica_kwargs`` (dict): Extra kwargs for ICA plots.

        Returns:
            Updated data dict with ``data['html_report']`` set to the output
            path string.
        """
        from .report import (
            collect_bad_channels_from_steps,
            create_bad_channels_topoplot,
            create_preprocessing_steps_table
        )

        picks_params = step_config.get('picks', None)
        excluded_channels = step_config.get('excluded_channels', None)
        outlines = step_config.get('outlines', 'head')
        compare_instances = step_config.get('compare_instances', [])
        plot_raw_kwargs = step_config.get('plot_raw_kwargs', {})
        plot_ica_kwargs = step_config.get('plot_ica_kwargs', {})
        plot_events_kwargs = step_config.get('plot_events_kwargs', {})
        plot_epochs_kwargs = step_config.get('plot_epochs_kwargs', {})
        plot_evokeds_kwargs = step_config.get('plot_evokeds_kwargs', {})

        if 'preprocessing_steps' not in data:
            raise ValueError("generate_html_report requires 'preprocessing_steps' in data")
        elif not isinstance(data['preprocessing_steps'], list):
            raise ValueError("data['preprocessing_steps'] must be a list")

        # Get info from epochs if available, otherwise from raw
        inst = data['raw'] if 'raw' in data else data['epochs'] if 'epochs' in data else None
        if inst is None:
            raise ValueError("generate_html_report requires either 'raw' or 'epochs' in data")

        # Compute picks for channel selection
        picks = self._get_picks(inst.info, picks_params, excluded_channels)

        preprocessing_steps = data['preprocessing_steps']

        html_report = mne.Report(title=f'Preprocessing Report - Subject {data["subject"]}')

        # Add bad channels topoplot section
        bad_channels = collect_bad_channels_from_steps(preprocessing_steps)

        # Create topoplot if we have bad channels and info
        if len(bad_channels) > 0:
            logger.info(f"Adding bad channels topoplot with {len(bad_channels)} bad channels")
            fig = create_bad_channels_topoplot(inst.info, bad_channels, outlines=outlines)

            if fig is not None:
                # Add to report
                html_report.add_figure(
                    fig=fig,
                    title='Bad Channels',
                    caption=f'Topoplot showing {len(bad_channels)} bad channels marked with red crosses'
                )
                plt.close(fig)

        # Add preprocessing steps table section
        html_content = create_preprocessing_steps_table(data['preprocessing_steps'])

        # ---------- Preprocessing steps ----------
        if html_content is not None:
            # Add the HTML table to the report
            html_report.add_html(
                html=html_content,
                title='Preprocessing Steps',
            )

        # ---------- ICA ----------
        if data.get('ica', None) is not None:

            section = "ICA"

            html_report.add_ica(
                ica=data['ica'],
                title='ICA Components',
                inst=None,
                **plot_ica_kwargs
            )

            ica_step = [step for step in preprocessing_steps if step['step'] == 'ica']
            ica_step = ica_step[-1] if len(ica_step) > 0 else {}
            eog_step_report = ica_step.get('eog_detection', {})
            eog_idx = eog_step_report.get('eog_excluded_components', []) or []
            eog_scores = eog_step_report.get('eog_scores', None)
            ecg_step_report = ica_step.get('ecg_detection', {})
            ecg_idx = ecg_step_report.get('ecg_excluded_components', [])
            ecg_scores = ecg_step_report.get('ecg_scores', None)

            if len(eog_idx) > 0:

                if eog_scores is not None:
                    scores = np.array(eog_scores, dtype=float)

                    if scores.ndim == 1:
                        scores = scores.reshape(-1, 1)  # Make it 2D for uniform processing

                    # Heatmap (EOG channels x ICA components)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    im = ax.imshow(scores, aspect="auto", origin="lower")

                    n_components = scores.shape[1]

                    # X axis: ICA components as discrete labels 1..N
                    ax.set_xticks(np.arange(n_components))
                    ax.set_xticklabels(np.arange(n_components))

                    ax.set_xlabel("ICA component")
                    ax.set_ylabel("EOG channel")

                    eog_names = (
                        eog_step_report.get("eog_channels_present", None)
                        or eog_step_report.get("eog_channels_requested", None)
                        or []
                    )
                    if isinstance(eog_names, list) and len(eog_names) == scores.shape[0]:
                        ax.set_yticks(np.arange(len(eog_names)))
                        ax.set_yticklabels(eog_names)

                    ax.set_title("EOG scores (per EOG channel × ICA component)")

                    fig.colorbar(im, ax=ax, shrink=0.8, label="EOG score")

                    html_report.add_figure(
                        fig=fig,
                        title="ICA - EOG scores heatmap",
                        section='ICA - EOG'
                    )
                    plt.close(fig)

                    # Aggregate to 1 score per component for barplot
                    scores_1d = np.max(np.abs(scores), axis=0)

                    # Barplot (always 1D after aggregation if needed)
                    fig1 = plt.figure()
                    ax = fig1.add_subplot(111)
                    ax.bar(np.arange(len(scores_1d)), scores_1d)
                    ax.set_xlabel("ICA component")
                    ax.set_ylabel("max |EOG score| across EOG channels" if (eog_scores is not None and np.array(eog_scores).ndim == 2) else "|EOG score|")
                    ax.set_title(f"EOG scores (selected: {eog_idx})")
                    html_report.add_figure(
                        fig=fig1,
                        title="ICA - EOG scores",
                        section='ICA - EOG'
                    )
                    plt.close(fig1)
            
            if len(ecg_idx) > 0:

                if ecg_scores is not None:
                    scores = np.array(ecg_scores, dtype=float)

                    if scores.ndim == 1:
                        scores = scores.reshape(-1, 1)  # Make it 2D for uniform processing

                    # Heatmap (ECG channels x ICA components)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    im = ax.imshow(scores, aspect="auto", origin="lower")

                    n_components = scores.shape[1]

                    # X axis: ICA components as discrete labels 1..N
                    ax.set_xticks(np.arange(n_components))
                    ax.set_xticklabels(np.arange(n_components))

                    ax.set_xlabel("ICA component")
                    ax.set_ylabel("ECG channel")

                    ecg_names = (
                        ecg_step_report.get("ecg_channels_present", None)
                        or ecg_step_report.get("ecg_channels_requested", None)
                        or []
                    )
                    if isinstance(ecg_names, list) and len(ecg_names) == scores.shape[0]:
                        ax.set_yticks(np.arange(len(ecg_names)))
                        ax.set_yticklabels(ecg_names)

                    ax.set_title("ECG scores (per ECG channel × ICA component)")
                    fig.colorbar(im, ax=ax, shrink=0.8, label="ECG score")

                    html_report.add_figure(
                        fig=fig,
                        title="ICA - ECG scores heatmap",
                        section='ICA - EOG'
                    )
                    plt.close(fig)

                    # Aggregate to 1 score per component for barplot
                    scores_1d = np.max(np.abs(scores), axis=0)

                    # Barplot (always 1D after aggregation if needed)
                    fig1 = plt.figure()
                    ax = fig1.add_subplot(111)
                    ax.bar(np.arange(len(scores_1d)), scores_1d)
                    ax.set_xlabel("ICA component")
                    ax.set_ylabel("max |ECG score| across ECG channels" if (ecg_scores is not None and np.array(ecg_scores).ndim == 2) else "|ECG score|")
                    ax.set_title(f"ECG scores (selected: {ecg_idx})")
                    html_report.add_figure(
                        fig=fig1,
                        title="ICA - ECG scores",
                        section='ICA - EOG'
                    )
                    plt.close(fig1)

        # ---------- Compare instances preprocessing (full recording) ----------
        for contrast in compare_instances:
            inst_a_name = contrast['instance_a']['name']
            inst_a_label = contrast['instance_a']['label']
            inst_b_name = contrast['instance_b']['name']
            inst_b_label = contrast['instance_b']['label']

            if inst_a_name not in data or inst_b_name not in data:
                raise ValueError(f"compare_instances step requires both '{inst_a_name}' and '{inst_b_name}' in data")

            inst_a = data[inst_a_name]
            inst_b = data[inst_b_name]

            # Ensure channel alignment (same channel order)
            ch_names_picks = self._get_picks(
                inst.info,
                picks_params,
                excluded_channels
            )
            ch_names_a = sorted([inst_a.ch_names[pick] for pick in ch_names_picks])
            ch_names_b = sorted([inst_b.ch_names[pick] for pick in ch_names_picks])
            if set(ch_names_a) != set(ch_names_b):
                raise ValueError(f"compare_instances step: channel mismatch between '{inst_a}' and '{inst_b}' after picking")

            raw_b = inst_b.copy().pick(picks=ch_names_picks).reorder_channels(ch_names_a)
            raw_a = inst_a.copy().pick(picks=ch_names_picks).reorder_channels(ch_names_b)

            Xb = raw_b.get_data()
            Xa = raw_a.get_data()
            times = raw_a.times

            # Metrics over full recording
            gfp_b = np.std(Xb, axis=0)
            gfp_a = np.std(Xa, axis=0)

            mean_b = np.mean(Xb, axis=0)
            mean_a = np.mean(Xa, axis=0)

            diff_abs = np.mean(np.abs(Xb - Xa), axis=0)

            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

            axes[0].plot(times, gfp_b, color='red', alpha=0.35, label=inst_b_label)
            axes[0].plot(times, gfp_a, color='black', linewidth=1.0, label=inst_a_label)
            axes[0].set_title('Global Field Power (full recording)')
            axes[0].legend(loc='upper right')

            axes[1].plot(times, mean_b, color='red', alpha=0.35, label=inst_b_label)
            axes[1].plot(times, mean_a, color='black', linewidth=1.0, label=inst_a_label)
            axes[1].set_title('Mean across channels (full recording)')
            axes[0].legend(loc='upper right')

            axes[2].plot(times, diff_abs, color='purple', linewidth=1.0)
            axes[2].set_title(f'Mean absolute difference |{inst_a_label} - {inst_b_label}| (full recording)')
            axes[2].set_xlabel('Time (s)')

            fig.tight_layout()
            html_report.add_figure(
                fig=fig,
                title=contrast['title'],
                section='Contrasts'
            )
            plt.close(fig)

        # ---------- Cleaned Raw report ----------
        if data.get('raw', None) is not None:
            html_report.add_raw(
                raw=data['raw'].copy().pick(picks=picks),
                title='Clean Raw Data',
                **plot_raw_kwargs
            )

        # ---------- Events report ----------
        if 'events' in data and data['events'] is not None:
            html_report.add_events(
                events=data['events'],
                event_id=data.get('event_id', None),
                sfreq=data['events_sfreq'],
                title='Found Events',
                **plot_events_kwargs
            )

        # ---------- Cleaned Epochs report ----------
        if data.get('epochs', None) is not None:

            epochs=data['epochs'].copy().pick(picks=picks)

            html_report.add_epochs(
                epochs=epochs,
                title='Clean Epochs',
                **plot_epochs_kwargs
            )
            
            html_report.add_evokeds(
                evokeds=epochs.average(by_event_type=True),
                n_time_points=step_config.get('n_time_points', None),
                **plot_evokeds_kwargs
            )

        # Derivatives root for this pipeline
        deriv_root = self._get_derivatives_root("reports")

        bids_path = BIDSPath(
            subject=data['subject'],
            task=data['task'],
            session=data.get('session', None),
            acquisition=data.get('acquisition', None),
            datatype="eeg",
            root=deriv_root,
            suffix="report",
            extension=".html",
            processing="clean",
            description="cleaned",
            check=False,
        )

        # Ensure directory exists
        bids_path.mkdir(exist_ok=True)

        html_report.save(bids_path.fpath, overwrite=True, open_browser=False)

        data['html_report'] = str(bids_path)
        return data

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

        # Read all files
        if io_backend == 'read_raw_bids':
            data['all_raw'] = [read_raw_bids(bids_path=bp, verbose=True) for bp in paths]
        else:
            read_func = getattr(mne.io, io_backend, None)
            if read_func is None:
                raise ValueError(f"Unknown io_backend '{io_backend}' specified")
            data['all_raw'] = [read_func(str(p), preload=True, verbose=True) for p in paths]

        # Ensure data are loaded into memory for processing
        for raw in data['all_raw']:
            if not raw.preload:
                raw.load_data()

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
            data = self.step_functions[step_name](data, step_config)

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
        extension : str
            File extension to match (default: .vhdr)

        Returns
        -------
        all_results : dict
            Dictionary mapping subject -> list of results for each matching file.
        """
        
        # Use the reader to find recordings
        recordings = self.reader.find_recordings(
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            acquisitions=acquisitions,
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
                    raise exc
                finally:
                    # Remove the step task after this recording is done
                    progress.remove_task(step_task_id)
                
                # Update overall progress
                progress.update(overall_task, completed=i+1)

        logger.info(f"Pipeline completed.")
        return all_results