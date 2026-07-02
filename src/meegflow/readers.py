#!/usr/bin/env python3
"""
Dataset readers for EEG preprocessing pipeline.

This module provides different strategies for finding and reading EEG data files:
- BIDSReader: Uses MNE-BIDS to discover files in BIDS-formatted datasets
- GlobReader: Uses glob patterns with variable extraction to find files

The reader abstraction allows flexible file discovery while maintaining a consistent
interface for the preprocessing pipeline.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import re
from itertools import product
import mne
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
from mne.utils import logger


class DatasetReader(ABC):
    """Abstract base class for dataset readers.

    A reader is responsible for discovering data files based on specified criteria
    and returning them in a format the pipeline can process.
    """

    @property
    @abstractmethod
    def root(self) -> Path:
        """Root directory of the dataset (e.g. BIDS root or glob root).

        Standardizes access to the dataset location across reader
        implementations, so callers don't need to know whether a given
        reader calls its root ``bids_root`` or ``data_root``.
        """
        pass

    @abstractmethod
    def find_recordings(
        self,
        subjects: Optional[Union[str, List[str]]] = None,
        sessions: Optional[Union[str, List[str]]] = None,
        tasks: Optional[Union[str, List[str]]] = None,
        acquisitions: Optional[Union[str, List[str]]] = None,
        runs: Optional[Union[str, List[str]]] = None,
        extension: str = '.vhdr'
    ) -> List[Dict[str, Any]]:
        """Find recordings matching the specified criteria.

        Parameters
        ----------
        subjects : str, list of str, or None
            Subject ID(s) to process
        sessions : str, list of str, or None
            Session ID(s) to process
        tasks : str, list of str, or None
            Task(s) to process
        acquisitions : str, list of str, or None
            Acquisition parameter(s) to process
        runs : str, list of str, or None
            Run ID(s) to process
        extension : str
            File extension to match

        Returns
        -------
        list of dict
            List of recording dictionaries, each containing:
            - 'paths': list of file paths (list of BIDSPath or Path objects)
            - 'metadata': dict with subject, session, task, acquisition, run info
            - 'recording_name': string identifier for logging
        """
        pass

    def read(
        self,
        paths: List[Any],
        io_backend: str = 'read_raw_bids'
    ) -> List[Any]:
        """Load the raw files for a single recording into memory.

        Parameters
        ----------
        paths : list of BIDSPath or Path
            File paths belonging to one recording (as returned in the 'paths'
            key of ``find_recordings``).
        io_backend : str
            How to read each file. ``'read_raw_bids'`` (default) uses
            ``mne_bids.read_raw_bids``; any other value is resolved as a
            function name on ``mne.io`` (e.g. ``'read_raw_eeglab'``).

        Returns
        -------
        list of mne.io.Raw
            Preloaded Raw objects, one per path.

        Raises
        ------
        ValueError
            If ``io_backend`` is not ``'read_raw_bids'`` and does not resolve to
            a function on ``mne.io``.
        """
        if io_backend == 'read_raw_bids':
            raws = [read_raw_bids(bids_path=bp, verbose=True) for bp in paths]
        else:
            read_func = getattr(mne.io, io_backend, None)
            if read_func is None:
                raise ValueError(f"Unknown io_backend '{io_backend}' specified")
            raws = [read_func(str(p), preload=True, verbose=True) for p in paths]

        # Ensure data are loaded into memory for processing
        for raw in raws:
            if not raw.preload:
                raw.load_data()

        return raws


class BIDSReader(DatasetReader):
    """Reader for BIDS-formatted datasets using MNE-BIDS.
    
    This reader uses MNE-BIDS utilities to discover files in a BIDS dataset structure.
    It supports the standard BIDS entities: subject, session, task, acquisition, etc.
    
    Parameters
    ----------
    bids_root : str or Path
        Path to the BIDS root directory
    """
    
    def __init__(self, bids_root: Union[str, Path]):
        self.bids_root = Path(bids_root)

    @property
    def root(self) -> Path:
        return self.bids_root


    def _build_include_patterns(
        self,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None
    ) -> Union[str, List[str]]:
        """Build include_match patterns for get_entity_vals.
        
        Creates patterns to narrow the search space when discovering entity values.
        Handles both subjects with and without sessions gracefully.
        
        Parameters
        ----------
        subjects : list of str, optional
            Known subject values to narrow the search
        sessions : list of str, optional
            Known session values to narrow the search
            
        Returns
        -------
        str or list of str
            Pattern(s) to use with get_entity_vals include_match parameter
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
        entity_value: Any,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None
    ) -> List[Optional[str]]:
        """Get all unique values for a given BIDS entity in the dataset.
        
        Parameters
        ----------
        entity_key : str
            The BIDS entity key (e.g., 'subject', 'task', 'session', 'acquisition')
        entity_value : str, list of str, or None
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
                root=self.bids_root,
                entity_key=entity_key,
                include_match=include_patterns
            )
            # Return the list of values, or [None] if no values found
            return list(all_values) if all_values else [None]

        raise ValueError(f"Invalid type for entity '{entity_key}': {type(entity_value)}")
        
    def find_recordings(
        self,
        subjects: Optional[Union[str, List[str]]] = None,
        sessions: Optional[Union[str, List[str]]] = None,
        tasks: Optional[Union[str, List[str]]] = None,
        acquisitions: Optional[Union[str, List[str]]] = None,
        runs: Optional[Union[str, List[str]]] = None,
        extension: str = '.vhdr'
    ) -> List[Dict[str, Any]]:
        """Find recordings in BIDS dataset matching the specified criteria.

        Parameters
        ----------
        subjects : str, list of str, or None
            Subject ID(s) to process. If None, processes all subjects.
        sessions : str, list of str, or None
            Session ID(s) to process. If None, processes all sessions.
        tasks : str, list of str, or None
            Task(s) to process. If None, processes all tasks.
        acquisitions : str, list of str, or None
            Acquisition parameter(s) to process. If None, processes all acquisitions.
        runs : str, list of str, or None
            Run ID(s) to process. If None, processes all runs.
        extension : str
            File extension (default: .vhdr)

        Returns
        -------
        list of dict
            List of recording dictionaries with paths and metadata
        """
        if isinstance(runs, str):
            runs = [runs]

        subjects = self._get_entity_values('subject', subjects)
        sessions = self._get_entity_values('session', sessions, subjects=subjects)
        tasks = self._get_entity_values('task', tasks, subjects=subjects, sessions=sessions)
        acquisitions = self._get_entity_values('acquisition', acquisitions, subjects=subjects, sessions=sessions)

        logger.info(f"Subjects to process: {subjects}")
        logger.info(f"Sessions to process: {sessions}")
        logger.info(f"Tasks to process: {tasks}")
        logger.info(f"Acquisitions to process: {acquisitions}")
        if runs is not None:
            logger.info(f"Filtering to runs: {runs}")

        n_combinations = len(subjects) * len(sessions) * len(tasks) * len(acquisitions)
        logger.info(f"Computing {n_combinations} matching file(s) to process")

        recordings = []

        for subject, session, task, acquisition in product(subjects, sessions, tasks, acquisitions):
            pb = BIDSPath(
                root=self.bids_root,
                subject=subject,
                session=session,
                task=task,
                acquisition=acquisition,
                extension=extension,
                suffix='eeg',
                datatype='eeg',
            )

            all_raw_paths = list(pb.match(ignore_nosub=True))

            # Filter to requested runs; when runs is None all runs are kept
            if runs is not None:
                all_raw_paths = [p for p in all_raw_paths if p.run in runs]

            if len(all_raw_paths) == 0:
                logger.warning(f"No files found for {subject} - {session} - {task} - {acquisition}, skipping.")
                continue

            logger.info(f"Found {len(all_raw_paths)} recording(s) for {subject} - {session} - {task} - {acquisition} to process together.")

            recording_name = f"{subject} - {session} - {task} - {acquisition}"
            recordings.append({
                'paths': all_raw_paths,
                'metadata': {
                    'subject': subject,
                    'session': session,
                    'task': task,
                    'acquisition': acquisition
                },
                'recording_name': recording_name
            })

        return recordings


class GlobReader(DatasetReader):
    """Reader for datasets using glob patterns with variable extraction.
    
    This reader allows flexible pattern matching using glob syntax with named variables.
    Variables are specified as {variable_name} in the pattern, which get converted to
    wildcards (*) for matching and then extracted from the matched filenames.
    
    Examples:
        Pattern: "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_task-{task}_eeg.vhdr"
        Matches: "data/sub-01/ses-01/eeg/sub-01_task-rest_eeg.vhdr"
        Extracts: {'subject': '01', 'session': '01', 'task': 'rest'}
    
    Parameters
    ----------
    data_root : str or Path
        Root directory to search for data files
    pattern : str
        Glob pattern with {variable_name} placeholders
    """
    
    def __init__(self, data_root: Union[str, Path], pattern: str):
        self.data_root = Path(data_root)
        self.pattern = pattern

        # Parse the pattern to extract variable names and create glob pattern
        self.variable_names = self._extract_variable_names(pattern)
        self.glob_pattern = self._create_glob_pattern(pattern)
        self.regex_pattern = self._create_regex_pattern(pattern)

    @property
    def root(self) -> Path:
        return self.data_root

    def _extract_variable_names(self, pattern: str) -> List[str]:
        """Extract variable names from pattern like {subject}, {task}, etc."""
        return re.findall(r'\{(\w+)\}', pattern)
        
    def _create_glob_pattern(self, pattern: str) -> str:
        """Convert pattern with {variables} to glob pattern with * wildcards."""
        return re.sub(r'\{(\w+)\}', '*', pattern)
        
    def _create_regex_pattern(self, pattern: str) -> re.Pattern:
        """Convert pattern with {variables} and * wildcards to a regex.

        {variable} becomes a named capture group (grouping key).
        * becomes [^/]* (matches anything within a path segment, not extracted).
        Duplicate {variable} names use backreferences to enforce consistency.
        """
        # Split into tokens: {variable}, *, or literal text
        tokens = re.split(r'(\{[^}]+\}|\*)', pattern)
        seen_vars = set()
        regex_parts = []

        for token in tokens:
            if token.startswith('{') and token.endswith('}'):
                var_name = token[1:-1]
                if var_name not in seen_vars:
                    seen_vars.add(var_name)
                    regex_parts.append(f'(?P<{var_name}>[^/]+)')
                else:
                    regex_parts.append(f'(?P={var_name})')
            elif token == '*':
                regex_parts.append('[^/]*')
            else:
                regex_parts.append(re.escape(token))

        return re.compile(''.join(regex_parts))
        
    def _extract_variables(self, file_path: Path) -> Dict[str, str]:
        """Extract variable values from a matched file path."""
        # Convert path to string relative to data_root for matching
        try:
            rel_path = file_path.relative_to(self.data_root)
        except ValueError:
            # If file_path is not relative to data_root, use absolute path
            rel_path = file_path
            
        path_str = str(rel_path)
        match = self.regex_pattern.match(path_str)
        
        if match:
            return match.groupdict()
        else:
            logger.warning(f"Could not extract variables from {path_str} using pattern {self.pattern}")
            return {}
            
    def _filter_by_criteria(
        self,
        variables: Dict[str, str],
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        acquisitions: Optional[List[str]] = None,
        runs: Optional[List[str]] = None
    ) -> bool:
        """Check if extracted variables match the specified criteria."""
        criteria_map = {
            'subject': subjects,
            'session': sessions,
            'task': tasks,
            'acquisition': acquisitions,
            'run': runs,
        }

        for entity_name, allowed_values in criteria_map.items():
            if allowed_values is None:
                continue
                
            # Check if this entity is in the extracted variables
            if entity_name in variables:
                if variables[entity_name] not in allowed_values:
                    return False
                    
        return True
        
    def find_recordings(
        self,
        subjects: Optional[Union[str, List[str]]] = None,
        sessions: Optional[Union[str, List[str]]] = None,
        tasks: Optional[Union[str, List[str]]] = None,
        acquisitions: Optional[Union[str, List[str]]] = None,
        runs: Optional[Union[str, List[str]]] = None,
        extension: str = '.vhdr'
    ) -> List[Dict[str, Any]]:
        """Find recordings using glob pattern matching.

        Parameters
        ----------
        subjects : str, list of str, or None
            Subject ID(s) to filter by (uses 'subject' variable from pattern)
        sessions : str, list of str, or None
            Session ID(s) to filter by (uses 'session' variable from pattern)
        tasks : str, list of str, or None
            Task(s) to filter by (uses 'task' variable from pattern)
        acquisitions : str, list of str, or None
            Acquisition parameter(s) to filter by (uses 'acquisition' variable from pattern)
        runs : str, list of str, or None
            Run ID(s) to filter by (uses 'run' variable from pattern)
        extension : str
            File extension filter (applied after glob matching)

        Returns
        -------
        list of dict
            List of recording dictionaries with paths and metadata
        """
        if isinstance(subjects, str):
            subjects = [subjects]
        if isinstance(sessions, str):
            sessions = [sessions]
        if isinstance(tasks, str):
            tasks = [tasks]
        if isinstance(acquisitions, str):
            acquisitions = [acquisitions]
        if isinstance(runs, str):
            runs = [runs]

        full_pattern = self.data_root / self.glob_pattern
        matched_files = list(self.data_root.glob(self.glob_pattern))

        logger.info(f"Glob pattern: {full_pattern}")
        logger.info(f"Found {len(matched_files)} file(s) matching pattern")

        recordings_dict = {}

        for file_path in matched_files:
            if not str(file_path).endswith(extension):
                continue

            variables = self._extract_variables(file_path)

            if not self._filter_by_criteria(variables, subjects, sessions, tasks, acquisitions, runs):
                continue
                
            # Group by all named variables — {var} defines separate recordings, * concatenates
            key = tuple(sorted(variables.items()))
            
            if key not in recordings_dict:
                recordings_dict[key] = {
                    'paths': [],
                    'metadata': variables,
                    'recording_name': ' - '.join([f"{k}:{v}" for k, v in sorted(variables.items())])
                }
                
            recordings_dict[key]['paths'].append(file_path)
            
        recordings = list(recordings_dict.values())
        
        logger.info(f"After filtering, {len(recordings)} recording(s) to process")
        
        return recordings
