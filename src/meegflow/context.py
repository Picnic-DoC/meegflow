"""Per-recording execution context shared by all pipeline steps.

``PipelineContext`` is a dict-like object (``MutableMapping``) wrapping the
shared ``data`` bag that steps read from and write to, plus the services a step
may need (channel picking, the derivatives root, provenance recording). Passing
one object to every step — built-in or custom — gives all steps the same
first-class access to those services.
"""
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, List, Union

import mne
from mne.utils import logger

if False:  # typing-only, avoids an import cycle
    from .readers import DatasetReader


class PipelineContext(MutableMapping):
    def __init__(self, data: Dict[str, Any], reader: "DatasetReader",
                 output_root: Union[str, Path] = None, config: Dict[str, Any] = None):
        self.data = data
        self.reader = reader
        self.output_root = Path(output_root) if output_root else None
        self.config = config or {}

    # ------------------------------------------------------------------ #
    # MutableMapping interface over the shared data bag                   #
    # ------------------------------------------------------------------ #
    def __getitem__(self, key): return self.data[key]
    def __setitem__(self, key, value): self.data[key] = value
    def __delitem__(self, key): del self.data[key]
    def __iter__(self): return iter(self.data)
    def __len__(self): return len(self.data)
    def __contains__(self, key): return key in self.data

    # ------------------------------------------------------------------ #
    # Services (ported verbatim from MEEGFlowPipeline)                   #
    # ------------------------------------------------------------------ #

    @property
    def dataset_root(self) -> Path:
        """Get the dataset root path from the reader."""
        return self.reader.root

    def derivatives_root(self, subdir: str = "") -> Path:
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

    def get_picks(self, info: mne.Info, picks_params: Any, excluded_channels: List[str] = None) -> List[int]:
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
