#!/usr/bin/env python3
"""
Integration test for the excluded_channels feature using mock data.

Creates minimal mock EEG data and validates that the PipelineContext picking
helpers correctly exclude channels from analysis.
"""

import sys
from pathlib import Path

import pytest

# Find the repository root and add src to the path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

try:
    import numpy as np
    import mne
    from meegflow.readers import BIDSReader
    from meegflow.context import PipelineContext
    MNE_AVAILABLE = True
except ImportError as e:  # pragma: no cover - only when deps are missing
    MNE_AVAILABLE = False
    print(f"Warning: Could not import required modules: {e}")

pytestmark = pytest.mark.skipif(not MNE_AVAILABLE, reason="MNE not available")


def _make_raw():
    ch_names = ['Fz', 'Cz', 'Pz', 'C3', 'C4']
    data = np.random.randn(len(ch_names), 1000) * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=['eeg'] * len(ch_names))
    return mne.io.RawArray(data, info, verbose=False)


def test_excluded_channels_integration():
    """get_picks / _apply_excluded_channels exclude channels as expected."""
    raw = _make_raw()
    ctx = PipelineContext({}, reader=BIDSReader(repo_root / "test_data"))

    # get_picks returns all channels when nothing is excluded
    picks_all = ctx.get_picks(raw.info, None, None)
    assert len(picks_all) == 5, f"Expected 5 channels, got {len(picks_all)}"

    # get_picks excludes the requested channel
    picks_excluded = ctx.get_picks(raw.info, None, ['Cz'])
    assert len(picks_excluded) == 4, f"Expected 4 channels after excluding Cz, got {len(picks_excluded)}"
    assert 'Cz' not in [raw.ch_names[p] for p in picks_excluded], "Cz should not be in excluded picks"

    # _apply_excluded_channels filters a given picks list
    picks = [0, 1, 2, 3, 4]
    filtered_picks = ctx._apply_excluded_channels(raw.info, picks, ['Cz', 'Fz'])
    assert len(filtered_picks) == 3, f"Expected 3 channels after excluding 2, got {len(filtered_picks)}"
    filtered_names = [raw.ch_names[p] for p in filtered_picks]
    assert 'Cz' not in filtered_names and 'Fz' not in filtered_names, \
        "Excluded channels should not be in filtered picks"

    # empty / None exclusion returns all picks
    assert len(ctx._apply_excluded_channels(raw.info, picks, [])) == 5, \
        "Empty exclusion list should return all picks"
    assert len(ctx._apply_excluded_channels(raw.info, picks, None)) == 5, \
        "None exclusion should return all picks"
