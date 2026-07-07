#!/usr/bin/env python3
"""
Functional tests for the find_flat_channels step with synthetic data.

These tests verify that the find_flat_channels step correctly identifies
flat channels based on variance threshold, handles excluded channels,
and works with different channel types.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from conftest import run_step

try:
    import mne
    from meegflow import MEEGFlowPipeline
    from meegflow.readers import BIDSReader
    MNE_AVAILABLE = True
except ImportError as e:  # pragma: no cover - only when deps are missing
    MNE_AVAILABLE = False
    print(f"Warning: Could not import required modules: {e}")

pytestmark = pytest.mark.skipif(not MNE_AVAILABLE, reason="MNE not available")


def _pipeline():
    return MEEGFlowPipeline(reader=BIDSReader('/tmp'), config={})


def create_test_raw_with_flat_channels():
    """Create test raw data with some flat channels.

    Returns
    -------
    raw : mne.io.RawArray
        Raw data with channels at different variance levels.
    flat_channel_names : list
        Names of channels that should be detected as flat.
    """
    n_channels = 10
    n_times = 1000
    sfreq = 100.0

    # Normal variance for most channels
    data = np.random.randn(n_channels, n_times) * 1e-5

    # Channels 2 and 5 completely flat (constant value)
    data[2, :] = 1e-6
    data[5, :] = -2e-6

    # Channel 7 very low variance (should be detected as flat)
    data[7, :] = np.random.randn(n_times) * 1e-13

    info = mne.create_info(
        ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types='eeg'
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    flat_channel_names = ['EEG002', 'EEG005', 'EEG007']
    return raw, flat_channel_names


def test_find_flat_channels_basic():
    """Basic functionality of find_flat_channels."""
    raw, expected_flat = create_test_raw_with_flat_channels()

    data = {'raw': raw, 'preprocessing_steps': []}
    result = run_step(_pipeline(), "find_flat_channels", data, {'threshold': 1e-12})

    detected_flat = result['preprocessing_steps'][-1]['bad_channels']

    for ch in expected_flat:
        assert ch in detected_flat, f"Expected flat channel {ch} not detected"
        assert ch in result['raw'].info['bads'], f"Flat channel {ch} not added to info['bads']"

    step_info = result['preprocessing_steps'][-1]
    assert step_info['step'] == 'find_flat_channels', "Step name incorrect"
    assert step_info['instance'] == 'raw', "Instance incorrect"
    assert step_info['threshold'] == 1e-12, "Threshold not recorded"
    assert step_info['n_bad_channels'] == len(detected_flat), "Bad channel count incorrect"


def test_find_flat_channels_with_custom_threshold():
    """Custom threshold changes how many channels are flagged."""
    raw, _ = create_test_raw_with_flat_channels()
    data = {'raw': raw, 'preprocessing_steps': []}
    result = run_step(_pipeline(), "find_flat_channels", data, {'threshold': 1e-9})
    detected_flat = result['preprocessing_steps'][-1]['bad_channels']

    # Should detect at least the completely flat channels
    assert len(detected_flat) >= 2, "Should detect at least 2 flat channels"

    raw2, _ = create_test_raw_with_flat_channels()
    data2 = {'raw': raw2, 'preprocessing_steps': []}
    result2 = run_step(_pipeline(), "find_flat_channels", data2, {'threshold': 1e-15})
    detected_flat2 = result2['preprocessing_steps'][-1]['bad_channels']

    # With lower threshold, should detect no more channels
    assert len(detected_flat2) <= len(detected_flat), \
        "Lower threshold should detect fewer channels"


def test_find_flat_channels_no_flat():
    """No detections when all channels have normal variance."""
    n_channels, n_times, sfreq = 5, 1000, 100.0
    data = np.random.randn(n_channels, n_times) * 1e-5
    info = mne.create_info(
        ch_names=[f'EEG{i:03d}' for i in range(n_channels)], sfreq=sfreq, ch_types='eeg'
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    data_dict = {'raw': raw, 'preprocessing_steps': []}
    result = run_step(_pipeline(), "find_flat_channels", data_dict, {'threshold': 1e-12})
    detected_flat = result['preprocessing_steps'][-1]['bad_channels']

    assert len(detected_flat) == 0, "Should not detect any flat channels"
    assert len(result['raw'].info['bads']) == 0, "info['bads'] should be empty"


def test_find_flat_channels_all_flat():
    """All channels flagged when all are flat."""
    n_channels, n_times, sfreq = 5, 1000, 100.0
    data = np.ones((n_channels, n_times)) * 1e-6
    info = mne.create_info(
        ch_names=[f'EEG{i:03d}' for i in range(n_channels)], sfreq=sfreq, ch_types='eeg'
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    data_dict = {'raw': raw, 'preprocessing_steps': []}
    result = run_step(_pipeline(), "find_flat_channels", data_dict, {'threshold': 1e-12})
    detected_flat = result['preprocessing_steps'][-1]['bad_channels']

    assert len(detected_flat) == n_channels, "Should detect all channels as flat"
    assert len(result['raw'].info['bads']) == n_channels, "All channels should be in info['bads']"


def test_find_flat_channels_with_excluded_channels():
    """Excluded channels are not flagged even if flat."""
    raw, _ = create_test_raw_with_flat_channels()
    data = {'raw': raw, 'preprocessing_steps': []}
    excluded = ['EEG002']
    result = run_step(_pipeline(), 
        "find_flat_channels", data, {'threshold': 1e-12, 'excluded_channels': excluded}
    )
    detected_flat = result['preprocessing_steps'][-1]['bad_channels']

    assert 'EEG002' not in detected_flat, "Excluded channel should not be detected"
    assert 'EEG005' in detected_flat or 'EEG007' in detected_flat, \
        "Other flat channels should still be detected"

    step_info = result['preprocessing_steps'][-1]
    assert step_info['excluded_channels'] == excluded, "excluded_channels not recorded correctly"


def test_find_flat_channels_with_picks():
    """picks restricts detection to the requested channel types."""
    n_channels, n_times, sfreq = 10, 1000, 100.0
    ch_names = [f'EEG{i:03d}' for i in range(8)] + ['EOG001', 'EOG002']
    ch_types = ['eeg'] * 8 + ['eog'] * 2

    data = np.random.randn(n_channels, n_times) * 1e-5
    data[2, :] = 1e-6  # EEG002 flat
    data[8, :] = 1e-6  # EOG001 flat

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)

    data_dict = {'raw': raw, 'preprocessing_steps': []}
    result = run_step(_pipeline(), 
        "find_flat_channels", data_dict, {'threshold': 1e-12, 'picks': ['eeg']}
    )
    detected_flat = result['preprocessing_steps'][-1]['bad_channels']

    assert 'EEG002' in detected_flat, "Flat EEG channel should be detected"
    assert 'EOG001' not in detected_flat, "EOG channel should not be checked"

    step_info = result['preprocessing_steps'][-1]
    assert step_info['picks'] == ['eeg'], "picks not recorded correctly"


def test_find_flat_channels_no_duplicate_bads():
    """A pre-existing bad channel is not duplicated in info['bads']."""
    raw, _ = create_test_raw_with_flat_channels()
    raw.info['bads'] = ['EEG002']

    data = {'raw': raw, 'preprocessing_steps': []}
    result = run_step(_pipeline(), "find_flat_channels", data, {'threshold': 1e-12})

    bads_count = result['raw'].info['bads'].count('EEG002')
    assert bads_count == 1, "Channel should not be duplicated in info['bads']"


def test_find_flat_channels_missing_raw():
    """find_flat_channels raises when 'raw' is missing."""
    data = {'preprocessing_steps': []}
    with pytest.raises(ValueError, match="requires 'raw'"):
        run_step(_pipeline(), "find_flat_channels", data, {'threshold': 1e-12})
