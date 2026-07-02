#!/usr/bin/env python3
"""
Functional tests for adaptive_reject functions with MNE >= 1.5.0.

These tests verify that the adaptive_reject functions work correctly with
public MNE APIs and do not use deprecated private methods. Any use of a
removed private API surfaces as an exception, which fails the test.
"""

import sys
import inspect
from pathlib import Path

import numpy as np
import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import mne
    from meegflow.steps import adaptive_reject
    MNE_AVAILABLE = True
except ImportError as e:  # pragma: no cover - only when deps are missing
    MNE_AVAILABLE = False
    print(f"Warning: Could not import required modules: {e}")

pytestmark = pytest.mark.skipif(not MNE_AVAILABLE, reason="MNE not available")


def create_test_epochs():
    """Create simple test epochs (with one high-amplitude channel) for testing."""
    n_channels, n_times, sfreq = 5, 1000, 100.0

    data = np.random.randn(n_channels, n_times) * 1e-6
    data[2, :] = np.random.randn(n_times) * 5e-5  # bad channel: high amplitude

    info = mne.create_info(
        ch_names=[f'EEG{i:03d}' for i in range(n_channels)], sfreq=sfreq, ch_types='eeg'
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    n_epochs = 10
    events = np.column_stack([
        np.arange(0, n_epochs * 200, 200),
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])
    epochs = mne.Epochs(
        raw, events, event_id={'test': 1},
        tmin=0.0, tmax=0.5, baseline=None, preload=True, verbose=False
    )
    return epochs, raw


def test_find_bads_channels_threshold():
    """find_bads_channels_threshold runs with public MNE APIs."""
    epochs, _ = create_test_epochs()
    picks = mne.pick_types(epochs.info, eeg=True)
    bad_chs = adaptive_reject.find_bads_channels_threshold(
        epochs, picks, {'eeg': 1e-4}, n_epochs_bad_ch=0.5
    )
    assert isinstance(bad_chs, list)


def test_find_bads_channels_variance():
    """find_bads_channels_variance runs with public MNE APIs."""
    epochs, _ = create_test_epochs()
    picks = mne.pick_types(epochs.info, eeg=True)
    bad_chs = adaptive_reject.find_bads_channels_variance(
        epochs, picks, zscore_thresh=4, max_iter=2
    )
    assert isinstance(bad_chs, list)


def test_find_bads_channels_high_frequency():
    """find_bads_channels_high_frequency runs with public MNE APIs."""
    epochs, _ = create_test_epochs()
    picks = mne.pick_types(epochs.info, eeg=True)
    bad_chs = adaptive_reject.find_bads_channels_high_frequency(
        epochs, picks, zscore_thresh=4, max_iter=2
    )
    assert isinstance(bad_chs, list)


def test_find_bads_epochs_threshold():
    """find_bads_epochs_threshold runs with public MNE APIs."""
    epochs, _ = create_test_epochs()
    picks = mne.pick_types(epochs.info, eeg=True)
    bad_epochs = adaptive_reject.find_bads_epochs_threshold(
        epochs, picks, {'eeg': 1e-4}, n_channels_bad_epoch=0.1
    )
    assert hasattr(bad_epochs, '__len__')


def test_with_raw_data():
    """Variance and high-frequency detection also work on raw data."""
    _, raw = create_test_epochs()
    picks = mne.pick_types(raw.info, eeg=True)

    bad_chs_var = adaptive_reject.find_bads_channels_variance(
        raw, picks, zscore_thresh=4, max_iter=2
    )
    assert isinstance(bad_chs_var, list)

    bad_chs_hf = adaptive_reject.find_bads_channels_high_frequency(
        raw, picks, zscore_thresh=4, max_iter=2
    )
    assert isinstance(bad_chs_hf, list)


def test_no_private_imports():
    """The source must not use removed private MNE APIs."""
    source = inspect.getsource(adaptive_reject)
    assert 'mne.preprocessing.bads._find_outliers' not in source, \
        "Still importing mne.preprocessing.bads._find_outliers"
    assert '._data' not in source, "Still using ._data attribute"
