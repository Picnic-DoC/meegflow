#!/usr/bin/env python3
"""
Tests for BIDS datatype selection on the writing side of the pipeline.

These tests verify that:
1. ``infer_datatype`` maps an MNE instance to its BIDS datatype
2. ``save_clean_instance`` names a saved raw recording after that datatype
3. The reports are filed under that datatype, or under an explicit override
"""

import sys
import tempfile
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
    from meegflow.utils import infer_datatype
    MNE_AVAILABLE = True
except ImportError as e:  # pragma: no cover - only when deps are missing
    MNE_AVAILABLE = False
    print(f"Warning: Could not import required modules: {e}")

pytestmark = pytest.mark.skipif(not MNE_AVAILABLE, reason="MNE not available")


def _pipeline(output_root):
    return MEEGFlowPipeline(
        reader=BIDSReader('/tmp'), output_root=output_root, config={}
    )


def _make_raw(ch_types='eeg', n_channels=4, sfreq=100.0, duration=1.0):
    """Create a short synthetic recording with the requested channel types."""
    n_times = int(sfreq * duration)
    data = np.random.randn(n_channels, n_times) * 1e-6
    ch_names = [f'CH{i:03d}' for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data, info, verbose=False)


class TestInferDatatype:
    def test_eeg_recording(self):
        assert infer_datatype(_make_raw('eeg')) == 'eeg'

    def test_meg_recording(self):
        assert infer_datatype(_make_raw('mag')) == 'meg'

    def test_ieeg_recording(self):
        assert infer_datatype(_make_raw('seeg')) == 'ieeg'

    def test_meg_wins_over_eeg(self):
        # BIDS files a recording holding MEG sensors under 'meg', even when it
        # also holds EEG channels (as in mne.datasets.sample).
        raw = _make_raw(['mag', 'grad', 'eeg', 'eeg'])
        assert infer_datatype(raw) == 'meg'

    def test_falls_back_to_eeg(self):
        # Nothing to infer from: keep what the output steps assumed before.
        assert infer_datatype(None) == 'eeg'
        assert infer_datatype(np.zeros(3)) == 'eeg'
        assert infer_datatype(_make_raw('misc')) == 'eeg'


class TestSaveCleanInstanceSuffix:
    def _save(self, raw, step_config=None):
        with tempfile.TemporaryDirectory() as tmp:
            data = {
                'subject': '01',
                'task': 'rest',
                'raw': raw,
                'preprocessing_steps': [],
            }
            config = {'instance': 'raw', 'format': 'numpy'}
            config.update(step_config or {})
            result = run_step(_pipeline(tmp), 'save_clean_instance', data, config)
            return Path(result['raw_file']).name

    def test_eeg_raw_keeps_eeg_suffix(self):
        assert self._save(_make_raw('eeg')) == 'sub-01_task-rest_eeg.npy'

    def test_meg_raw_gets_meg_suffix(self):
        assert self._save(_make_raw('mag')) == 'sub-01_task-rest_meg.npy'

    def test_explicit_suffix_wins(self):
        name = self._save(_make_raw('mag'), {'suffix': 'eeg'})
        assert name == 'sub-01_task-rest_eeg.npy'


class TestReportDatatype:
    def _report_datatype(self, raw, step_config=None):
        with tempfile.TemporaryDirectory() as tmp:
            data = {
                'subject': '01',
                'task': 'rest',
                'raw': raw,
                'preprocessing_steps': [],
            }
            result = run_step(
                _pipeline(tmp), 'generate_json_report', data, step_config or {}
            )
            report_path = Path(result['json_report'])
            assert report_path.exists(), f"Report not written: {report_path}"
            return report_path.parent.name

    def test_eeg_report_stays_under_eeg(self):
        assert self._report_datatype(_make_raw('eeg')) == 'eeg'

    def test_meg_report_goes_under_meg(self):
        assert self._report_datatype(_make_raw('mag')) == 'meg'

    def test_explicit_datatype_wins(self):
        assert self._report_datatype(_make_raw('mag'), {'datatype': 'eeg'}) == 'eeg'

    def test_report_without_data_falls_back_to_eeg(self):
        from meegflow.report import _resolve_report_datatype
        assert _resolve_report_datatype({'subject': '01', 'task': 'rest'}, {}) == 'eeg'
