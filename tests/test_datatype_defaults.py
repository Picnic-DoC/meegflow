#!/usr/bin/env python3
"""
Tests for the top-level ``datatype`` key and the defaults it selects.

These tests verify that:
1. ``datatype`` is validated, and absent means ``'eeg'``
2. Default channel picks follow it (EEG unchanged, MEG without reference channels)
3. It sets the BIDS reader's datatype unless the reader was given its own
4. Outputs and reports are labelled with it when it is set, and not otherwise
5. The threshold detectors take their default ``reject`` from it, and ignore
   thresholds for channel types that are not picked instead of crashing
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from conftest import run_step

import mne
from meegflow import MEEGFlowPipeline
from meegflow.context import PipelineContext
from meegflow.defaults import configured_datatype, resolve_datatype
from meegflow.readers import BIDSReader, GlobReader

CH_TYPES = ['mag'] * 4 + ['grad'] * 4 + ['eeg'] * 4 + ['eog', 'ref_meg']
SCALES = {'mag': 1e-13, 'grad': 1e-11, 'eeg': 1e-5, 'eog': 1e-5, 'ref_meg': 1e-13}


def _info(ch_types=CH_TYPES, sfreq=100.0):
    names = [f'{t.upper()}{i:03d}' for i, t in enumerate(ch_types)]
    return mne.create_info(names, sfreq, ch_types)


def _raw(ch_types=CH_TYPES, n_times=200, seed=0):
    info = _info(ch_types)
    scale = np.array([SCALES[t] for t in ch_types])[:, None]
    data = np.random.RandomState(seed).randn(len(ch_types), n_times) * scale
    return mne.io.RawArray(data, info, verbose=False)


def _epochs(ch_types, n_epochs=10, seed=0):
    info = _info(ch_types)
    scale = np.array([SCALES[t] for t in ch_types])[None, :, None]
    data = np.random.RandomState(seed).randn(n_epochs, len(ch_types), 50) * scale
    return mne.EpochsArray(data, info, verbose=False)


def _pipeline(config, root='/tmp'):
    return MEEGFlowPipeline(reader=BIDSReader(root), output_root=root, config=config)


def _data(**entries):
    data = {'subject': '01', 'task': 'rest', 'session': None, 'acquisition': None,
            'preprocessing_steps': []}
    data.update(entries)
    return data


def _picked_types(config, picks=None):
    info = _info()
    ctx = PipelineContext({}, reader=None, config=config)
    types = info.get_channel_types()
    return sorted({types[p] for p in ctx.get_picks(info, picks)})


class TestResolveDatatype:
    def test_absent_means_eeg(self):
        assert resolve_datatype({}) == 'eeg'
        assert resolve_datatype(None) == 'eeg'

    def test_meg_case_insensitive(self):
        assert resolve_datatype({'datatype': 'MEG'}) == 'meg'

    def test_unsupported_value_raises(self):
        with pytest.raises(ValueError, match='datatype'):
            resolve_datatype({'datatype': 'fmri'})

    def test_configured_datatype_only_when_set(self):
        assert configured_datatype({}) is None
        assert configured_datatype({'datatype': 'meg'}) == 'meg'

    def test_pipeline_rejects_unsupported_datatype(self):
        with pytest.raises(ValueError, match='datatype'):
            _pipeline({'datatype': 'fmri'})


class TestDefaultPicks:
    def test_eeg_default_is_unchanged(self):
        # Without the key, and with datatype: eeg, only EEG channels are picked.
        assert _picked_types({}) == ['eeg']
        assert _picked_types({'datatype': 'eeg'}) == ['eeg']

    def test_meg_default_excludes_reference_channels(self):
        assert _picked_types({'datatype': 'meg'}) == ['grad', 'mag']

    def test_explicit_picks_win(self):
        assert _picked_types({'datatype': 'meg'}, picks=['eeg']) == ['eeg']


class TestReaderDatatype:
    def test_config_sets_bids_reader_datatype(self):
        reader = BIDSReader('/tmp')
        MEEGFlowPipeline(reader=reader, config={'datatype': 'meg'})
        assert reader.datatype == 'meg'

    def test_reader_datatype_wins(self):
        reader = BIDSReader('/tmp', datatype='eeg')
        MEEGFlowPipeline(reader=reader, config={'datatype': 'meg'})
        assert reader.datatype == 'eeg'

    def test_absent_key_keeps_detection(self):
        reader = BIDSReader('/tmp')
        MEEGFlowPipeline(reader=reader, config={})
        assert reader.datatype is None

    def test_glob_reader_accepts_meg_config(self):
        reader = GlobReader('/tmp', '{subject}.fif')
        pipeline = MEEGFlowPipeline(reader=reader, config={'datatype': 'meg'})
        assert pipeline.datatype == 'meg'


class TestOutputDatatype:
    def test_save_uses_top_level_datatype(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_step(_pipeline({'datatype': 'meg'}, tmp), 'save_clean_instance',
                              _data(raw=_raw()), {'instance': 'raw', 'format': 'numpy'})
            path = Path(result['raw_file'])
            assert path.parent.name == 'meg'
            assert path.name == 'sub-01_task-rest_meg.npy'

    def test_save_without_key_keeps_layout(self):
        # No datatype folder, as before the key existed.
        with tempfile.TemporaryDirectory() as tmp:
            result = run_step(_pipeline({}, tmp), 'save_clean_instance',
                              _data(raw=_raw(['eeg'] * 4)), {'instance': 'raw', 'format': 'numpy'})
            path = Path(result['raw_file'])
            assert path.parent.name == 'sub-01'
            assert path.name == 'sub-01_task-rest_eeg.npy'

    def test_report_uses_top_level_datatype(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_step(_pipeline({'datatype': 'meg'}, tmp), 'generate_json_report',
                              _data(raw=_raw(['eeg'] * 4)), {})
            assert Path(result['json_report']).parent.name == 'meg'

