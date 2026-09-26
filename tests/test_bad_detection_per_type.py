#!/usr/bin/env python3
"""
Tests for bad-channel detection on recordings with several channel types.

These tests verify that:
1. ``find_flat_channels`` uses one variance threshold per channel type, so
   MEG channels (variances around 1e-26 T^2) are not all declared flat
2. A number or a per-type dict can still be given as ``threshold``
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from conftest import run_step

import mne
from meegflow import MEEGFlowPipeline
from meegflow.readers import BIDSReader

SCALES = {'mag': 2e-13, 'grad': 5e-12, 'eeg': 2e-5}


def _pipeline(config=None):
    return MEEGFlowPipeline(reader=BIDSReader('/tmp'), output_root='/tmp', config=config or {})


def _names(ch_types):
    return [f'{t.upper()}{i:03d}' for i, t in enumerate(ch_types)]


def _raw(ch_types, flat=(), near_flat=(), n_times=1000):
    """Noise at each type's typical scale; ``flat`` rows zeroed, ``near_flat`` tiny."""
    rng = np.random.RandomState(0)
    scale = np.array([SCALES[t] for t in ch_types])[:, None]
    data = rng.randn(len(ch_types), n_times) * scale
    names = _names(ch_types)
    for ch in flat:
        data[names.index(ch)] = 0.0
    for ch in near_flat:
        data[names.index(ch)] = rng.randn(n_times) * 1e-17
    info = mne.create_info(names, 100.0, ch_types)
    return mne.io.RawArray(data, info, verbose=False)


def _data(**entries):
    data = {'subject': '01', 'task': 'rest', 'preprocessing_steps': []}
    data.update(entries)
    return data


MIXED = ['mag'] * 6 + ['grad'] * 6 + ['eeg'] * 6


class TestFlatChannelsPerType:
    def test_default_flags_only_the_flat_channel_of_each_type(self):
        flat = ['MAG001', 'GRAD007', 'EEG013']
        result = run_step(_pipeline(), 'find_flat_channels', _data(raw=_raw(MIXED, flat=flat)),
                          {'picks': ['meg', 'eeg']})
        step = result['preprocessing_steps'][-1]
        assert sorted(step['bad_channels']) == sorted(flat)
        assert step['threshold'] == {'mag': 1e-30, 'grad': 1e-26, 'eeg': 1e-12}

    def test_eeg_defaults_unchanged(self):
        # Default picks (EEG) and the historical 1e-12 threshold for EEG.
        result = run_step(_pipeline(), 'find_flat_channels',
                          _data(raw=_raw(MIXED, flat=['EEG013'])), {})
        step = result['preprocessing_steps'][-1]
        assert step['bad_channels'] == ['EEG013']
        assert step['threshold'] == {'eeg': 1e-12}

    def test_meg_datatype_checks_meg_by_default(self):
        result = run_step(_pipeline({'datatype': 'meg'}), 'find_flat_channels',
                          _data(raw=_raw(MIXED, flat=['GRAD007'])), {})
        assert result['preprocessing_steps'][-1]['bad_channels'] == ['GRAD007']

    def test_scalar_threshold_applies_to_every_channel(self):
        # The old single threshold, given explicitly, still works as before:
        # on MEG it flags every channel, which is why the default is per type.
        result = run_step(_pipeline(), 'find_flat_channels', _data(raw=_raw(MIXED)),
                          {'picks': ['meg'], 'threshold': 1e-12})
        step = result['preprocessing_steps'][-1]
        assert step['n_bad_channels'] == 12
        assert step['threshold'] == 1e-12

    def test_dict_threshold_overrides_one_type(self):
        raw_kwargs = dict(near_flat=['MAG001'])
        default = run_step(_pipeline(), 'find_flat_channels', _data(raw=_raw(MIXED, **raw_kwargs)),
                           {'picks': ['meg']})
        assert default['preprocessing_steps'][-1]['bad_channels'] == ['MAG001']
        stricter = run_step(_pipeline(), 'find_flat_channels', _data(raw=_raw(MIXED, **raw_kwargs)),
                            {'picks': ['meg'], 'threshold': {'mag': 1e-36}})
        step = stricter['preprocessing_steps'][-1]
        assert step['bad_channels'] == []
        assert step['threshold'] == {'mag': 1e-36, 'grad': 1e-26}
