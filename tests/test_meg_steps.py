#!/usr/bin/env python3
"""
Tests for the MEG steps and for MEG runs end to end, on real MEG recordings.

The recordings come from MNE's testing dataset
(``mne.datasets.testing.data_path()``): a 20 s excerpt of the MNE sample data
(Elekta Neuromag, 102 magnetometers, 204 gradiometers, 60 EEG), a recording
with continuous head-position indicator (cHPI) coils, the matching
fine-calibration and cross-talk files, and a CTF recording. The tests are
skipped when the dataset is not available; CI downloads it.

These tests verify that:
1. ``maxwell_filter`` applies SSS, tSSS and movement compensation
2. ``find_bads_maxwell`` marks noisy and flat MEG channels as bad
3. ``compute_head_pos`` estimates head positions from cHPI and writes them
4. ``compute_ssp`` adds ECG and EOG projectors, with a synthetic ECG
5. ``apply_gradient_compensation`` changes the CTF compensation grade
6. The MEG steps refuse data without MEG channels
7. The HTML report draws MEG bad channels
8. A ``datatype: meg`` pipeline runs end to end through both readers
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from conftest import run_step

import mne
from meegflow import MEEGFlowPipeline
from meegflow import report
from meegflow.readers import BIDSReader, GlobReader

mne.set_log_level('ERROR')


def _testing_path():
    """The MNE testing dataset, or None if it has not been downloaded."""
    try:
        path = Path(mne.datasets.testing.data_path(download=False))
    except Exception:  # pragma: no cover - depends on the environment
        return None
    # data_path returns an empty path, not None, when the dataset is missing,
    # and Path('') is the current directory, so check for an actual recording.
    sample = path / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'
    return path if str(path) not in ('', '.') and sample.exists() else None


TESTING = _testing_path()
pytestmark = pytest.mark.skipif(TESTING is None, reason="MNE testing data not available")

if TESTING is not None:
    SAMPLE = TESTING / 'MEG' / 'sample' / 'sample_audvis_trunc_raw.fif'
    CHPI = TESTING / 'SSS' / 'test_move_anon_raw.fif'
    CALIBRATION = str(TESTING / 'SSS' / 'sss_cal_mgh.dat')
    CROSS_TALK = str(TESTING / 'SSS' / 'ct_sparse_mgh.fif')
    CTF = TESTING / 'CTF' / 'testdata_ctf.ds'


def _pipeline(root, config=None):
    return MEEGFlowPipeline(reader=BIDSReader(root), output_root=root,
                            config=config or {'datatype': 'meg'})


def _data(**entries):
    data = {'subject': '01', 'task': 'test', 'session': None, 'acquisition': None,
            'preprocessing_steps': []}
    data.update(entries)
    return data


@pytest.fixture
def sample_raw():
    return mne.io.read_raw_fif(SAMPLE).load_data()


@pytest.fixture
def tmp_root():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


class TestMaxwell:
    def test_sss(self, sample_raw, tmp_root):
        result = run_step(_pipeline(tmp_root), 'maxwell_filter', _data(raw=sample_raw),
                          {'calibration': CALIBRATION, 'cross_talk': CROSS_TALK})
        max_info = result['raw'].info['proc_history'][0]['max_info']
        assert max_info['sss_info']['nfree'] > 0
        step = result['preprocessing_steps'][-1]
        assert step['step'] == 'maxwell_filter'
        assert step['params']['calibration'] == CALIBRATION
        assert step['movement_compensation'] is False

    def test_tsss(self, sample_raw, tmp_root):
        result = run_step(_pipeline(tmp_root), 'maxwell_filter', _data(raw=sample_raw),
                          {'st_duration': 4.0, 'st_overlap': False})
        max_st = result['raw'].info['proc_history'][0]['max_info']['max_st']
        assert max_st['buflen'] == pytest.approx(4.0, abs=0.01)

    def test_find_bads_maxwell_marks_bads(self, sample_raw, tmp_root):
        # A dead gradiometer must be reported as flat.
        sample_raw._data[sample_raw.ch_names.index('MEG 0113')] = 0.0
        result = run_step(_pipeline(tmp_root), 'find_bads_maxwell', _data(raw=sample_raw),
                          {'calibration': CALIBRATION, 'cross_talk': CROSS_TALK})
        step = result['preprocessing_steps'][-1]
        assert 'MEG 0113' in step['flat_channels']
        assert 'MEG 0113' in result['raw'].info['bads']
        assert step['n_bad_channels'] == len(step['bad_channels'])


class TestHeadPosition:
    @pytest.fixture
    def chpi_raw(self):
        return mne.io.read_raw_fif(CHPI, allow_maxshield='yes').crop(0, 4).load_data()

    def test_compute_save_and_compensate(self, chpi_raw, tmp_root):
        pipeline = _pipeline(tmp_root)
        data = run_step(pipeline, 'compute_head_pos', _data(raw=chpi_raw), {'save': True})
        head_pos = data['head_pos']
        assert head_pos.ndim == 2 and head_pos.shape[1] == 10 and len(head_pos) > 0
        assert Path(data['head_pos_file']).exists()
        assert data['preprocessing_steps'][-1]['n_positions'] == len(head_pos)

        data = run_step(pipeline, 'maxwell_filter', data, {'head_pos': 'head_pos', 'origin': (0.0, 0.0, 0.04)})
        assert data['preprocessing_steps'][-1]['movement_compensation'] is True

    def test_read_pos_file(self, chpi_raw, tmp_root):
        pipeline = _pipeline(tmp_root)
        saved = run_step(pipeline, 'compute_head_pos', _data(raw=chpi_raw.copy()), {'save': True})
        read = run_step(pipeline, 'compute_head_pos', _data(raw=chpi_raw),
                        {'pos_file': saved['head_pos_file'], 'var_name': 'pos'})
        np.testing.assert_allclose(read['pos'], saved['head_pos'], atol=1e-5)


class TestSSP:
    def test_ecg_projectors_from_synthetic_ecg(self, sample_raw, tmp_root):
        # The sample data have no ECG channel: MNE builds one from the magnetometers.
        n_before = len(sample_raw.info['projs'])
        result = run_step(_pipeline(tmp_root), 'compute_ssp', _data(raw=sample_raw),
                          {'artifact': 'ecg', 'n_eeg': 0})
        step = result['preprocessing_steps'][-1]
        assert step['n_events'] > 0
        assert len(step['projectors']) == 4
        assert all(p.startswith('ECG-') for p in step['projectors'])
        assert len(result['raw'].info['projs']) == n_before + 4

    def test_eog_projectors_applied(self, sample_raw, tmp_root):
        result = run_step(_pipeline(tmp_root), 'compute_ssp', _data(raw=sample_raw),
                          {'artifact': 'eog', 'n_eeg': 0, 'apply': True})
        step = result['preprocessing_steps'][-1]
        assert step['applied'] is True
        assert all(p['active'] for p in result['raw'].info['projs'])

    def test_artifact_is_required(self, sample_raw, tmp_root):
        with pytest.raises(ValueError, match='artifact'):
            run_step(_pipeline(tmp_root), 'compute_ssp', _data(raw=sample_raw), {})


class TestGradientCompensation:
    def test_ctf_grade_changes(self, tmp_root):
        raw = mne.io.read_raw_ctf(CTF).crop(0, 1).load_data()
        result = run_step(_pipeline(tmp_root), 'apply_gradient_compensation', _data(raw=raw),
                          {'grade': 3})
        assert result['raw'].compensation_grade == 3
        step = result['preprocessing_steps'][-1]
        assert step['grade'] == 3 and step['previous_grade'] == 0


@pytest.mark.parametrize('step_name, config', [
    ('maxwell_filter', {}),
    ('find_bads_maxwell', {}),
    ('compute_head_pos', {}),
])
def test_meg_steps_require_meg_channels(step_name, config, tmp_root):
    info = mne.create_info(['EEG1', 'EEG2'], 100.0, 'eeg')
    raw = mne.io.RawArray(np.zeros((2, 100)), info, verbose=False)
    with pytest.raises(ValueError, match='MEG channels'):
        run_step(_pipeline(tmp_root), step_name, _data(raw=raw), config)


class TestReportBadChannelMap:
    def test_one_panel_per_sensor_type_with_bads(self, sample_raw):
        fig = report.create_bad_channels_topoplot(
            sample_raw.info, ['MEG 0113', 'MEG 0112', 'MEG 0121', 'EEG 001'])
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert titles == ['EEG (n=1)', 'Magnetometers (n=1)', 'Gradiometers (n=2)']

    def test_meg_only_bads(self, sample_raw):
        fig = report.create_bad_channels_topoplot(sample_raw.info, ['MEG 0111'])
        assert fig is not None
        assert [ax.get_title() for ax in fig.axes if ax.get_title()] == ['Bad Channels (n=1)']

    def test_non_sensor_bads_give_no_figure(self, sample_raw):
        assert report.create_bad_channels_topoplot(sample_raw.info, ['STI 014']) is None


MEG_PIPELINE = [
    {'name': 'concatenate_recordings'},
    {'name': 'find_bads_maxwell', 'calibration': None, 'cross_talk': None},
    {'name': 'maxwell_filter', 'calibration': None, 'cross_talk': None},
    {'name': 'bandpass_filter', 'l_freq': 1.0, 'h_freq': 40.0},
    {'name': 'find_flat_channels'},
    {'name': 'chunk_in_epoch', 'duration': 2.0},
    {'name': 'find_bads_channels_variance'},
    {'name': 'find_bads_epochs_threshold'},
    {'name': 'save_clean_instance', 'instance': 'epochs'},
    {'name': 'generate_json_report'},
]


def _meg_config():
    pipeline = [dict(step) for step in MEG_PIPELINE]
    for step in pipeline:
        if 'calibration' in step:
            step['calibration'], step['cross_talk'] = CALIBRATION, CROSS_TALK
    return {'datatype': 'meg', 'pipeline': pipeline}


def _check_meg_outputs(result, root):
    assert 'error' not in result, result.get('error')
    assert Path(result['epochs_file']).relative_to(root).parts[:3] == ('epochs', 'sub-01', 'meg')
    assert Path(result['json_report']).parent.name == 'meg'
    epochs = mne.read_epochs(result['epochs_file'])
    assert set(epochs.get_channel_types()) >= {'mag', 'grad'}
    steps = [s['step'] for s in result['preprocessing_steps']]
    assert steps[:2] == ['concatenate_recordings', 'find_bads_maxwell']


def test_meg_pipeline_glob_reader():
    with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as out:
        target = Path(data_root) / 'sub-01' / 'sub-01_task-audvis_raw.fif'
        target.parent.mkdir()
        shutil.copy(SAMPLE, target)
        reader = GlobReader(data_root, 'sub-{subject}/sub-{subject}_task-{task}_raw.fif')
        pipeline = MEEGFlowPipeline(reader=reader, output_root=out, config=_meg_config())
        results = pipeline.run_pipeline(subjects=['01'], extension='.fif', io_backend='read_raw_fif')
        _check_meg_outputs(results['01'][0], out)


def test_meg_pipeline_bids_reader():
    mne_bids = pytest.importorskip('mne_bids')
    with tempfile.TemporaryDirectory() as bids_root, tempfile.TemporaryDirectory() as out:
        raw = mne.io.read_raw_fif(SAMPLE)
        bids_path = mne_bids.BIDSPath(subject='01', task='audvis', datatype='meg', root=bids_root)
        mne_bids.write_raw_bids(raw, bids_path, verbose=False)
        # The reader is given no datatype: the configuration's datatype: meg selects it.
        reader = BIDSReader(bids_root)
        pipeline = MEEGFlowPipeline(reader=reader, output_root=out, config=_meg_config())
        assert reader.datatype == 'meg'
        results = pipeline.run_pipeline(subjects=['01'], tasks=['audvis'], extension='.fif')
        _check_meg_outputs(results['01'][0], out)
