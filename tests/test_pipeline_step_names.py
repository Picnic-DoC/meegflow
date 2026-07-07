"""Tests for correct step name recording in preprocessing_steps.

Regression tests for bugs where step methods recorded the wrong name
in data['preprocessing_steps'], causing misleading reports.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from conftest import run_step


def _make_raw(n_channels=4, sfreq=100.0, duration=2.0):
    import mne
    n_times = int(sfreq * duration)
    data = np.random.randn(n_channels, n_times) * 1e-6
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


def _make_pipeline():
    from meegflow import MEEGFlowPipeline
    from unittest.mock import MagicMock
    reader = MagicMock()
    return MEEGFlowPipeline(reader=reader, config={})


class TestChunkInEpochStepName:
    """chunk_in_epoch was recording 'epoch' instead of 'chunk_in_epoch'."""

    def test_step_name_is_chunk_in_epoch(self):
        pipeline = _make_pipeline()
        data = {
            "raw": _make_raw(),
            "preprocessing_steps": [],
        }
        result = run_step(pipeline, "chunk_in_epoch", data, {"duration": 1.0})
        recorded_names = [s["step"] for s in result["preprocessing_steps"]]
        assert "chunk_in_epoch" in recorded_names, (
            f"Expected 'chunk_in_epoch' in step names, got {recorded_names}"
        )
        assert "epoch" not in recorded_names, (
            "'epoch' should not appear — that name belongs to _step_epoch"
        )


class TestConcatenateRecordingsErrorMessage:
    """concatenate_recordings was raising an error mentioning 'notch_filter'."""

    def test_error_message_mentions_correct_step(self):
        pipeline = _make_pipeline()
        data = {"preprocessing_steps": []}  # no 'all_raw'
        with pytest.raises(ValueError, match="concatenate_recordings"):
            run_step(pipeline, "concatenate_recordings", data, {})

    def test_error_message_does_not_mention_notch_filter(self):
        pipeline = _make_pipeline()
        data = {"preprocessing_steps": []}
        try:
            run_step(pipeline, "concatenate_recordings", data, {})
        except ValueError as exc:
            assert "notch_filter" not in str(exc), (
                f"Error message should not mention 'notch_filter': {exc}"
            )
