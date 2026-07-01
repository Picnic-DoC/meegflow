# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

"""
Adaptive rejection methods for bad channel and epoch detection.

This module provides functions for detecting bad channels and epochs using
adaptive threshold-based and statistical methods. These functions are used
by the preprocessing pipeline's find_bads_* steps.

Functions
---------
find_bads_channels_threshold(epochs, picks, reject, n_epochs_bad_ch)
    Find bad channels based on how often they exceed rejection thresholds.
    Marks channels as bad if they exceed thresholds in too many epochs.

find_bads_channels_variance(inst, picks, zscore_thresh, max_iter)
    Find bad channels with abnormally high or low variance.
    Uses iterative z-score based outlier detection.

find_bads_channels_high_frequency(inst, picks, zscore_thresh, max_iter)
    Find bad channels with excessive high-frequency noise.
    Applies high-pass filter (25 Hz) and detects outliers in standard deviation.

_iteratively_find_outliers(X, threshold, max_iter)
    Internal helper for iterative z-score based outlier detection.
    
Parameters
----------
All functions accept:
  inst/epochs : mne.io.Raw or mne.Epochs
      The MNE data object to analyze
  picks : array-like
      Channel indices to check for bad channels
  zscore_thresh : float (variance/high_frequency methods)
      Z-score threshold for outlier detection (typically 3-4)
  max_iter : int (variance/high_frequency methods)
      Maximum iterations for iterative outlier removal
  reject : dict (threshold method)
      Rejection thresholds by channel type (e.g., {'eeg': 150e-6})
  n_epochs_bad_ch : float or int (threshold method)
      Fraction (0-1) or number of epochs a channel must be bad in

Returns
-------
bad_chs : list of str
    List of channel names identified as bad

Examples
--------
Find bad channels using threshold method:
```python
bad_channels = find_bads_channels_threshold(
    epochs, picks=[0,1,2,3], reject={'eeg': 150e-6}, n_epochs_bad_ch=0.5
)
```

Find bad channels using variance:
```python
bad_channels = find_bads_channels_variance(
    epochs, picks=[0,1,2,3], zscore_thresh=4, max_iter=2
)
```
"""

import math
import numpy as np

from scipy.signal import butter, filtfilt
from scipy.stats import zscore

import mne
from mne.utils import logger


def _iteratively_find_outliers(X, threshold=3.0, max_iter=4):
    """Find outliers based on iterated Z-scoring. """

    X = np.ma.masked_array(X, mask=False)
    for _ in range(max_iter):
        
        X_z = np.abs(zscore(X))
        current_bad = X_z > threshold
        if np.all(~current_bad):
            break
        
        X.mask |= current_bad

    return np.where(X.mask)[0]

def _find_outliers(X, threshold=3.0, max_iter=2, tail=0):
    """Find outliers based on iterated Z-scoring.

    This procedure compares the absolute z-score against the threshold.
    After excluding local outliers, the comparison is repeated until no
    local outlier is present any more.

    Parameters
    ----------
    X : np.ndarray of float, shape (n_elemenets,)
        The scores for which to find outliers.
    threshold : float
        The value above which a feature is classified as outlier.
    max_iter : int
        The maximum number of iterations.
    tail : {0, 1, -1}
        Whether to search for outliers on both extremes of the z-scores (0),
        or on just the positive (1) or negative (-1) side.

    Returns
    -------
    bad_idx : np.ndarray of int, shape (n_features)
        The outlier indices.
    """
    from scipy.stats import zscore
    my_mask = np.zeros(len(X), dtype=bool)
    for _ in range(max_iter):
        X = np.ma.masked_array(X, my_mask)
        if tail == 0:
            this_z = np.abs(zscore(X))
        elif tail == 1:
            this_z = zscore(X)
        elif tail == -1:
            this_z = -zscore(X)
        else:
            raise ValueError("Tail parameter %s not recognised." % tail)
        local_bad = this_z > threshold
        my_mask = np.max([my_mask, local_bad], 0)
        if not np.any(local_bad):
            break

    bad_idx = np.where(my_mask)[0]
    return bad_idx


def find_bads_channels_threshold(epochs, picks, reject, n_epochs_bad_ch=0.5):
    """Identify bad channels that exceed amplitude thresholds in too many epochs.

    A channel is marked bad when its peak-to-peak amplitude exceeds the
    rejection threshold in more than ``n_epochs_bad_ch`` epochs.

    Args:
        epochs: MNE Epochs object to analyse.
        picks: Channel indices to consider.
        reject: Rejection thresholds by channel type, e.g. ``{'eeg': 150e-6}``.
        n_epochs_bad_ch: Fraction (0–1) or absolute count of epochs in which a
            channel must exceed the threshold to be marked bad. Default 0.5.

    Returns:
        List of channel names identified as bad.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]

    if isinstance(n_epochs_bad_ch, float):
        n_epochs_bad_ch = math.floor(n_epochs_bad_ch * n_epochs)

    ch_types_inds = mne.channel_indices_by_type(epochs.info)
    data = np.transpose(data, (1, 0, 2))
    bad_ch_idx = np.ndarray((0,), dtype=int)
    for key, reject_thresh in reject.items():
        idx = np.array([x for x in ch_types_inds[key] if x in picks])
        if len(idx) == 0:
            continue
        count_bad_epochs = np.zeros(len(idx), dtype=int)
        for i_ch, channel in enumerate(data[idx]):
            deltas = channel.max(axis=1) - channel.min(axis=1)
            idx_deltas = np.where(np.greater(deltas, reject_thresh))[0]
            count_bad_epochs[i_ch] = idx_deltas.shape[0]
        # count_bad_epochs is positional within idx; map back to absolute
        # channel indices so the correct channel names are returned.
        reject_bad_channels = idx[np.where(count_bad_epochs > n_epochs_bad_ch)[0]]
        logger.info('Reject by threshold %f on %s %d : bad_channels: %s' %
                    (reject_thresh, key.upper(), len(reject_bad_channels),
                     reject_bad_channels))
        bad_ch_idx = np.concatenate((bad_ch_idx, reject_bad_channels))

    bad_chs = list({epochs.ch_names[i] for i in bad_ch_idx})
    return bad_chs


def find_bads_channels_variance(inst, picks, zscore_thresh=4, max_iter=2):
    """Identify bad channels with abnormally high or low variance.

    Uses iterative z-score outlier detection on per-channel variance computed
    across all time samples.

    Args:
        inst: MNE Raw or Epochs object.
        picks: Channel indices to consider.
        zscore_thresh: Z-score threshold above which a channel is considered an
            outlier. Default 4.
        max_iter: Maximum number of outlier-removal iterations. Default 2.

    Returns:
        List of channel names identified as bad.
    """
    logger.info('Looking for bad channels with variance')
    if isinstance(inst, mne.Epochs):
        data = inst.get_data()
    else:
        data = inst.get_data()[None, :]
    masked_data = np.ma.masked_array(data, fill_value=np.nan)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    ch_var = np.ma.hstack(masked_data).var(axis=-1)
    bad_ch_var = _find_outliers(
        ch_var, threshold=zscore_thresh, max_iter=max_iter)
    logger.info('Reject by variance: bad_channels: %s' % bad_ch_var)
    bad_chs = list({inst.ch_names[i] for i in bad_ch_var})
    return bad_chs


def find_bads_channels_high_frequency(inst, picks, zscore_thresh=4, max_iter=2):
    """Identify bad channels with excessive high-frequency noise.

    Applies a 25 Hz high-pass Butterworth filter, then detects channels whose
    standard deviation of filtered data is a z-score outlier.

    Args:
        inst: MNE Raw or Epochs object.
        picks: Channel indices to consider.
        zscore_thresh: Z-score threshold above which a channel is considered an
            outlier. Default 4.
        max_iter: Maximum number of outlier-removal iterations. Default 2.

    Returns:
        List of channel names identified as bad.
    """
    logger.info('Looking for bad channels with high frequency variance')
    if isinstance(inst, mne.Epochs):
        data = inst.get_data()
    else:
        data = inst.get_data()[None, :]
    masked_data = np.ma.masked_array(data, fill_value=np.nan)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    filter_freq = 25
    b, a = butter(4, 2.0 * filter_freq / inst.info['sfreq'], 'highpass')
    filt_data = filtfilt(b, a, np.ma.hstack(masked_data))
    filt_masked_data = np.ma.masked_array(filt_data, fill_value=np.nan)
    if len(exclude) > 0:
        filt_masked_data[exclude, :] = np.ma.masked
    bad_ch_hf = _find_outliers(
        filt_masked_data.std(axis=-1), threshold=zscore_thresh,
        max_iter=max_iter)
    logger.info('Reject by high frequency std: bad_channels: %s' % bad_ch_hf)
    bad_chs = list({inst.ch_names[i] for i in bad_ch_hf})
    return bad_chs


def find_bads_epochs_threshold(epochs, picks, reject, n_channels_bad_epoch=0.1):
    """Identify bad epochs in which too many channels exceed amplitude thresholds.

    An epoch is marked bad when the number of channels exceeding the rejection
    threshold is greater than ``n_channels_bad_epoch``.

    Args:
        epochs: MNE Epochs object to analyse.
        picks: Channel indices to consider.
        reject: Rejection thresholds by channel type, e.g. ``{'eeg': 150e-6}``.
        n_channels_bad_epoch: Fraction (0–1) or absolute count of channels that
            must exceed the threshold for an epoch to be rejected. Default 0.1.

    Returns:
        Array of epoch indices identified as bad.
    """
    n_channels = len(picks)
    bad_ep_idx = np.ndarray((0,), dtype=int)
    if isinstance(n_channels_bad_epoch, float):
        n_channels_bad_epoch = math.floor(n_channels_bad_epoch * n_channels)

    data = epochs.get_data()
    masked_data = np.ma.masked_array(data, fill_value=np.nan)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    ch_types_inds = mne.channel_indices_by_type(epochs.info)
    n_epochs = masked_data.shape[0]
    for key, reject_thresh in reject.items():
        idx = np.array([x for x in ch_types_inds[key] if x in picks])
        count_bad_chans = np.zeros((n_epochs), dtype=int)
        for i_ep, epoch in enumerate(masked_data[:, idx]):
            deltas = epoch.max(axis=1) - epoch.min(axis=1)
            idx_deltas = np.where(np.greater(deltas, reject_thresh))[0]
            count_bad_chans[i_ep] = idx_deltas.shape[0]
        reject_bad_epochs = np.where(count_bad_chans > n_channels_bad_epoch)[0]
        logger.info('Reject by threshold %f on %s : bad_epochs: %s' %
                    (reject_thresh, key.upper(), reject_bad_epochs))
        # Accumulate across all channel types, not just the last one.
        bad_ep_idx = np.concatenate((bad_ep_idx, reject_bad_epochs))

    bad_epochs = np.unique(bad_ep_idx)

    return bad_epochs

