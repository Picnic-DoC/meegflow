import numpy as np
import mne
from .registry import register
from . import adaptive_reject
from ..defaults import DEFAULT_REJECT, FLAT_VARIANCE, FLAT_VARIANCE_FALLBACK
from mne.utils import logger


def _datatype(data):
    """Top-level datatype of the run (``'eeg'`` for a plain data dict)."""
    return getattr(data, 'datatype', 'eeg')


def _group_picks_by_type(info, picks):
    """Split channel indices by channel type, keeping their order."""
    ch_types = info.get_channel_types()
    groups = {}
    for p in picks:
        groups.setdefault(ch_types[p], []).append(int(p))
    return groups


def _detect_per_channel_type(detect, inst, picks, zscore_thresh, max_iter):
    """Run a z-score detector separately on each channel type in ``picks``.

    Channel types are recorded in different units and at different scales
    (Neuromag magnetometer and gradiometer variances differ by about three
    orders of magnitude), so pooling them into one z-score hides outliers
    within a type. A type with a single channel cannot be an outlier among
    its own kind and is skipped.
    """
    bad_chs, by_type = [], {}
    for ch_type, idx in _group_picks_by_type(inst.info, picks).items():
        if len(idx) < 2:
            logger.info(f"Skipping channel type '{ch_type}': fewer than 2 channels")
            continue
        found = sorted(detect(inst, idx, zscore_thresh, max_iter), key=inst.ch_names.index)
        by_type[ch_type] = found
        bad_chs.extend(ch for ch in found if ch not in bad_chs)
    return bad_chs, by_type


def _flat_thresholds(ch_types, threshold):
    """Variance threshold of each picked channel, and a record of what was used.

    ``threshold`` may be a number (one threshold for every channel), a dict
    by channel type (types not listed use the defaults), or None (the
    per-type defaults).
    """
    if threshold is not None and not isinstance(threshold, dict):
        return np.full(len(ch_types), float(threshold)), threshold
    per_type = dict(FLAT_VARIANCE)
    per_type.update(threshold or {})
    used = {t: per_type.get(t, FLAT_VARIANCE_FALLBACK) for t in dict.fromkeys(ch_types)}
    return np.array([used[t] for t in ch_types]), used


def _usable_reject(data, info, picks, reject):
    """Rejection thresholds restricted to the channel types among ``picks``.

    With ``reject`` omitted, the thresholds of the top-level datatype are
    used. A threshold for a type with no picked channel is dropped rather
    than passed on, since it has nothing to act on.
    """
    if reject is None:
        reject = DEFAULT_REJECT[_datatype(data)]
    present = set(_group_picks_by_type(info, picks))
    usable = {t: v for t, v in reject.items() if t in present}
    dropped = sorted(set(reject) - set(usable))
    if dropped:
        logger.warning(f"No picked channels of type {dropped}; their rejection thresholds are ignored")
    return usable



@register("find_flat_channels")
def find_flat_channels(data, step_config):
    """
    Find flat channels based on variance threshold.
    
    Flat channels often indicate disconnected electrodes or other hardware issues.
    Channels with variance below the threshold are marked as bad.
    
    Parameters (via step_config)
    -----------------------------
    picks : list, optional
        Channel types to analyze (default: all MEEG channels)
    excluded_channels : list, optional
        Channel names to exclude from analysis (e.g., reference channels)
    threshold : float or dict, optional
        Variance below which a channel is considered flat, in SI units
        squared. A number applies to every picked channel; a dict gives one
        threshold per channel type. By default (or for types missing from a
        dict): 1e-30 for magnetometers and MEG reference channels, 1e-26 for
        gradiometers, and 1e-12 for every other type (EEG, EOG, ...)
    
    Updates
    -------
    data['raw'].info['bads'] : list
        Adds detected flat channels (without duplicates)
    data['preprocessing_steps'] : list
        Appends step information including detected bad channels
    
    Returns
    -------
    data : dict
        Updated data dictionary with flat channels marked as bad
    """
    if 'raw' not in data:
        raise ValueError("find_flat_channels requires 'raw' in data")

    picks_params = step_config.get('picks', None)
    excluded_channels = step_config.get('excluded_channels', None)
    threshold = step_config.get('threshold', None)
    
    # Get picks with exclusions
    picks = data.get_picks(data['raw'].info, picks_params, excluded_channels)

    # Get data only for selected picks
    raw_data = data['raw'].get_data(picks=picks)
    variances = raw_data.var(axis=1)
    ch_types = [data['raw'].get_channel_types()[p] for p in picks]
    thresholds, threshold_used = _flat_thresholds(ch_types, threshold)
    flat_idx = np.where(variances < thresholds)[0]
    # Map back to channel names using picks
    flat_chs = [data['raw'].ch_names[picks[i]] for i in flat_idx]
    
    if flat_chs:
        data['raw'].info['bads'].extend([ch for ch in flat_chs if ch not in data['raw'].info['bads']])
    
    data['preprocessing_steps'].append({
        'step': 'find_flat_channels',
        'instance': 'raw',
        'picks': picks_params,
        'excluded_channels': excluded_channels,
        'apply_on': ['raw'],
        'threshold': threshold_used,
        'bad_channels': flat_chs,
        'n_bad_channels': len(flat_chs)
    })

    return data


@register("find_bads_channels_threshold")
def find_bads_channels_threshold(data, step_config):
    """Find bad channels that exceed amplitude thresholds across epochs.

    Delegates to :func:`adaptive_reject.find_bads_channels_threshold` and
    marks detected channels as bad in all instances listed in
    ``step_config['apply_on']``.

    Args:
        data: Pipeline data dict. Must contain ``'epochs'``.
        step_config: Step parameters:
            - ``picks`` (list|None): Channel types to consider. Default all.
            - ``excluded_channels`` (list|None): Channels to skip.
            - ``reject`` (dict): Amplitude thresholds per channel type,
              e.g. ``{'eeg': 100e-6}``. Default: from the top-level
              ``datatype``, ``{'eeg': 100e-6}`` for EEG and
              ``{'mag': 4e-12, 'grad': 4e-10}`` for MEG.
            - ``n_epochs_bad_ch`` (float): Fraction of epochs in which a
              channel must exceed the threshold to be marked bad. Default 0.5.
            - ``apply_on`` (list): Instances to mark bad channels on.
              Default ``['epochs']``.

    Returns:
        Updated data dict with bad channels added to ``info['bads']``.

    Raises:
        ValueError: If ``'epochs'`` or any ``apply_on`` instance is absent.
    """
    if 'epochs' not in data:
        raise ValueError("find_bads_channels_threshold requires 'epochs' in data")

    picks_params = step_config.get('picks', None)
    excluded_channels = step_config.get('excluded_channels', None)
    reject = step_config.get('reject', None)
    n_epochs_bad_ch = step_config.get('n_epochs_bad_ch', 0.5)
    apply_on = step_config.get('apply_on', ['epochs'])

    if not isinstance(apply_on, list):
        apply_on = [apply_on]

    if any(inst not in data for inst in apply_on):
        raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

    picks = data.get_picks(data['epochs'].info, picks_params, excluded_channels)

    reject = _usable_reject(data, data['epochs'].info, picks, reject)
    bad_chs = adaptive_reject.find_bads_channels_threshold(
        data['epochs'], picks, reject, n_epochs_bad_ch
    ) if reject else []

    if bad_chs:
        for instance_to_apply in apply_on:
            data[instance_to_apply].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_to_apply].info['bads']])

    data['preprocessing_steps'].append({
        'step': 'find_bads_channels_threshold',
        'picks': picks_params,
        'excluded_channels': excluded_channels,
        'apply_on': apply_on,
        'reject': reject,
        'n_epochs_bad_ch': n_epochs_bad_ch,
        'bad_channels': bad_chs,
        'n_bad_channels': len(bad_chs)
    })

    return data


@register("find_bads_channels_variance")
def find_bads_channels_variance(data, step_config):
    """Find bad channels with abnormal variance using z-score outlier detection.

    Delegates to :func:`adaptive_reject.find_bads_channels_variance` and
    marks detected channels as bad in all instances listed in
    ``step_config['apply_on']``.

    Args:
        data: Pipeline data dict. Must contain the key specified by
            ``step_config['instance']`` (default ``'epochs'``).
        step_config: Step parameters:
            - ``instance`` (str): Data key to analyse. Default ``'epochs'``.
            - ``picks`` (list|None): Channel types to consider. Default all.
            - ``excluded_channels`` (list|None): Channels to skip.
            - ``zscore_thresh`` (float): Z-score threshold. Default 4.
            - ``max_iter`` (int): Maximum outlier-removal iterations.
              Default 2.
            - ``apply_on`` (list): Instances to mark bad channels on.
              Defaults to ``[instance]``.

    Returns:
        Updated data dict with bad channels added to ``info['bads']``.

    Raises:
        ValueError: If the requested instance or any ``apply_on`` key is absent.
    """
    # Check which instance to use
    instance = step_config.get('instance', 'epochs')
    if instance not in data:
        raise ValueError(f"find_bads_channels_variance requires '{instance}' in data")

    picks_params = step_config.get('picks', None)
    excluded_channels = step_config.get('excluded_channels', None)
    zscore_thresh = step_config.get('zscore_thresh', 4)
    max_iter = step_config.get('max_iter', 2)
    apply_on = step_config.get('apply_on', [instance])

    if not isinstance(apply_on, list):
        apply_on = [apply_on]

    if any(inst not in data for inst in apply_on):
        raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

    picks = data.get_picks(data[instance].info, picks_params, excluded_channels)

    bad_chs, bad_chs_by_type = _detect_per_channel_type(
        adaptive_reject.find_bads_channels_variance,
        data[instance], picks, zscore_thresh, max_iter
    )

    # Mark channels as bad
    if bad_chs:
        for instance_to_apply in apply_on:
            data[instance_to_apply].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_to_apply].info['bads']])

    data['preprocessing_steps'].append({
        'step': 'find_bads_channels_variance',
        'bad_channels_by_type': bad_chs_by_type,
        'instance': instance,
        'picks': picks_params,
        'excluded_channels': excluded_channels,
        'apply_on': apply_on,
        'zscore_thresh': zscore_thresh,
        'max_iter': max_iter,
        'bad_channels': bad_chs,
        'n_bad_channels': len(bad_chs)
    })

    return data


@register("find_bads_channels_high_frequency")
def find_bads_channels_high_frequency(data, step_config):
    """Find bad channels with excessive high-frequency noise.

    Applies a 25 Hz high-pass filter and identifies channels whose filtered
    variance is a z-score outlier. Delegates to
    :func:`adaptive_reject.find_bads_channels_high_frequency`.

    Args:
        data: Pipeline data dict. Must contain the key specified by
            ``step_config['instance']`` (default ``'epochs'``).
        step_config: Step parameters:
            - ``instance`` (str): Data key to analyse. Default ``'epochs'``.
            - ``picks`` (list|None): Channel types to consider. Default all.
            - ``excluded_channels`` (list|None): Channels to skip.
            - ``zscore_thresh`` (float): Z-score threshold. Default 4.
            - ``max_iter`` (int): Maximum outlier-removal iterations.
              Default 2.
            - ``apply_on`` (list): Instances to mark bad channels on.
              Defaults to ``[instance]``.

    Returns:
        Updated data dict with bad channels added to ``info['bads']``.

    Raises:
        ValueError: If the requested instance or any ``apply_on`` key is absent.
    """
    # Check which instance to use
    instance = step_config.get('instance', 'epochs')
    if instance not in data:
        raise ValueError(f"find_bads_channels_high_frequency requires '{instance}' in data")

    picks_params = step_config.get('picks', None)
    excluded_channels = step_config.get('excluded_channels', None)
    zscore_thresh = step_config.get('zscore_thresh', 4)
    max_iter = step_config.get('max_iter', 2)
    apply_on = step_config.get('apply_on', [instance])

    if not isinstance(apply_on, list):
        apply_on = [apply_on]
    
    if any(inst not in data for inst in apply_on):
        raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

    picks = data.get_picks(data[instance].info, picks_params, excluded_channels)

    bad_chs, bad_chs_by_type = _detect_per_channel_type(
        adaptive_reject.find_bads_channels_high_frequency,
        data[instance], picks, zscore_thresh, max_iter
    )

    # Mark channels as bad
    if bad_chs:
        for instance_to_apply in apply_on:
            data[instance_to_apply].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_to_apply].info['bads']])

    data['preprocessing_steps'].append({
        'step': 'find_bads_channels_high_frequency',
        'bad_channels_by_type': bad_chs_by_type,
        'instance': instance,
        'picks': picks_params,
        'excluded_channels': excluded_channels,
        'apply_on': apply_on,
        'zscore_thresh': zscore_thresh,
        'max_iter': max_iter,
        'bad_channels': bad_chs,
        'n_bad_channels': len(bad_chs)
    })

    return data


@register("find_bads_epochs_threshold")
def find_bads_epochs_threshold(data, step_config):
    """Drop epochs in which too many channels exceed amplitude thresholds.

    Delegates to :func:`adaptive_reject.find_bads_epochs_threshold` and
    drops the identified bad epochs in-place.

    Args:
        data: Pipeline data dict. Must contain ``'epochs'``.
        step_config: Step parameters:
            - ``picks`` (list|None): Channel types to consider. Default all.
            - ``excluded_channels`` (list|None): Channels to skip.
            - ``reject`` (dict): Amplitude thresholds per channel type.
              Default: from the top-level
              ``datatype``, ``{'eeg': 100e-6}`` for EEG and
              ``{'mag': 4e-12, 'grad': 4e-10}`` for MEG.
            - ``n_channels_bad_epoch`` (float): Fraction of channels that
              must exceed the threshold for an epoch to be dropped.
              Default 0.1.

    Returns:
        Updated data dict with bad epochs dropped from ``data['epochs']``.

    Raises:
        ValueError: If ``'epochs'`` is not in ``data``.
    """
    if 'epochs' not in data:
        raise ValueError("find_bads_epochs_threshold requires 'epochs' in data")

    picks_params = step_config.get('picks', None)
    excluded_channels = step_config.get('excluded_channels', None)
    reject = step_config.get('reject', None)
    n_channels_bad_epoch = step_config.get('n_channels_bad_epoch', 0.1)

    picks = data.get_picks(data['epochs'].info, picks_params, excluded_channels)

    reject = _usable_reject(data, data['epochs'].info, picks, reject)
    bad_epochs = adaptive_reject.find_bads_epochs_threshold(
        data['epochs'], picks, reject, n_channels_bad_epoch
    ) if reject else np.array([], dtype=int)

    # Drop bad epochs
    if len(bad_epochs) > 0:
        data['epochs'].drop(bad_epochs, reason='ADAPTIVE AUTOREJECT')

    data['preprocessing_steps'].append({
        'step': 'find_bads_epochs_threshold',
        'picks': picks_params,
        'excluded_channels': excluded_channels,
        'apply_on': ['epochs'], # only for compatibility with others reject steps
        'reject': reject,
        'n_channels_bad_epoch': n_channels_bad_epoch,
        'bad_epochs': bad_epochs.tolist() if hasattr(bad_epochs, 'tolist') else list(bad_epochs),
        'n_bad_epochs': len(bad_epochs),
        'n_epochs_remaining': len(data['epochs'])
    })

    return data
