import numpy as np
import mne
from .registry import register
from . import adaptive_reject


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
    threshold : float, optional
        Variance threshold below which channels are considered flat
        (default: 1e-12)
    
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
    threshold = step_config.get('threshold', 1e-12)
    
    # Get picks with exclusions
    picks = data.get_picks(data['raw'].info, picks_params, excluded_channels)

    # Get data only for selected picks
    raw_data = data['raw'].get_data(picks=picks)
    variances = raw_data.var(axis=1)
    flat_idx = np.where(variances < threshold)[0]
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
        'threshold': threshold,
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
              e.g. ``{'eeg': 100e-6}``. Default ``{'eeg': 100e-6}``.
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
    reject = step_config.get('reject', {'eeg': 100e-6})
    n_epochs_bad_ch = step_config.get('n_epochs_bad_ch', 0.5)
    apply_on = step_config.get('apply_on', ['epochs'])

    if not isinstance(apply_on, list):
        apply_on = [apply_on]

    if any(inst not in data for inst in apply_on):
        raise ValueError(f"find_bads_channels_threshold requires all instances of apply_on ({apply_on}) to be present in data")

    picks = data.get_picks(data['epochs'].info, picks_params, excluded_channels)

    bad_chs = adaptive_reject.find_bads_channels_threshold(
        data['epochs'], picks, reject, n_epochs_bad_ch
    )

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

    bad_chs = adaptive_reject.find_bads_channels_variance(
        data[instance], picks, zscore_thresh, max_iter
    )

    # Mark channels as bad
    if bad_chs:
        for instance_to_apply in apply_on:
            data[instance_to_apply].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_to_apply].info['bads']])

    data['preprocessing_steps'].append({
        'step': 'find_bads_channels_variance',
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

    bad_chs = adaptive_reject.find_bads_channels_high_frequency(
        data[instance], picks, zscore_thresh, max_iter
    )

    # Mark channels as bad
    if bad_chs:
        for instance_to_apply in apply_on:
            data[instance_to_apply].info['bads'].extend([ch for ch in bad_chs if ch not in data[instance_to_apply].info['bads']])

    data['preprocessing_steps'].append({
        'step': 'find_bads_channels_high_frequency',
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
              Default ``{'eeg': 100e-6}``.
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
    reject = step_config.get('reject', {'eeg': 100e-6})
    n_channels_bad_epoch = step_config.get('n_channels_bad_epoch', 0.1)

    picks = data.get_picks(data['epochs'].info, picks_params, excluded_channels)

    bad_epochs = adaptive_reject.find_bads_epochs_threshold(
        data['epochs'], picks, reject, n_channels_bad_epoch
    )

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
