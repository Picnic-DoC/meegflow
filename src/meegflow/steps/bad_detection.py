import numpy as np
from .registry import register
from ..defaults import FLAT_VARIANCE, FLAT_VARIANCE_FALLBACK

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