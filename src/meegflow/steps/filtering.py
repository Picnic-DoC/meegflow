import mne
from mne.utils import logger
from .registry import register


@register("bandpass_filter")
def bandpass_filter(data, step_config):
    """
    Apply bandpass filtering.
    
    Applies both high-pass and low-pass filters using IIR Butterworth filters.
    
    Parameters (via step_config)
    -----------------------------
    l_freq : float, optional
        High-pass filter frequency in Hz (default: 0.5)
    h_freq : float, optional
        Low-pass filter frequency in Hz (default: 45.0)
    l_freq_order : int, optional
        Filter order for high-pass filter (default: 6)
    h_freq_order : int, optional
        Filter order for low-pass filter (default: 8)
    picks : list, optional
        Channel types to filter (e.g., ['eeg']). If None, defaults to MEEG channels.
    excluded_channels : list of str, optional
        Channel names to exclude from filtering (e.g., reference channels)
    n_jobs : int, optional
        Number of parallel jobs (default: 1)
    
    Updates
    -------
    data['raw'] : mne.io.Raw
        Filters applied in-place
    data['preprocessing_steps'] : list
        Appends step information for both high-pass and low-pass filters
    
    Returns
    -------
    data : dict
        Updated data dictionary with filtered raw data
    """
    if 'raw' not in data:
        raise ValueError("bandpass_filter requires 'raw' in data")

    picks_params = step_config.get('picks', None)
    excluded_channels = step_config.get('excluded_channels', None)
    l_freq = step_config.get('l_freq', 0.5)
    l_freq_order = step_config.get('l_freq_order', 6)
    h_freq = step_config.get('h_freq', 45.0)
    h_freq_order = step_config.get('h_freq_order', 8)
    n_jobs = step_config.get('n_jobs', 1)

    # Compute picks if provided, otherwise None (all channels)
    picks = data.get_picks(data['raw'].info, picks_params, excluded_channels)

    # Apply filtering in 2 steps: high-pass and low-pass
    high_pass_filter_params = dict(
        method='iir',
        l_trans_bandwidth=0.1,
        iir_params=dict(ftype='butter', order=l_freq_order),
        l_freq=l_freq,
        h_freq=None,
        n_jobs=n_jobs
    )
    data['raw'].filter(
        picks=picks,
        **high_pass_filter_params
    )

    low_pass_filter_params = dict(
        method='iir',
        h_trans_bandwidth=0.1,
        iir_params=dict(ftype='butter', order=h_freq_order),
        l_freq=None,
        h_freq=h_freq,
        n_jobs=n_jobs
    )
    data['raw'].filter(
        picks=picks,
        **low_pass_filter_params
    )

    # Store info for reporting
    data['preprocessing_steps'].extend([
        {
            'step': 'high_pass_filter',
            'picks': picks_params,
            'excluded_channels': excluded_channels,
            'params': high_pass_filter_params
        },
        {
            'step': 'low_pass_filter',
            'picks': picks_params,
            'excluded_channels': excluded_channels,
            'params': low_pass_filter_params
        }
    ])

    return data


@register("notch_filter")
def notch_filter(data, step_config):
    """
    Apply notch filtering to remove line noise.
    
    Removes power line interference at specified frequencies (e.g., 50 Hz or 60 Hz
    and their harmonics).
    
    Parameters (via step_config)
    -----------------------------
    freqs : list of float
        Frequencies to notch filter in Hz (e.g., [50.0, 100.0])
    notch_widths : float or list, optional
        Width of notch filters. If None, uses MNE default.
    method : str, optional
        Filtering method (default: 'fft')
    picks : list, optional
        Channel types to filter. If None, defaults to MEEG channels.
    excluded_channels : list of str, optional
        Channel names to exclude from filtering
    n_jobs : int, optional
        Number of parallel jobs (default: 1)
    
    Updates
    -------
    data['raw'] : mne.io.Raw
        Notch filters applied in-place
    data['preprocessing_steps'] : list
        Appends step information
    
    Returns
    -------
    data : dict
        Updated data dictionary with notch-filtered raw data
    """
    if 'raw' not in data:
        raise ValueError("notch_filter requires 'raw' in data")

    picks_params = step_config.get('picks', None)
    excluded_channels = step_config.get('excluded_channels', None)
    freqs = step_config.get('freqs', [50.0, 100.0])
    notch_widths = step_config.get('notch_widths', None)
    method = step_config.get('method', 'fft')
    n_jobs = step_config.get('n_jobs', 1)

    # Compute picks if provided
    picks = data.get_picks(data['raw'].info, picks_params, excluded_channels)

    data['raw'].notch_filter(
        freqs=freqs,
        method=method,
        notch_widths=notch_widths,
        picks=picks,
        n_jobs=n_jobs
    )

    # Store info for reporting
    data['preprocessing_steps'].append({
        'step': 'notch_filter',
        'picks': picks_params,
        'excluded_channels': excluded_channels,
        'freqs': freqs,
        'method': method,
        'notch_widths': notch_widths
    })

    return data
