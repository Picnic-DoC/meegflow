import mne
from mne.utils import logger
from .registry import register


@register("set_montage")
def set_montage(data, step_config):
    """
    Set channel montage for MEEG data.
    
    Useful when data lacks electrode position information. Sets standard electrode
    positions based on the specified montage.
    
    Parameters (via step_config)
    -----------------------------
    montage : str, optional
        Name of standard montage to use (default: 'standard_1020').
        Examples: 'standard_1020', 'standard_1005', 'biosemi64', etc.
        See MNE documentation for available montages.
    
    Updates
    -------
    data['raw'] : mne.io.Raw
        Electrode positions are set based on the montage
    data['preprocessing_steps'] : list
        Appends step information
    
    Returns
    -------
    data : dict
        Updated data dictionary with montage set
    """
    if 'raw' not in data:
        raise ValueError("set_montage requires 'raw' in data")

    montage_name = step_config.get('montage', 'standard_1020')

    montage = mne.channels.make_standard_montage(montage_name)
    data['raw'].set_montage(montage, on_missing="ignore")

    data['preprocessing_steps'].append({
        'step': 'set_montage',
        'montage': montage_name
    })
    return data


@register("drop_unused_channels")
def drop_unused_channels(data, step_config):
    """
    Drop unused channels from the data.
    
    This step explicitly removes specified channels from the dataset.
    Different from drop_bad_channels, this step drops channels by name
    regardless of whether they are marked as bad.
    
    Parameters (via step_config)
    -----------------------------
    channels_to_drop : list of str
        List of channel names to drop from the data
    instance : str, optional
        Which data instance to drop channels from - 'raw' or 'epochs' (default: 'raw')
    
    Updates
    -------
    data[instance] : mne.io.Raw or mne.Epochs
        Specified channels are removed from the data
    data['preprocessing_steps'] : list
        Appends step information including list of dropped channels
    
    Returns
    -------
    data : dict
        Updated data dictionary with specified channels removed
    """
    channels_to_drop = step_config.get('channels_to_drop', [])
    instance = step_config.get('instance', 'raw') 

    if instance not in data:
        raise ValueError(f"drop_unused_channels step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

    data[instance].drop_channels(channels_to_drop)

    data['preprocessing_steps'].append({
        'step': 'drop_unused_channels',
        'instance': instance,
        'channels_dropped': channels_to_drop
    })

    return data


@register("interpolate_bad_channels")
def interpolate_bad_channels(data, step_config):
    """
    Interpolate bad channels using spherical spline interpolation.
    
    Repairs bad channels by interpolating their values from neighboring channels.
    After interpolation, the channels are removed from the info['bads'] list.
    
    Parameters (via step_config)
    -----------------------------
    instance : str, optional
        Which data instance to interpolate - 'raw' or 'epochs' (default: 'epochs')
    excluded_channels : list of str, optional
        Channel names to exclude from interpolation even if marked as bad.
        These channels will remain in info['bads'] after interpolation.
    
    Updates
    -------
    data[instance] : mne.io.Raw or mne.Epochs
        Bad channels (except excluded ones) are interpolated and removed from info['bads']
    data['preprocessing_steps'] : list
        Appends step information
    
    Returns
    -------
    data : dict
        Updated data dictionary with bad channels interpolated
    """
    instance = step_config.get('instance', 'epochs')
    excluded_channels = step_config.get('excluded_channels', [])

    if instance not in data:
        raise ValueError(f"interpolate_bad_channels step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

    data[instance].interpolate_bads(
        reset_bads=True,
        exclude=excluded_channels
    )

    data['preprocessing_steps'].append({
        'step': 'interpolate_bad_channels',
        'excluded_channels': excluded_channels,
        'instance': instance
    })

    return data


@register("drop_bad_channels")
def drop_bad_channels(data, step_config):
    """
    Drop bad channels without interpolation.
    
    This step removes channels marked as bad from the data instead of interpolating them.
    Useful when you want to permanently remove problematic channels from the dataset.
    
    Parameters (via step_config)
    -----------------------------
    instance : str, optional
        Which data instance to drop channels from - 'raw' or 'epochs' (default: 'epochs')
    excluded_channels : list of str, optional
        List of channel names to exclude from dropping even if marked as bad.
        These channels will remain in the data even if they are in info['bads'].
    
    Updates
    -------
    data[instance] : mne.io.Raw or mne.Epochs
        Channels marked as bad (except excluded ones) are removed from the data
    data['preprocessing_steps'] : list
        Appends step information including list of dropped channels
    
    Returns
    -------
    data : dict
        Updated data dictionary with bad channels removed
    """
    instance = step_config.get('instance', 'epochs')
    excluded_channels = step_config.get('excluded_channels', None)

    if instance not in data:
        raise ValueError(f"drop_bad_channels step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

    # Get the list of bad channels before dropping
    bad_channels = list(data[instance].info['bads'])
    
    # Filter out excluded channels if specified
    if excluded_channels:
        channels_to_drop = [ch for ch in bad_channels if ch not in excluded_channels]
        excluded_bads = [ch for ch in bad_channels if ch in excluded_channels]
        if excluded_bads:
            logger.info(f"Excluding {len(excluded_bads)} bad channels from dropping: {excluded_bads}")
    else:
        channels_to_drop = bad_channels
    
    if channels_to_drop:
        # Drop the bad channels
        data[instance].drop_channels(channels_to_drop)
        logger.info(f"Dropped {len(channels_to_drop)} bad channels: {channels_to_drop}")
    else:
        logger.info("No bad channels to drop")

    data['preprocessing_steps'].append({
        'step': 'drop_bad_channels',
        'instance': instance,
        'excluded_channels': excluded_channels,
        'dropped_channels': channels_to_drop,
        'n_bad_channels': len(channels_to_drop)
    })

    return data
