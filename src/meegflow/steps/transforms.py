import mne
import numpy as np
from .registry import register


@register("resample")
def resample(data, step_config):
    """Resample a Raw or Epochs instance to a new sampling frequency.

    Args:
        data: Pipeline data dict. Must contain the key named by
            ``step_config['instance']`` (default ``'raw'``).
        step_config: Step parameters:
            - ``instance`` (str): Key in ``data`` to resample. Default ``'raw'``.
            - ``sfreq`` (float): Target sampling frequency. Default 250.
            - ``npad`` (str|int): Padding mode passed to MNE. Default ``'auto'``.
            - ``n_jobs`` (int): Parallel jobs. Default 1.
            - ``resample_events`` (bool): Also resample the events array.
              Default ``False``.

    Returns:
        Updated data dict with the instance resampled in-place.

    Raises:
        ValueError: If the requested instance is not present in ``data``.
    """
    instance = step_config.get('instance', 'raw')
    
    if instance not in data:
        raise ValueError(f"resample requires '{instance}' in data")

    resample_events = step_config.get('resample_events', False)
    sfreq = step_config.get('sfreq', 250)
    npad = step_config.get('npad', 'auto')
    n_jobs = step_config.get('n_jobs', 1)

    data[instance].resample(
        sfreq=sfreq,
        npad=npad,
        n_jobs=n_jobs
    )

    if resample_events and data.get('events') is not None:
        old_sfreq = data.get('events_sfreq')
        if old_sfreq:
            events = data['events'].copy()
            # Scale the sample column to the new sampling frequency and
            # store the result (the previous call discarded its return).
            events[:, 0] = np.round(
                events[:, 0] * (sfreq / old_sfreq)
            ).astype(events.dtype)
            data['events'] = events
            data['events_sfreq'] = sfreq

    # Store info for reporting
    data['preprocessing_steps'].append({
        'step': 'resample',
        'instance': instance,
        'resample_events': resample_events,
        'sfreq': sfreq,
        'npad': npad,
    })

    return data


@register("reference")
def reference(data, step_config):
    """Apply EEG re-referencing to a Raw or Epochs instance.

    Args:
        data: Pipeline data dict. Must contain the key named by
            ``step_config['instance']`` (default ``'epochs'``).
        step_config: Step parameters:
            - ``instance`` (str): Key in ``data`` to re-reference.
              Default ``'epochs'``.
            - ``ref_channels`` (str|list): Reference channel(s) passed to
              ``mne.set_eeg_reference``. Default ``'average'``.

    Returns:
        Updated data dict with the instance re-referenced in-place.

    Raises:
        ValueError: If the requested instance is not present in ``data``.
    """
    ref_channels = step_config.get('ref_channels', 'average')
    instance = step_config.get('instance', 'epochs')

    if instance not in data:
        raise ValueError(f"reference step requires '{instance}' to be present in data (either 'raw' or 'epochs')")

    mne.set_eeg_reference(
        inst=data[instance],
        ref_channels=ref_channels,
    )

    data['preprocessing_steps'].append({
        'step': 'reference',
        'ref_channels': ref_channels
    })

    return data
