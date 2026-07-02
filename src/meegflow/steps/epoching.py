import mne
from .registry import register
from ._helpers import find_events_from_raw


@register("find_events")
def find_events(data, step_config):
    """Extract events from raw data and store them in the pipeline dict.

    Args:
        data: Pipeline data dict. Must contain ``'raw'``.
        step_config: Step parameters:
            - ``get_events_from`` (str): ``'annotations'`` or ``'stim'``.
              Default ``'annotations'``.
            - ``shortest_event`` (int): Minimum event duration in samples.
              Default ``1``.
            - ``event_id`` (dict | ``'auto'``): Mapping of event names to
              integer IDs, or ``'auto'`` to infer from the data.
              Default ``'auto'``.
            - ``stim_channel`` (str | None): Stimulus channel name used
              when ``get_events_from='stim'``. Default ``None`` (MNE
              auto-detects).

    Returns:
        Updated data dict with ``data['events']`` (ndarray, shape
        ``(n_events, 3)``), ``data['event_id']`` (dict), and
        ``data['events_sfreq']`` (float) set.

    Raises:
        ValueError: If ``'raw'`` is not in ``data``.
    """
    if 'raw' not in data:
        raise ValueError("find_events requires 'raw' in data")

    get_events_from = step_config.get('get_events_from', 'annotations')
    shortest_event = step_config.get('shortest_event', 1)
    event_id = step_config.get('event_id', 'auto')
    stim_channel = step_config.get('stim_channel', None)
    
    data['events'], found_event_id = find_events_from_raw(
        data['raw'],
        get_events_from=get_events_from,
        shortest_event=shortest_event,
        event_id=event_id,
        stim_channel=stim_channel
    )
    data['event_id'] = found_event_id
    data['events_sfreq'] = data['raw'].info['sfreq']

    data['preprocessing_steps'].append({
        'step': 'find_events',
        'found_event_id': found_event_id,
        'found_events': data['events'].tolist(),
        'n_events': data['events'].shape[0]
    })

    return data


@register("epoch")
def epoch(data, step_config):
    """Create event-locked epochs from raw data.

    Args:
        data: Pipeline data dict. Must contain ``'raw'`` and ``'events'``.
        step_config: Step parameters:
            - ``event_id`` (dict|str): Event IDs to epoch around. Falls
              back to ``data['event_id']`` if not provided.
            - ``tmin`` (float): Epoch start time relative to event.
              Default -0.2 s.
            - ``tmax`` (float): Epoch end time relative to event.
              Default 0.5 s.
            - ``baseline`` (tuple): Baseline correction window.
              Default ``(None, 0.0)``.
            - ``reject`` (dict|None): Peak-to-peak rejection thresholds.
              Default ``None``.

    Returns:
        Updated data dict with ``data['epochs']`` set to the new
        ``mne.Epochs`` object.

    Raises:
        ValueError: If ``'raw'`` or ``'events'`` are not in ``data``.
    """
    if data.get('raw', None) is None or data.get('events', None) is None:
        raise ValueError("epoch step requires both 'raw' and 'events' in data")

    event_id = step_config.get('event_id', None) or data.get('event_id', 'NOT FOUND')
    tmin = step_config.get('tmin', -0.2)
    tmax = step_config.get('tmax', 0.5)
    baseline = step_config.get('baseline', (None, 0.0))
    reject = step_config.get('reject', None)

    data['epochs'] = mne.Epochs(
        data['raw'],
        events=data['events'],
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True
    )

    data['preprocessing_steps'].append({
        'step': 'epoch',
        'event_id': event_id,
        'tmin': tmin,
        'tmax': tmax,
        'baseline': baseline,
        'reject': reject,
        'n_epochs': len(data['epochs'])
    })

    return data


@register("chunk_in_epoch")
def chunk_in_epoch(data, step_config):
    """Segment raw data into fixed-length epochs (no events required).

    Args:
        data: Pipeline data dict. Must contain ``'raw'``.
        step_config: Step parameters:
            - ``duration`` (float): Length of each epoch in seconds.
              Default 1.

    Returns:
        Updated data dict with ``data['epochs']`` set to the new
        fixed-length ``mne.Epochs`` object.

    Raises:
        ValueError: If ``'raw'`` is not in ``data``.
    """
    if data.get('raw', None) is None:
        raise ValueError("epoch step requires 'raw' in data")

    duration = step_config.get('duration', 1)

    data['epochs'] = mne.make_fixed_length_epochs(data['raw'], duration=duration, preload=True)

    data['preprocessing_steps'].append({
        'step': 'chunk_in_epoch',
        'type': 'fixed_length_epochs',
        'duration': duration,
    })

    return data
