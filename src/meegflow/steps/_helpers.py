import mne


def find_events_from_raw(raw, get_events_from='annotations', shortest_event=1, event_id='auto', stim_channel=None):
    """Extract events from a Raw object using annotations or a stim channel.

    Args:
        raw: MNE Raw object to extract events from.
        get_events_from: Source of events — ``'annotations'`` (default) or
            ``'stim_channel'``.
        shortest_event: Minimum number of samples for an event. Used only
            when ``get_events_from='stim_channel'``. Default 1.
        event_id: Event ID mapping passed to
            ``mne.events_from_annotations``. Default ``'auto'``.
        stim_channel: Stimulus channel name. Required when
            ``get_events_from='stim_channel'``.

    Returns:
        Tuple of ``(events, event_id_map)`` where ``events`` is a NumPy
        array of shape ``(n_events, 3)`` and ``event_id_map`` is a dict
        mapping event names to integer codes (``None`` for stim-channel
        mode).

    Raises:
        ValueError: If ``get_events_from`` is not a recognised method.
    """
    if get_events_from == 'stim_channel':
        events = mne.find_events(
            raw,
            shortest_event=shortest_event,
            stim_channel=stim_channel,
            verbose=False
        )
        return events, None
    
    if get_events_from == 'annotations':
        events, found_event_id = mne.events_from_annotations(raw, event_id=event_id)
        return events, found_event_id

    raise ValueError(f"Invalid get_events_from method: {get_events_from}")
