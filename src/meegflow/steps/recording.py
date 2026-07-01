import importlib
import mne
from mne.utils import logger
from .registry import register
from ._helpers import find_events_from_raw


@register("strip_recording")
def strip_recording(data, step_config):
    """Crop recordings to the window spanned by their first and last events.

    Works on a single instance or a list of instances (e.g. ``'all_raw'``).
    Each recording is cropped in-place; optional padding is kept around the
    event boundaries.

    Args:
        data: Pipeline data dict. Must contain the key named by
            ``step_config['instance']``.
        step_config: Step parameters:
            - ``instance`` (str): Key in ``data`` to crop — ``'raw'`` or
              ``'all_raw'``. Default ``'raw'``.
            - ``get_events_from`` (str): ``'annotations'`` or ``'stim'``.
              Default ``'annotations'``.
            - ``shortest_event`` (int): Minimum event duration in samples.
              Default ``1``.
            - ``event_id`` (dict | ``'auto'``): Event IDs to consider when
              locating the first/last event. Default ``'auto'`` (all events).
            - ``start_padding`` (float): Seconds to keep before the first
              event. Default ``1``.
            - ``end_padding`` (float): Seconds to keep after the last event.
              Default ``1``.

    Returns:
        Updated data dict. Each recording is cropped in-place; a
        ``preprocessing_steps`` entry is appended for every cropped
        recording.

    Raises:
        ValueError: If ``instance`` is not present in ``data``.
    """
    instance = step_config.get('instance', 'raw')
    start_padding = step_config.get('start_padding', 1)
    end_padding = step_config.get('end_padding', 1)
    get_events_from = step_config.get('get_events_from', 'annotations')
    shortest_event = step_config.get('shortest_event', 1)
    event_id = step_config.get('event_id', 'auto')
    

    if instance not in data:
        raise ValueError(f"strip recordings step requires '{instance}' to be present in data (either 'all_raw', 'raw')")

    # TODO: improve this and make it general to all corresponding steps        
    all_instances = data[instance]
    if not isinstance(all_instances, list):
        all_instances = [all_instances]

    for i, inst in enumerate(all_instances):
        events, _ = find_events_from_raw(
            inst,
            get_events_from=get_events_from,
            shortest_event=shortest_event,
            event_id=event_id
        )
        
        # events[:, 0] are absolute sample numbers (they include
        # ``first_samp``); convert to seconds relative to the start of the
        # recording, where ``inst.times[0] == 0``.
        sfreq = inst.info['sfreq']
        start = (events[0, 0] - inst.first_samp) / sfreq - start_padding
        end = (events[-1, 0] - inst.first_samp) / sfreq + end_padding

        start = max(0, start)
        end = min(inst.times[-1], end)
        
        inst.crop(start, end)
        
        data['preprocessing_steps'].append({
            'step': 'strip_recording',
            'instance': f'{instance}-{i}',
            'start': start,
            'end': end
        })
    
    return data


@register("concatenate_recordings")
def concatenate_recordings(data, step_config):
    if 'all_raw' not in data:
        raise ValueError("concatenate_recordings requires 'all_raw' in data")

    if len(data['all_raw']) > 1:
        data['raw'] = mne.concatenate_raws(data['all_raw'])
    else:
        data['raw'] = data['all_raw'][0]

    data['preprocessing_steps'].append({
        'step': 'concatenate_recordings',
    })
    
    return data


@register("copy_instance")
def copy_instance(data, step_config):
    from_instance = step_config.get('from_instance', 'raw')
    to_instance = step_config.get('to_instance', 'raw_cleaned')

    if from_instance not in data:
        raise ValueError(f"copy_instance step requires '{from_instance}' to be in data")

    data[to_instance] = data[from_instance].copy()
    data['preprocessing_steps'].append({
        'step': 'copy_instance',
        'from_instance': from_instance,
        'to_instance': to_instance
    })

    return data


@register("call_module")
def call_module(data, step_config):
    """Dynamically call any importable function or object method and store the result in data.

    Useful for using MNE or third-party functions directly from the pipeline
    config without writing a custom step. Supports both module-level functions
    and methods on objects already present in the data dict.

    Any string value (in ``args`` or keyword arguments) that starts with
    ``data__`` is resolved as a path into the data dict using ``__`` as the
    key separator (e.g. ``'data__raw'`` → ``data['raw']``,
    ``'data__house__dog'`` → ``data['house']['dog']``).

    Args:
        data: Pipeline data dict.
        step_config: Step parameters:
            - ``module`` (str): Fully-qualified callable
              (e.g. ``'mne.channels.make_standard_montage'``) when calling
              a module-level function, or just the method name
              (e.g. ``'set_montage'``) when ``target`` is provided.
            - ``target`` (str, optional): ``data__``-prefixed reference to
              an object already in ``data``. When present, ``module`` is
              treated as a method name on that object rather than an
              importable path.
            - ``var_name`` (str | None, optional): Key under which to store
              the return value in ``data``. Set to ``null`` to discard the
              result (useful for in-place methods). Mutually exclusive with
              ``unpack_as``.
            - ``unpack_as`` (list of str, optional): Unpack a multi-value
              return into separate ``data`` keys in order. Mutually
              exclusive with ``var_name``.
            - ``args`` (list, optional): Positional arguments forwarded to
              the callable in order. Each element supports ``data__``
              resolution.
            - Any remaining key is forwarded as a keyword argument.
              Values support ``data__`` resolution.

    Returns:
        Updated data dict.

    Raises:
        ValueError: If ``module`` is missing, ``var_name`` and
            ``unpack_as`` are both set, or a ``data__`` path cannot be
            resolved.

    Example config::

        # Module-level function with keyword args
        - name: call_module
          module: mne.channels.make_standard_montage
          var_name: montage
          kind: standard_1020

        # Method call on a data object (target)
        - name: call_module
          target: "data__raw"
          module: set_montage
          var_name: null
          montage: "data__montage"
          on_missing: ignore

        # Positional-only function via args list
        - name: call_module
          module: os.path.join
          var_name: out_path
          args:
            - "/derivatives"
            - "data__subject"

        # Unpack a multi-value return into separate data keys
        - name: call_module
          module: mne.events_from_annotations
          unpack_as: [events, event_id]
          args:
            - "data__raw"
    """
    module_str = step_config.get('module')
    var_name = step_config.get('var_name')
    target_ref = step_config.get('target')
    unpack_as = step_config.get('unpack_as')

    if not module_str:
        raise ValueError("call_module step requires 'module' in step_config")

    if var_name is not None and unpack_as is not None:
        raise ValueError("call_module: 'var_name' and 'unpack_as' are mutually exclusive")

    def _resolve(value):
        if isinstance(value, str) and value.startswith('data__'):
            path = value[6:]
            try:
                obj = data
                for k in path.split('__'):
                    obj = obj[k]
                return obj
            except KeyError as e:
                raise ValueError(
                    f"call_module: could not resolve '{value}' in data: key {e} not found"
                ) from e
        return value

    if target_ref is not None:
        func = getattr(_resolve(target_ref), module_str)
    else:
        mod_path, func_name = module_str.rsplit('.', 1)
        func = getattr(importlib.import_module(mod_path), func_name)

    reserved = {'module', 'var_name', 'args', 'target', 'unpack_as'}
    args = [_resolve(v) for v in step_config.get('args', [])]
    kwargs = {key: _resolve(value) for key, value in step_config.items() if key not in reserved}

    result = func(*args, **kwargs)

    if unpack_as is not None:
        for key, val in zip(unpack_as, result):
            data[key] = val
    elif var_name is not None:
        data[var_name] = result

    data['preprocessing_steps'].append({
        'step': 'call_module',
        'module': module_str,
        'target': target_ref,
        'var_name': var_name,
        'unpack_as': unpack_as,
        'args': step_config.get('args', []),
        'kwargs': {k: v for k, v in step_config.items() if k not in reserved},
    })

    return data
