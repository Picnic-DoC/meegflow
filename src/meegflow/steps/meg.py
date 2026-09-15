"""MEG-specific preprocessing steps.

Thin wrappers around MNE-Python's standard implementations, so that the usual
MEG operations can be written in a configuration without ``call_module``:

- ``maxwell_filter``: signal-space separation (SSS), temporal SSS (tSSS) and
  movement compensation (:func:`mne.preprocessing.maxwell_filter`);
- ``find_bads_maxwell``: noisy and flat MEG channel detection
  (:func:`mne.preprocessing.find_bad_channels_maxwell`);
- ``compute_head_pos``: continuous head position from cHPI coils
  (:mod:`mne.chpi`), for movement compensation in ``maxwell_filter``;
- ``compute_ssp``: signal-space projectors for cardiac or ocular artifacts
  (:func:`mne.preprocessing.compute_proj_ecg` / ``compute_proj_eog``);
- ``apply_gradient_compensation``: CTF synthetic-gradiometer compensation
  (:meth:`mne.io.Raw.apply_gradient_compensation`).

Every MNE keyword argument can be given in the step configuration and is
passed through unchanged, so the defaults are MNE's own.
"""
from pathlib import Path

import mne
import numpy as np
from mne.utils import logger
from mne_bids import BIDSPath

from .registry import register


def _require_instance(data, instance, step):
    if instance not in data:
        raise ValueError(f"{step} requires '{instance}' in data")
    return data[instance]


def _require_meg(inst, step):
    if len(mne.pick_types(inst.info, meg=True, ref_meg=False, exclude=[])) == 0:
        raise ValueError(f"{step} requires MEG channels, but the instance has none")


def _passthrough(step_config, reserved):
    """Keyword arguments for the MNE function: every key not used by the step."""
    return {k: v for k, v in step_config.items() if k not in reserved}


def _resolve_head_pos(data, head_pos):
    """Head positions from a context entry, a ``.pos`` file, or None."""
    if head_pos is None:
        return None
    if isinstance(head_pos, str):
        if head_pos in data:
            return data[head_pos]
        if Path(head_pos).exists():
            return mne.chpi.read_head_pos(head_pos)
        raise ValueError(
            f"head_pos '{head_pos}' is neither an entry in the context nor an existing file"
        )
    return np.asarray(head_pos)


def _mark_bads(data, instances, bad_chs):
    for name in instances:
        bads = data[name].info['bads']
        bads.extend(ch for ch in bad_chs if ch not in bads)


@register("maxwell_filter")
def maxwell_filter(data, step_config):
    """Apply Maxwell filtering (SSS or tSSS), optionally with movement compensation.

    Delegates to :func:`mne.preprocessing.maxwell_filter` and replaces the
    instance with the filtered recording. Mark bad MEG channels before this
    step (for example with ``find_bads_maxwell``): SSS reconstructs them.

    Args:
        data: Pipeline context. Must contain the key named by ``instance``.
        step_config: Step parameters:
            - ``instance`` (str): Continuous recording to filter. Default ``'raw'``.
            - ``head_pos`` (str|None): Head positions for movement compensation,
              either the name of a context entry (as written by
              ``compute_head_pos``) or the path to a ``.pos`` file. Default None.
            - Any keyword argument of :func:`mne.preprocessing.maxwell_filter`,
              e.g. ``calibration`` and ``cross_talk`` (paths to the
              fine-calibration and cross-talk files), ``st_duration`` (seconds;
              enables tSSS), ``st_correlation``, ``origin``, ``int_order``,
              ``ext_order``, ``coord_frame``, ``destination``. MNE's defaults
              apply to anything omitted.

    Returns:
        Updated context with the Maxwell-filtered instance.
    """
    instance = step_config.get('instance', 'raw')
    head_pos_ref = step_config.get('head_pos', None)
    inst = _require_instance(data, instance, 'maxwell_filter')
    _require_meg(inst, 'maxwell_filter')

    kwargs = _passthrough(step_config, ('instance', 'head_pos'))
    head_pos = _resolve_head_pos(data, head_pos_ref)
    data[instance] = mne.preprocessing.maxwell_filter(inst, head_pos=head_pos, **kwargs)

    data['preprocessing_steps'].append({
        'step': 'maxwell_filter',
        'instance': instance,
        'head_pos': head_pos_ref,
        'movement_compensation': head_pos is not None,
        'params': kwargs,
    })
    return data


@register("find_bads_maxwell")
def find_bads_maxwell(data, step_config):
    """Detect noisy and flat MEG channels with Maxwell filtering.

    Delegates to :func:`mne.preprocessing.find_bad_channels_maxwell` and marks
    the detected channels as bad. Run it on data that has not been Maxwell
    filtered yet.

    Args:
        data: Pipeline context. Must contain the key named by ``instance``.
        step_config: Step parameters:
            - ``instance`` (str): Continuous recording to analyse. Default ``'raw'``.
            - ``apply_on`` (list): Instances to mark the bad channels on.
              Defaults to ``[instance]``.
            - ``head_pos`` (str|None): As in ``maxwell_filter``. Default None.
            - Any keyword argument of
              :func:`mne.preprocessing.find_bad_channels_maxwell`, e.g.
              ``calibration``, ``cross_talk``, ``limit``, ``duration``,
              ``min_count``, ``h_freq``, ``origin``, ``coord_frame``.

    Returns:
        Updated context with the detected channels added to ``info['bads']``.
    """
    instance = step_config.get('instance', 'raw')
    head_pos_ref = step_config.get('head_pos', None)
    apply_on = step_config.get('apply_on', [instance])
    if not isinstance(apply_on, list):
        apply_on = [apply_on]
    inst = _require_instance(data, instance, 'find_bads_maxwell')
    _require_meg(inst, 'find_bads_maxwell')
    if any(name not in data for name in apply_on):
        raise ValueError(f"find_bads_maxwell requires all instances of apply_on ({apply_on}) to be present in data")

    kwargs = _passthrough(step_config, ('instance', 'head_pos', 'apply_on', 'return_scores'))
    noisy, flat = mne.preprocessing.find_bad_channels_maxwell(
        inst, head_pos=_resolve_head_pos(data, head_pos_ref), **kwargs
    )[:2]
    bad_chs = list(noisy) + [ch for ch in flat if ch not in noisy]
    _mark_bads(data, apply_on, bad_chs)

    data['preprocessing_steps'].append({
        'step': 'find_bads_maxwell',
        'instance': instance,
        'apply_on': apply_on,
        'head_pos': head_pos_ref,
        'params': kwargs,
        'noisy_channels': list(noisy),
        'flat_channels': list(flat),
        'bad_channels': bad_chs,
        'n_bad_channels': len(bad_chs),
    })
    return data


@register("compute_head_pos")
def compute_head_pos(data, step_config):
    """Estimate the continuous head position from cHPI coils.

    Runs :func:`mne.chpi.compute_chpi_amplitudes`,
    :func:`mne.chpi.compute_chpi_locs` and :func:`mne.chpi.compute_head_pos`,
    or reads an existing ``.pos`` file, and stores the positions in the
    context so that ``maxwell_filter`` can compensate head movements
    (``head_pos: head_pos``).

    Args:
        data: Pipeline context. Must contain the key named by ``instance``.
        step_config: Step parameters:
            - ``instance`` (str): Continuous recording with cHPI. Default ``'raw'``.
            - ``pos_file`` (str|None): Read positions from this ``.pos`` file
              instead of computing them. Default None.
            - ``var_name`` (str): Context entry the positions are stored in.
              Default ``'head_pos'``.
            - ``t_step_min`` (float), ``t_window`` (float|'auto'),
              ``ext_order`` (int), ``tmin`` (float), ``tmax`` (float|None):
              passed to ``compute_chpi_amplitudes``. Defaults ``0.01``,
              ``'auto'``, ``1``, ``0``, ``None``.
            - ``t_step_max`` (float), ``too_close`` (str): passed to
              ``compute_chpi_locs``. Defaults ``1.0``, ``'raise'``.
            - ``dist_limit`` (float), ``gof_limit`` (float): passed to
              ``compute_head_pos``. Defaults ``0.005``, ``0.98``.
            - ``save`` (bool): Also write the positions as a ``.pos`` file
              under the derivatives root. Default False.

    Returns:
        Updated context with ``data[var_name]``, an ``(n_positions, 10)``
        array in MNE's head-position format.
    """
    instance = step_config.get('instance', 'raw')
    pos_file = step_config.get('pos_file', None)
    var_name = step_config.get('var_name', 'head_pos')
    save = step_config.get('save', False)
    inst = _require_instance(data, instance, 'compute_head_pos')

    if pos_file is not None:
        head_pos = mne.chpi.read_head_pos(pos_file)
    else:
        _require_meg(inst, 'compute_head_pos')
        amplitudes = mne.chpi.compute_chpi_amplitudes(
            inst,
            t_step_min=step_config.get('t_step_min', 0.01),
            t_window=step_config.get('t_window', 'auto'),
            ext_order=step_config.get('ext_order', 1),
            tmin=step_config.get('tmin', 0),
            tmax=step_config.get('tmax', None),
        )
        locs = mne.chpi.compute_chpi_locs(
            inst.info, amplitudes,
            t_step_max=step_config.get('t_step_max', 1.0),
            too_close=step_config.get('too_close', 'raise'),
        )
        head_pos = mne.chpi.compute_head_pos(
            inst.info, locs,
            dist_limit=step_config.get('dist_limit', 0.005),
            gof_limit=step_config.get('gof_limit', 0.98),
        )

    data[var_name] = head_pos

    if save:
        bids_path = BIDSPath(
            subject=data['subject'], task=data['task'],
            session=data.get('session', None), acquisition=data.get('acquisition', None),
            datatype='meg', root=data.derivatives_root('head_pos'),
            suffix='headpos', extension='.pos', check=False,
        )
        bids_path.mkdir(exist_ok=True)
        mne.chpi.write_head_pos(bids_path.fpath, head_pos)
        data[f'{var_name}_file'] = str(bids_path)

    # Largest displacement of the head origin from its first position, in mm.
    translations = head_pos[:, 4:7] if len(head_pos) else np.zeros((0, 3))
    max_shift = float(np.max(np.linalg.norm(translations - translations[0], axis=1)) * 1e3) \
        if len(translations) else 0.0

    data['preprocessing_steps'].append({
        'step': 'compute_head_pos',
        'instance': instance,
        'pos_file': pos_file,
        'var_name': var_name,
        'n_positions': int(len(head_pos)),
        'max_displacement_mm': max_shift,
        'saved_to': data.get(f'{var_name}_file', None),
    })
    logger.info(f"compute_head_pos: {len(head_pos)} positions, max displacement {max_shift:.1f} mm")
    return data


@register("compute_ssp")
def compute_ssp(data, step_config):
    """Compute signal-space projectors (SSP) for cardiac or ocular artifacts.

    Delegates to :func:`mne.preprocessing.compute_proj_ecg` or
    :func:`mne.preprocessing.compute_proj_eog` and adds the projectors to the
    instance. Projectors added this way are applied by ``epoch`` (MNE's
    ``Epochs`` apply them by default), or immediately with ``apply: true``.
    Without an ECG channel, MNE builds a synthetic ECG from the magnetometers.

    Args:
        data: Pipeline context. Must contain the key named by ``instance``.
        step_config: Step parameters:
            - ``artifact`` (str): ``'ecg'`` or ``'eog'``. Required.
            - ``instance`` (str): Continuous recording. Default ``'raw'``.
            - ``apply`` (bool): Apply the projectors to the data now.
              Default False.
            - Any keyword argument of ``compute_proj_ecg`` / ``compute_proj_eog``,
              e.g. ``n_grad``, ``n_mag``, ``n_eeg``, ``ch_name``, ``average``,
              ``reject``, ``tmin``, ``tmax``. MNE's defaults apply otherwise.

    Returns:
        Updated context with the projectors added to the instance.
    """
    artifact = step_config.get('artifact', None)
    instance = step_config.get('instance', 'raw')
    apply = step_config.get('apply', False)
    if artifact not in ('ecg', 'eog'):
        raise ValueError(f"compute_ssp requires artifact: 'ecg' or 'eog' (got {artifact!r})")
    inst = _require_instance(data, instance, 'compute_ssp')

    kwargs = _passthrough(step_config, ('artifact', 'instance', 'apply', 'return_drop_log'))
    compute = (mne.preprocessing.compute_proj_ecg if artifact == 'ecg'
               else mne.preprocessing.compute_proj_eog)
    # MNE returns the recording's existing projectors along with the new ones.
    existing = {proj['desc'] for proj in inst.info['projs']}
    projs, events = compute(inst, **kwargs)[:2]
    new_projs = [proj for proj in (projs or []) if proj['desc'] not in existing]
    if new_projs:
        inst.add_proj(new_projs)
    if apply:
        inst.apply_proj()

    data['preprocessing_steps'].append({
        'step': 'compute_ssp',
        'instance': instance,
        'artifact': artifact,
        'params': kwargs,
        'n_events': int(len(events)) if events is not None else 0,
        'projectors': [proj['desc'] for proj in new_projs],
        'applied': bool(apply),
    })
    return data


@register("apply_gradient_compensation")
def apply_gradient_compensation(data, step_config):
    """Set the CTF synthetic-gradiometer compensation grade.

    Delegates to :meth:`mne.io.Raw.apply_gradient_compensation`. The CTF
    reference channels must still be present, since the compensation is
    computed from them.

    Args:
        data: Pipeline context. Must contain the key named by ``instance``.
        step_config: Step parameters:
            - ``instance`` (str): Recording to compensate. Default ``'raw'``.
            - ``grade`` (int): Compensation grade, 0 (none) to 3 (third-order
              gradiometer). Default 3.

    Returns:
        Updated context with the compensation applied in place.
    """
    instance = step_config.get('instance', 'raw')
    grade = step_config.get('grade', 3)
    inst = _require_instance(data, instance, 'apply_gradient_compensation')

    previous = inst.compensation_grade
    inst.apply_gradient_compensation(grade)

    data['preprocessing_steps'].append({
        'step': 'apply_gradient_compensation',
        'instance': instance,
        'previous_grade': previous,
        'grade': grade,
    })
    return data
