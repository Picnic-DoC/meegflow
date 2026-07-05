import mne
from mne.utils import logger
from mne_bids import BIDSPath
from .registry import register
from ..savers import SAVERS as _SAVERS, FORMAT_EXTENSIONS as _FORMAT_EXTENSIONS


@register("save_clean_instance")
def save_clean_instance(data, step_config):
    """Save a preprocessed Raw or Epochs instance to the BIDS derivatives tree.

    The output path follows BIDS conventions under the pipeline's
    derivatives root. The save format is controlled by ``step_config['format']``;
    if omitted it defaults to ``'fif'`` for MNE objects and ``'pickle'`` otherwise.

    Args:
        data: Pipeline data dict. Must contain the key named by
            ``step_config['instance']`` (default ``'epochs'``).
        step_config: Step parameters:
            - ``instance`` (str): Key in ``data`` to save. Default ``'epochs'``.
            - ``format`` (str): One of ``'fif'``, ``'pickle'``, ``'hdf5'``,
              ``'numpy'``. Auto-detected if omitted.
            - ``overwrite`` (bool): Overwrite existing file. Default ``True``.
            - ``processing``, ``description``, ``datatype``, ``suffix``,
              ``extension``: BIDS path components (all optional).

    Returns:
        Updated data dict with ``data['{instance}_file']`` set to the saved
        path string.

    Raises:
        ValueError: If the requested instance is not in ``data``, or if an
            unknown format is requested.
    """
    instance = step_config.get('instance', 'epochs')
    overwrite = step_config.get('overwrite', True)
    processing = step_config.get('processing', None)
    description = step_config.get('description', None)
    datatype = step_config.get('datatype', None)
    suffix = step_config.get('suffix', None)
    extension = step_config.get('extension', None)
    fmt = step_config.get('format', None)

    if instance not in data:
        raise ValueError(
            f"save_clean_instance step requires '{instance}' to be present in data"
        )

    obj = data[instance]

    # Auto-detect format
    if fmt is None:
        fmt = 'fif' if isinstance(obj, (mne.io.BaseRaw, mne.BaseEpochs)) else 'pickle'

    if fmt not in _SAVERS:
        raise ValueError(f"Unknown format '{fmt}'. Choose from: {list(_SAVERS)}, or pass a callable.")

    saver = _SAVERS[fmt]

    # Default BIDS suffix based on instance type
    if suffix is None:
        if instance == 'epochs':
            suffix = 'epo'
        elif instance == 'raw':
            suffix = 'eeg'

    # Default extension from format — required when using a custom callable
    if extension is None:
        extension = _FORMAT_EXTENSIONS.get(fmt, '.pkl')

    deriv_root = data.derivatives_root(instance)

    bids_path = BIDSPath(
        subject=data['subject'],
        task=data['task'],
        session=data.get('session', None),
        acquisition=data.get('acquisition', None),
        datatype=datatype,
        root=deriv_root,
        suffix=suffix,
        extension=extension,
        processing=processing,
        description=description,
        check=False,
    )

    bids_path.mkdir(exist_ok=True)

    saver(obj, bids_path.fpath, overwrite)

    data[f'{instance}_file'] = str(bids_path)

    return data


@register("generate_json_report")
def generate_json_report(data, step_config):
    """Thin wrapper delegating to report.generate_json_report."""
    from ..report import generate_json_report as _render_json_report
    data['json_report'] = _render_json_report(
        data, step_config, data.derivatives_root("reports")
    )
    return data


@register("generate_html_report")
def generate_html_report(data, step_config):
    """Thin wrapper delegating to report.generate_html_report."""
    from ..report import generate_html_report as _render_html_report
    data['html_report'] = _render_html_report(
        data,
        step_config,
        get_picks=data.get_picks,
        deriv_root=data.derivatives_root("reports"),
    )
    return data
