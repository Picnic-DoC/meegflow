import numpy as np
import json

# BIDS datatypes MEEGFlow can process, mapped to the MNE channel types that
# identify them. Ordered by BIDS precedence: a recording containing MEG
# channels belongs to the 'meg' datatype even when it also contains EEG ones.
DATATYPE_CHANNEL_TYPES = {
    'meg': {'mag', 'grad', 'ref_meg'},
    'eeg': {'eeg'},
    'ieeg': {'ecog', 'seeg', 'dbs'},
    'nirs': {'fnirs_cw_amplitude', 'fnirs_fd_ac_amplitude', 'fnirs_fd_phase',
             'fnirs_od', 'hbo', 'hbr'},
}


def infer_datatype(inst, default='eeg'):
    """Infer the BIDS datatype of an MNE instance from its channel types.

    Used by the output steps to label derivatives with the datatype of the
    data actually being written, instead of assuming EEG.

    Parameters
    ----------
    inst : mne.io.BaseRaw | mne.BaseEpochs | mne.Evoked | None
        Instance whose channel types are inspected. Anything that does not
        expose ``get_channel_types`` (including None) yields ``default``.
    default : str
        Datatype returned when no known sensor type is found. Defaults to
        ``'eeg'``, which is what the output steps assumed unconditionally
        before this helper existed.

    Returns
    -------
    str
        One of the keys of ``DATATYPE_CHANNEL_TYPES``, or ``default``.
    """
    if inst is None or not hasattr(inst, 'get_channel_types'):
        return default

    ch_types = set(inst.get_channel_types(unique=True))
    for datatype, datatype_ch_types in DATATYPE_CHANNEL_TYPES.items():
        if ch_types & datatype_ch_types:
            return datatype

    return default


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalar and array types.

    Converts ``np.integer`` → ``int``, ``np.floating`` → ``float``, and
    ``np.ndarray`` → ``list`` so that NumPy values can be serialised with
    ``json.dumps``.
    """

    def default(self, obj):
        """Serialise NumPy types; fall back to the base encoder for others.

        Args:
            obj: Object to serialise.

        Returns:
            A JSON-serialisable Python built-in type.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)