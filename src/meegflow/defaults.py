"""Datatype-dependent defaults.

A configuration may declare the kind of data it processes with the optional
top-level key ``datatype`` (``'eeg'`` or ``'meg'``). Steps read their defaults
from it whenever a parameter is omitted: which channels are picked, the
peak-to-peak rejection thresholds, and the datatype that derivatives are
labelled with. A parameter given explicitly in a step always wins.

``'eeg'`` is the default, so a configuration without the key behaves exactly
as it did before the key existed.
"""
from typing import Any, Dict, Optional

DEFAULT_DATATYPE = 'eeg'
SUPPORTED_DATATYPES = ('eeg', 'meg')

# Keyword arguments for ``mne.pick_types`` when a step's ``picks`` is omitted
# (``PipelineContext.get_picks`` adds ``exclude='bads'``). MEG reference
# channels are left out: they are not brain signals, and MNE's ICA refuses them.
DEFAULT_PICKS = {
    'eeg': dict(eeg=True, eog=False, meg=False),
    'meg': dict(meg=True, ref_meg=False, eeg=False, eog=False),
}

# Variance below which ``find_flat_channels`` marks a channel as flat, per
# channel type, in SI units squared: (1 fT)^2 for magnetometers and MEG
# reference channels, (1 fT/cm)^2 for planar gradiometers. Every other channel
# type uses FLAT_VARIANCE_FALLBACK, the original single threshold of
# (1 uV)^2, which is calibrated for voltages (EEG, EOG, ECG, EMG, ...).
FLAT_VARIANCE = {'mag': 1e-30, 'grad': 1e-26, 'ref_meg': 1e-30}
FLAT_VARIANCE_FALLBACK = 1e-12


def resolve_datatype(config: Optional[Dict[str, Any]]) -> str:
    """Return the configuration's top-level ``datatype``, validated.

    Parameters
    ----------
    config : dict or None
        Pipeline configuration.

    Returns
    -------
    str
        ``'eeg'`` or ``'meg'``; ``'eeg'`` when the key is absent.

    Raises
    ------
    ValueError
        If ``datatype`` is set to anything other than a supported value.
    """
    datatype = (config or {}).get('datatype', DEFAULT_DATATYPE)
    if datatype is None:
        return DEFAULT_DATATYPE
    datatype = str(datatype).lower()
    if datatype not in SUPPORTED_DATATYPES:
        raise ValueError(
            f"Unsupported top-level datatype '{datatype}'. "
            f"Choose one of {list(SUPPORTED_DATATYPES)}."
        )
    return datatype


def configured_datatype(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the top-level ``datatype`` only if the configuration sets it.

    Used where the historical behaviour must be kept when the key is absent,
    for instance the datatype folder a derivative is written to.
    """
    if not config or config.get('datatype') is None:
        return None
    return resolve_datatype(config)
