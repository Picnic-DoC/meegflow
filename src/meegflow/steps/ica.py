from collections import defaultdict
import mne
import numpy as np
from mne.utils import logger
from .registry import register


@register("ica")
def ica(data, step_config):
    """
    Apply Independent Component Analysis (ICA) for artifact removal.

    Decomposes the signal into independent components and can automatically
    detect and remove EOG and ECG artifacts.

    Parameters (via step_config)
    -----------------------------
    n_components : int, optional
        Number of ICA components (default: 20)
    method : str, optional
        ICA method: 'fastica', 'infomax', or 'picard' (default: 'fastica')
    random_state : int, optional
        Random state for reproducibility (default: 97)
    picks : list, optional
        Channel types to include in ICA. If None, defaults to MEEG channels.
    excluded_channels : list of str, optional
        Channel names to exclude from ICA decomposition
    find_eog : bool, optional
        Automatically find and exclude EOG artifacts (default: False)
    find_ecg : bool, optional
        Automatically find and exclude ECG artifacts (default: False)
    selected_indices : list of int, optional
        Manually specify component indices to exclude
    apply : bool, optional
        Apply ICA to remove artifacts (default: True)

    Updates
    -------
    data['ica'] : mne.preprocessing.ICA
        Fitted ICA object (stored for optional visualization)
    data['raw'] : mne.io.Raw
        If apply=True, artifacts are removed from raw data
    data['preprocessing_steps'] : list
        Appends step information including excluded components

    Returns
    -------
    data : dict
        Updated data dictionary with ICA applied
    """
    if 'raw' not in data:
        raise ValueError("ica step requires 'raw' in data")

    n_components = step_config.get('n_components', 20)
    random_state = step_config.get('random_state', 97)
    method = step_config.get('method', 'fastica')
    picks_params = step_config.get('picks', None)
    excluded_channels = step_config.get('excluded_channels', None)
    ica_l_freq = step_config.get('ica_fit_l_freq', 1.0)
    ica_h_freq = step_config.get('ica_fit_h_freq', None)
    eog_measure = step_config.get('eog_measure', 'correlation')
    eog_threshold = step_config.get('eog_threshold', 'auto')
    eog_channels = step_config.get('eog_channels', None)
    eog_l_freq = step_config.get('eog_l_freq', 1.0)
    eog_h_freq = step_config.get('eog_h_freq', 10.0)
    ecg_measure = step_config.get('ecg_measure', 'correlation')
    ecg_threshold = step_config.get('ecg_threshold', 'auto')
    ecg_channels = step_config.get('ecg_channels', None)
    ecg_l_freq = step_config.get('ecg_l_freq', 1.0)
    ecg_h_freq = step_config.get('ecg_h_freq', 10.0)
    selected_indices = step_config.get('selected_indices', None)
    apply = step_config.get('apply', True)

    raw = data['raw'].copy().filter(l_freq=ica_l_freq, h_freq=ica_h_freq)

    # --- Fit ICA on MEEG only (your _get_picks already defaults to eeg=True, eog=False) ---
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        method=method,
        max_iter='auto'
    )

    # Compute picks if provided
    picks = data.get_picks(raw.info, picks_params, excluded_channels)

    # Fit ICA
    ica.fit(raw, picks=picks)

    excluded_components = defaultdict(list)
    eog_detection_report = None
    ecg_detection_report = None

    # EOG
    if step_config.get('find_eog', False):

        if eog_channels is None:
            eog_channels = mne.pick_types(
                raw.info,
                eog=True,
                exclude='bads'
            )

        if isinstance(eog_channels, str):
            eog_channels = [eog_channels]

        if eog_channels is None or len(eog_channels) == 0:
            raise ValueError("No eog_channels on instance and no channel selected in the config. Can't perform automatic EOG ICA without EOG channels.")

        present_eog = [ch for ch in eog_channels if ch in raw.ch_names]
        if len(present_eog) == 0:
            raise ValueError('All EOG channels from config are not in the instance.')

        if len(present_eog) < len(eog_channels):
            non_existent_eog = [ch for ch in eog_channels if ch not in raw.ch_names]
            logger.warning(f'The following selected EOG channels are not in the instance: {non_existent_eog}')

        eog_indices = []
        eog_scores = []
        for ch_name in present_eog:
            cur_eog_indices, cur_eog_scores = ica.find_bads_eog(
                raw,
                ch_name=ch_name,
                measure=eog_measure,
                l_freq=eog_l_freq,
                h_freq=eog_h_freq,
                threshold=eog_threshold
            )

            eog_indices.extend(cur_eog_indices)
            eog_scores.append(
                cur_eog_scores.tolist()
                if isinstance(cur_eog_scores, np.ndarray)
                else cur_eog_scores
            )

        eog_indices = list(set(eog_indices))  # Unique indices

        for idx in eog_indices:
            excluded_components[idx].append('eog')

        eog_detection_report = {
            'eog_channels_requested': eog_channels,
            'eog_channels_present': present_eog,
            'eog_l_freq': eog_l_freq,
            'eog_h_freq': eog_h_freq,
            'eog_measure': eog_measure,
            'eog_threshold': eog_threshold,
            'eog_excluded_components': eog_indices,
            'eog_scores': eog_scores,
        }

    # ECG
    if step_config.get('find_ecg', False):

        if ecg_channels is None:
            ecg_channels = mne.pick_types(
                raw.info,
                ecg=True,
                exclude='bads'
            )

        if isinstance(ecg_channels, str):
            ecg_channels = [ecg_channels]

        if ecg_channels is None or len(ecg_channels) == 0:
            raise ValueError("No ecg_channels on instance and no channel selected in the config. Can't perform automatic ECG ICA without ECG channels.")

        present_ecg = [ch for ch in ecg_channels if ch in raw.ch_names]
        if len(present_ecg) == 0:
            raise ValueError('All ECG channels from config are not in the instance.')

        if len(present_ecg) < len(ecg_channels):
            non_existent_dropped_ecg = [ch for ch in ecg_channels if ch not in raw.ch_names]
            logger.warning(f'The following selected ECG channels are not in the instance: {non_existent_dropped_ecg}')

        ecg_indices = []
        ecg_scores = []
        for ch_name in present_ecg:
            cur_ecg_indices, cur_ecg_scores = ica.find_bads_ecg(
                raw,
                ch_name=ch_name,
                measure=ecg_measure,
                l_freq=ecg_l_freq,
                h_freq=ecg_h_freq,
                threshold=ecg_threshold
            )

            ecg_indices.extend(cur_ecg_indices)
            ecg_scores.append(
                cur_ecg_scores.tolist()
                if isinstance(cur_ecg_scores, np.ndarray)
                else cur_ecg_scores
            )

        ecg_indices = list(set(ecg_indices))  # Unique indices

        for idx in ecg_indices:
            excluded_components[idx].append('ecg')

        ecg_detection_report = {
            'ecg_channels_requested': ecg_channels,
            'ecg_channels_present': present_ecg,
            'ecg_l_freq': ecg_l_freq,
            'ecg_h_freq': ecg_h_freq,
            'ecg_measure': ecg_measure,
            'ecg_threshold': ecg_threshold,
            'ecg_excluded_components': ecg_indices,
            'ecg_scores': ecg_scores,
        }

    # Manual selection optional
    if selected_indices is not None:
        for idx in selected_indices:
            excluded_components[idx].append('selected')

    ica.exclude = sorted(excluded_components.keys())

    # Apply ICA to remove artifacts if requested
    if apply:
        ica.apply(data['raw'])

    data['ica'] = ica

    data['preprocessing_steps'].append({
        'step': 'ica',
        'n_components': n_components,
        'random_state': random_state,
        'method': method,
        'picks': picks_params,
        'excluded_channels': excluded_channels,
        'ica_l_freq': ica_l_freq,
        'ica_h_freq': ica_h_freq,
        'eog_detection': eog_detection_report or {},
        'ecg_detection': ecg_detection_report or {},
        'excluded_components': ica.exclude,
        'apply': apply,
    })

    return data
