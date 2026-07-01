"""
Report Generation Utilities for EEG Preprocessing Pipeline.

This module provides helper functions for generating HTML reports with
interactive visualizations and detailed preprocessing information.

Functions
---------
collect_bad_channels_from_steps(preprocessing_steps)
    Collects all unique bad channels from preprocessing step records.
    Aggregates channels marked as bad across multiple detection steps.

create_bad_channels_topoplot(info, bad_channels, figsize)
    Creates a topographical plot showing bad channels marked with red crosses.
    Uses the montage from the info object for electrode positions.

create_preprocessing_steps_table(preprocessing_steps)
    Creates an interactive HTML table with collapsible sections for each
    preprocessing step. Displays step parameters in a formatted, readable way.

Usage
-----
These functions are typically called by the generate_html_report step:

```python
from report import (
    collect_bad_channels_from_steps,
    create_bad_channels_topoplot,
    create_preprocessing_steps_table
)

# Collect bad channels
bad_channels = collect_bad_channels_from_steps(preprocessing_steps)

# Create topoplot
fig = create_bad_channels_topoplot(info, bad_channels)

# Create steps table
html_table = create_preprocessing_steps_table(preprocessing_steps)
```

The HTML reports include:
- Bad channels visualization (topoplot with red crosses)
- Preprocessing steps summary (collapsible table with parameters)
- Raw data plots, ICA components, epochs, and evoked responses (via MNE Report)

Report Structure
----------------
Generated HTML reports contain:
1. Bad Channels section (if any bad channels detected)
   - Topoplot showing channel positions
   - List of bad channel names
2. Preprocessing Steps section
   - Collapsible table with each step's parameters
   - Parameter values formatted based on type (dict, list, number, etc.)
3. Data Visualizations (via MNE Report)
   - ICA components (if ICA was applied)
   - Raw data traces
   - Events timeline (if events exist)
   - Epochs and evoked responses (if epochs exist)

See Also
--------
mne.Report : MNE's report generation class
"""
import json
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import numpy as np
import mne
from mne_bids import BIDSPath
from mne.utils import logger
import matplotlib.pyplot as plt
from .utils import NpEncoder


def collect_bad_channels_from_steps(preprocessing_steps: List[Dict[str, Any]]) -> List[str]:
    """
    Collect all bad channels from preprocessing steps.
    
    Parameters
    ----------
    preprocessing_steps : list of dict
        List of preprocessing step dictionaries, each potentially containing
        a 'bad_channels' key.
    
    Returns
    -------
    bad_channels : list of str
        Unique list of all bad channels found in preprocessing steps.
    """
    bad_channels = []
    for step in preprocessing_steps:
        if 'bad_channels' in step and step['bad_channels']:
            # Add bad channels from this step
            step_bad_channels = step['bad_channels']
            if isinstance(step_bad_channels, list):
                bad_channels.extend(step_bad_channels)
            elif isinstance(step_bad_channels, str):
                bad_channels.append(step_bad_channels)
    
    # TODO: replace by return list(set(bad_channels))
    # Return unique channels preserving order
    seen = set()
    unique_bad_channels = []
    for ch in bad_channels:
        if ch not in seen:
            seen.add(ch)
            unique_bad_channels.append(ch)
    
    return unique_bad_channels


def create_bad_channels_topoplot(
    info: mne.Info,
    bad_channels: List[str],
    outlines: Optional[Dict[str, List[int]]] = None,
    figsize: tuple = (8, 6)
) -> Optional[plt.Figure]:
    """
    Create a topoplot showing bad channels marked with red crosses.
    
    Uses the montage from the info object to determine the appropriate
    head shape and electrode positions.
    
    Parameters
    ----------
    info : mne.Info
        MNE Info object containing channel information and montage.
    bad_channels : list of str
        List of bad channel names to mark on the topoplot.
    outlines : dict or None, optional
        Dictionary defining head shape outlines. Default is None.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (8, 6).
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure containing the topoplot, or None if creation failed.
    """

    if not bad_channels:
        logger.info("No bad channels to plot")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get all EEG channel positions
    eeg_picks = mne.pick_types(info, eeg=True, exclude=[])

    if len(eeg_picks) == 0:
        logger.warning("No EEG channels found for topoplot")
        plt.close(fig)
        return None
    
    # Get channel names
    ch_names = [info['ch_names'][i] for i in eeg_picks]
    
    # Create data array (all zeros for white background)
    data_to_plot = np.zeros(len(eeg_picks))
    
    # Create mask for bad channels
    mask = np.array([ch in bad_channels for ch in ch_names])
    
    if not np.any(mask):
        logger.warning("None of the bad channels are in the EEG channels")
        plt.close(fig)
        return None
    
    # Plot topomap with white background
    # The montage from info will be used automatically by plot_topomap
    from mne.viz import plot_topomap
    im, cn = plot_topomap(
        data_to_plot, 
        info,
        axes=ax,
        show=False,
        cmap='Greys',
        vlim=(0, 0.1),
        outlines=outlines,
        mask=mask,
        mask_params=dict(
            marker='x',
            markerfacecolor='red',
            markeredgecolor='red',
            linewidth=0,
            markersize=15
        ),
        sensors=True,
        contours=0
    )
    
    ax.set_title(f'Bad Channels (n={len(bad_channels)})', fontsize=14, fontweight='bold')
    
    # Add text listing bad channels
    bad_channels_text = ', '.join(bad_channels)
    fig.text(0.5, 0.05, f'Bad channels: {bad_channels_text}', 
            ha='center', fontsize=10, wrap=True)
    
    plt.tight_layout()
    
    return fig


def create_preprocessing_steps_table(preprocessing_steps: List[Dict[str, Any]]) -> str:
    """
    Create an HTML table with collapsible rows for preprocessing steps.
    
    Uses MNE Report table styling with bootstrap-table classes.
    Each step's parameters are displayed in a two-column table:
    - First column: parameter keys
    - Second column: parameter values
      - Numbers displayed as numbers
      - Lists displayed as bullet points
      - Dicts displayed as prettified JSON with indent 4
    
    Parameters
    ----------
    preprocessing_steps : list of dict
        List of preprocessing step dictionaries containing step information.
    
    Returns
    -------
    html_content : str
        HTML string containing the styled, collapsible table.
    """
    if not preprocessing_steps:
        return ""
    
    def format_value(value):
        """Format a value based on its type."""
        if isinstance(value, dict):
            # Format dicts as prettified JSON with indent 4
            return f'<pre style="margin: 0; background-color: #f8f9fa; padding: 8px; border-radius: 4px;">{json.dumps(value, indent=4, cls=NpEncoder)}</pre>'
        elif isinstance(value, list):
            # Format lists as bullet points
            if not value:
                return '<em>empty list</em>'
            bullet_points = ''.join([f'<li>{item}</li>' for item in value])
            return f'<ul style="margin: 0; padding-left: 20px;">{bullet_points}</ul>'
        elif isinstance(value, (int, float)):
            # Format numbers as numbers (not strings)
            return str(value)
        elif value is None:
            return '<em>None</em>'
        else:
            # Format everything else as string
            return str(value)
    
    # Create HTML with collapsible sections for each step
    html_content = """
    <style>
        .step-container {
            margin-bottom: 20px;
        }
        .step-header {
            cursor: pointer;
            padding: 12px;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            font-weight: bold;
            user-select: none;
            transition: background-color 0.2s;
        }
        .step-header:hover {
            background-color: #e9ecef;
        }
        .step-header .toggle-icon {
            float: right;
            font-weight: bold;
        }
        .step-details {
            display: none;
            margin-top: 10px;
        }
        .step-details.active {
            display: block;
        }
        .params-table {
            width: 100%;
            margin-top: 10px;
        }
        .params-table td {
            padding: 8px;
            border: 1px solid #dee2e6;
            vertical-align: top;
        }
        .params-table td:first-child {
            background-color: #f8f9fa;
            font-weight: 500;
            width: 30%;
        }
    </style>
    <script>
        function toggleStep(stepId) {
            var details = document.getElementById('details-' + stepId);
            var icon = document.getElementById('icon-' + stepId);
            if (details.classList.contains('active')) {
                details.classList.remove('active');
                icon.textContent = '▼';
            } else {
                details.classList.add('active');
                icon.textContent = '▲';
            }
        }
    </script>
    """
    
    for idx, step in enumerate(preprocessing_steps, 1):
        step_name = step.get('step', 'Unknown')
        step_id = f"step-{idx}"
        
        # Create collapsible section for this step
        html_content += f"""
        <div class="step-container">
            <div class="step-header" onclick="toggleStep('{step_id}')">
                <span>Step {idx}: {step_name}</span>
                <span class="toggle-icon" id="icon-{step_id}">▼</span>
            </div>
            <div class="step-details" id="details-{step_id}">
                <table class="params-table table table-hover">
                    <tbody>
        """
        
        # Add each parameter as a row in the table
        for key, value in step.items():
            if key != 'step':
                formatted_value = format_value(value)
                html_content += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{formatted_value}</td>
                        </tr>
                """
        
        html_content += """
                    </tbody>
                </table>
            </div>
        </div>
        """
    
    return html_content


def generate_json_report(
    data: Dict[str, Any],
    step_config: Dict[str, Any],
    deriv_root: Path,
) -> str:
    """Write a JSON report of the preprocessing steps to the BIDS derivatives tree.

    The report includes subject/task/session metadata, raw data properties, and
    the full list of preprocessing steps recorded in ``data['preprocessing_steps']``.

    Parameters
    ----------
    data : dict
        Pipeline data dict containing at minimum ``'subject'``, ``'task'``, and
        ``'preprocessing_steps'``.
    step_config : dict
        Currently unused; reserved for future options.
    deriv_root : Path
        Derivatives root under which the report is written.

    Returns
    -------
    str
        Path to the written JSON report.
    """
    # JSON report
    report = {
        'subject': data['subject'],
        'task': data['task'],
        'session': data.get('session', None),
        'acquisition': data.get('acquisition', None),
        'preprocessing_steps': data.get('preprocessing_steps', []),
    }

    if 'raw' in data:
        report['raw'] = dict(
            n_channels=data['raw'].info.get('nchan'),
            sfreq=data['raw'].info.get('sfreq'),
            n_times=data['raw'].n_times
        )

    bids_path = BIDSPath(
        subject=data['subject'],
        task=data['task'],
        session=data.get('session', None),
        acquisition=data.get('acquisition', None),
        datatype="eeg",
        root=deriv_root,
        suffix="report",
        extension=".json",
        processing="clean",
        description="cleaned",
        check=False,
    )

    # Ensure directory exists
    bids_path.mkdir(exist_ok=True)

    with open(bids_path.fpath, 'w') as f:
        json.dump(report, f, indent=2, cls=NpEncoder)

    return str(bids_path)


def generate_html_report(
    data: Dict[str, Any],
    step_config: Dict[str, Any],
    get_picks: Callable,
    deriv_root: Path,
) -> str:
    """Generate an interactive HTML report of the preprocessing results.

    Produces a self-contained HTML file in the BIDS derivatives tree that
    includes a bad-channels topoplot, a preprocessing-steps table, and optional
    raw/ICA comparison plots.

    Parameters
    ----------
    data : dict
        Pipeline data dict containing at minimum ``'subject'``, ``'task'``, and
        ``'preprocessing_steps'``.
    step_config : dict
        Step parameters (picks, excluded_channels, outlines, compare_instances,
        and the various plot_*_kwargs).
    get_picks : callable
        ``get_picks(info, picks_params, excluded_channels) -> list[int]`` used to
        select channels for the figures.
    deriv_root : Path
        Derivatives root under which the report is written.

    Returns
    -------
    str
        Path to the written HTML report.
    """
    picks_params = step_config.get('picks', None)
    excluded_channels = step_config.get('excluded_channels', None)
    outlines = step_config.get('outlines', 'head')
    compare_instances = step_config.get('compare_instances', [])
    plot_raw_kwargs = step_config.get('plot_raw_kwargs', {})
    plot_ica_kwargs = step_config.get('plot_ica_kwargs', {})
    plot_events_kwargs = step_config.get('plot_events_kwargs', {})
    plot_epochs_kwargs = step_config.get('plot_epochs_kwargs', {})
    plot_evokeds_kwargs = step_config.get('plot_evokeds_kwargs', {})

    if 'preprocessing_steps' not in data:
        raise ValueError("generate_html_report requires 'preprocessing_steps' in data")
    elif not isinstance(data['preprocessing_steps'], list):
        raise ValueError("data['preprocessing_steps'] must be a list")

    # Get info from epochs if available, otherwise from raw
    inst = data['raw'] if 'raw' in data else data['epochs'] if 'epochs' in data else None
    if inst is None:
        raise ValueError("generate_html_report requires either 'raw' or 'epochs' in data")

    # Compute picks for channel selection
    picks = get_picks(inst.info, picks_params, excluded_channels)

    preprocessing_steps = data['preprocessing_steps']

    html_report = mne.Report(title=f'Preprocessing Report - Subject {data["subject"]}')

    # Add bad channels topoplot section
    bad_channels = collect_bad_channels_from_steps(preprocessing_steps)

    # Create topoplot if we have bad channels and info
    if len(bad_channels) > 0:
        logger.info(f"Adding bad channels topoplot with {len(bad_channels)} bad channels")
        fig = create_bad_channels_topoplot(inst.info, bad_channels, outlines=outlines)

        if fig is not None:
            # Add to report
            html_report.add_figure(
                fig=fig,
                title='Bad Channels',
                caption=f'Topoplot showing {len(bad_channels)} bad channels marked with red crosses'
            )
            plt.close(fig)

    # Add preprocessing steps table section
    html_content = create_preprocessing_steps_table(data['preprocessing_steps'])

    # ---------- Preprocessing steps ----------
    if html_content is not None:
        # Add the HTML table to the report
        html_report.add_html(
            html=html_content,
            title='Preprocessing Steps',
        )

    # ---------- ICA ----------
    if data.get('ica', None) is not None:

        html_report.add_ica(
            ica=data['ica'],
            title='ICA Components',
            inst=None,
            **plot_ica_kwargs
        )

        ica_step = [step for step in preprocessing_steps if step['step'] == 'ica']
        ica_step = ica_step[-1] if len(ica_step) > 0 else {}
        eog_step_report = ica_step.get('eog_detection', {})
        eog_idx = eog_step_report.get('eog_excluded_components', []) or []
        eog_scores = eog_step_report.get('eog_scores', None)
        ecg_step_report = ica_step.get('ecg_detection', {})
        ecg_idx = ecg_step_report.get('ecg_excluded_components', [])
        ecg_scores = ecg_step_report.get('ecg_scores', None)

        if len(eog_idx) > 0:

            if eog_scores is not None:
                scores = np.array(eog_scores, dtype=float)

                if scores.ndim == 1:
                    scores = scores.reshape(-1, 1)  # Make it 2D for uniform processing

                # Heatmap (EOG channels x ICA components)
                fig = plt.figure()
                ax = fig.add_subplot(111)

                im = ax.imshow(scores, aspect="auto", origin="lower")

                n_components = scores.shape[1]

                # X axis: ICA components as discrete labels 1..N
                ax.set_xticks(np.arange(n_components))
                ax.set_xticklabels(np.arange(n_components))

                ax.set_xlabel("ICA component")
                ax.set_ylabel("EOG channel")

                eog_names = (
                    eog_step_report.get("eog_channels_present", None)
                    or eog_step_report.get("eog_channels_requested", None)
                    or []
                )
                if isinstance(eog_names, list) and len(eog_names) == scores.shape[0]:
                    ax.set_yticks(np.arange(len(eog_names)))
                    ax.set_yticklabels(eog_names)

                ax.set_title("EOG scores (per EOG channel × ICA component)")

                fig.colorbar(im, ax=ax, shrink=0.8, label="EOG score")

                html_report.add_figure(
                    fig=fig,
                    title="ICA - EOG scores heatmap",
                    section='ICA - EOG'
                )
                plt.close(fig)

                # Aggregate to 1 score per component for barplot
                scores_1d = np.max(np.abs(scores), axis=0)

                # Barplot (always 1D after aggregation if needed)
                fig1 = plt.figure()
                ax = fig1.add_subplot(111)
                ax.bar(np.arange(len(scores_1d)), scores_1d)
                ax.set_xlabel("ICA component")
                ax.set_ylabel("max |EOG score| across EOG channels" if (eog_scores is not None and np.array(eog_scores).ndim == 2) else "|EOG score|")
                ax.set_title(f"EOG scores (selected: {eog_idx})")
                html_report.add_figure(
                    fig=fig1,
                    title="ICA - EOG scores",
                    section='ICA - EOG'
                )
                plt.close(fig1)

        if len(ecg_idx) > 0:

            if ecg_scores is not None:
                scores = np.array(ecg_scores, dtype=float)

                if scores.ndim == 1:
                    scores = scores.reshape(-1, 1)  # Make it 2D for uniform processing

                # Heatmap (ECG channels x ICA components)
                fig = plt.figure()
                ax = fig.add_subplot(111)

                im = ax.imshow(scores, aspect="auto", origin="lower")

                n_components = scores.shape[1]

                # X axis: ICA components as discrete labels 1..N
                ax.set_xticks(np.arange(n_components))
                ax.set_xticklabels(np.arange(n_components))

                ax.set_xlabel("ICA component")
                ax.set_ylabel("ECG channel")

                ecg_names = (
                    ecg_step_report.get("ecg_channels_present", None)
                    or ecg_step_report.get("ecg_channels_requested", None)
                    or []
                )
                if isinstance(ecg_names, list) and len(ecg_names) == scores.shape[0]:
                    ax.set_yticks(np.arange(len(ecg_names)))
                    ax.set_yticklabels(ecg_names)

                ax.set_title("ECG scores (per ECG channel × ICA component)")
                fig.colorbar(im, ax=ax, shrink=0.8, label="ECG score")

                html_report.add_figure(
                    fig=fig,
                    title="ICA - ECG scores heatmap",
                    section='ICA - EOG'
                )
                plt.close(fig)

                # Aggregate to 1 score per component for barplot
                scores_1d = np.max(np.abs(scores), axis=0)

                # Barplot (always 1D after aggregation if needed)
                fig1 = plt.figure()
                ax = fig1.add_subplot(111)
                ax.bar(np.arange(len(scores_1d)), scores_1d)
                ax.set_xlabel("ICA component")
                ax.set_ylabel("max |ECG score| across ECG channels" if (ecg_scores is not None and np.array(ecg_scores).ndim == 2) else "|ECG score|")
                ax.set_title(f"ECG scores (selected: {ecg_idx})")
                html_report.add_figure(
                    fig=fig1,
                    title="ICA - ECG scores",
                    section='ICA - EOG'
                )
                plt.close(fig1)

    # ---------- Compare instances preprocessing (full recording) ----------
    for contrast in compare_instances:
        inst_a_name = contrast['instance_a']['name']
        inst_a_label = contrast['instance_a']['label']
        inst_b_name = contrast['instance_b']['name']
        inst_b_label = contrast['instance_b']['label']

        if inst_a_name not in data or inst_b_name not in data:
            raise ValueError(f"compare_instances step requires both '{inst_a_name}' and '{inst_b_name}' in data")

        inst_a = data[inst_a_name]
        inst_b = data[inst_b_name]

        # Ensure channel alignment (same channel order)
        ch_names_picks = get_picks(
            inst.info,
            picks_params,
            excluded_channels
        )
        ch_names_a = sorted([inst_a.ch_names[pick] for pick in ch_names_picks])
        ch_names_b = sorted([inst_b.ch_names[pick] for pick in ch_names_picks])
        if set(ch_names_a) != set(ch_names_b):
            raise ValueError(f"compare_instances step: channel mismatch between '{inst_a}' and '{inst_b}' after picking")

        raw_b = inst_b.copy().pick(picks=ch_names_picks).reorder_channels(ch_names_a)
        raw_a = inst_a.copy().pick(picks=ch_names_picks).reorder_channels(ch_names_b)

        Xb = raw_b.get_data()
        Xa = raw_a.get_data()
        times = raw_a.times

        # Metrics over full recording
        gfp_b = np.std(Xb, axis=0)
        gfp_a = np.std(Xa, axis=0)

        mean_b = np.mean(Xb, axis=0)
        mean_a = np.mean(Xa, axis=0)

        diff_abs = np.mean(np.abs(Xb - Xa), axis=0)

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(times, gfp_b, color='red', alpha=0.35, label=inst_b_label)
        axes[0].plot(times, gfp_a, color='black', linewidth=1.0, label=inst_a_label)
        axes[0].set_title('Global Field Power (full recording)')
        axes[0].legend(loc='upper right')

        axes[1].plot(times, mean_b, color='red', alpha=0.35, label=inst_b_label)
        axes[1].plot(times, mean_a, color='black', linewidth=1.0, label=inst_a_label)
        axes[1].set_title('Mean across channels (full recording)')
        axes[0].legend(loc='upper right')

        axes[2].plot(times, diff_abs, color='purple', linewidth=1.0)
        axes[2].set_title(f'Mean absolute difference |{inst_a_label} - {inst_b_label}| (full recording)')
        axes[2].set_xlabel('Time (s)')

        fig.tight_layout()
        html_report.add_figure(
            fig=fig,
            title=contrast['title'],
            section='Contrasts'
        )
        plt.close(fig)

    # ---------- Cleaned Raw report ----------
    if data.get('raw', None) is not None:
        html_report.add_raw(
            raw=data['raw'].copy().pick(picks=picks),
            title='Clean Raw Data',
            **plot_raw_kwargs
        )

    # ---------- Events report ----------
    if 'events' in data and data['events'] is not None:
        html_report.add_events(
            events=data['events'],
            event_id=data.get('event_id', None),
            sfreq=data['events_sfreq'],
            title='Found Events',
            **plot_events_kwargs
        )

    # ---------- Cleaned Epochs report ----------
    if data.get('epochs', None) is not None:

        epochs = data['epochs'].copy().pick(picks=picks)

        html_report.add_epochs(
            epochs=epochs,
            title='Clean Epochs',
            **plot_epochs_kwargs
        )

        html_report.add_evokeds(
            evokeds=epochs.average(by_event_type=True),
            n_time_points=step_config.get('n_time_points', None),
            **plot_evokeds_kwargs
        )

    bids_path = BIDSPath(
        subject=data['subject'],
        task=data['task'],
        session=data.get('session', None),
        acquisition=data.get('acquisition', None),
        datatype="eeg",
        root=deriv_root,
        suffix="report",
        extension=".html",
        processing="clean",
        description="cleaned",
        check=False,
    )

    # Ensure directory exists
    bids_path.mkdir(exist_ok=True)

    html_report.save(bids_path.fpath, overwrite=True, open_browser=False)

    return str(bids_path)
