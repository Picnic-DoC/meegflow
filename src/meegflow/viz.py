"""Visualization utilities for dropped epoch analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

def _epoch_event_labels(epochs: mne.Epochs) -> np.ndarray:
    """Return a string label per epoch (event type)."""
    # epochs.events[:, 2] are the event codes
    inv = {v: k for k, v in epochs.event_id.items()}
    codes = epochs.events[:, 2]
    labels = np.array([inv.get(int(c), f"code_{int(c)}") for c in codes], dtype=object)
    return labels

def droplog_dataframe(epochs: mne.Epochs) -> pd.DataFrame:
    """Build a tidy DataFrame of dropped epochs and their rejection reasons.

    Args:
        epochs: Epochs object after rejection (``drop_log`` must be populated).

    Returns:
        DataFrame with columns ``epoch_ix``, ``event_type``, and ``reason``.
        One row per (dropped epoch, rejection reason) pair. Empty if no epochs
        were dropped.
    """
    # `drop_log` has one entry per *original* epoch, while event labels are only
    # available for the *surviving* epochs. Map the surviving labels onto their
    # position in `drop_log` via `epochs.selection` so indices never go out of
    # range. Dropped epochs have no recoverable event type; use a non-null
    # sentinel so they aren't silently dropped by pandas' groupby(dropna=True).
    surviving_labels = _epoch_event_labels(epochs)
    label_by_droplog_ix = {
        orig_ix: surviving_labels[kept_ix]
        for kept_ix, orig_ix in enumerate(epochs.selection)
    }
    UNKNOWN_EVENT_TYPE = "unknown"

    rows = []
    for ei, reasons in enumerate(epochs.drop_log):
        # epochs.drop_log[ei] is a tuple/list of ’reasons’; empty means kept
        if not reasons:
            continue
        event_type = label_by_droplog_ix.get(ei, UNKNOWN_EVENT_TYPE)
        for r in reasons:
            rows.append({"epoch_ix": ei, "event_type": event_type, "reason": r})
    return pd.DataFrame(rows)

def plot_drops_by_reason_and_type(
    epochs: mne.Epochs,
    title: str = "Dropped epochs by reason and event type",
) -> plt.Figure:
    """Plot a stacked bar chart of dropped epochs broken down by reason and event type.

    Args:
        epochs: Epochs object after rejection (``drop_log`` must be populated).
        title: Title for the figure axes.

    Returns:
        Matplotlib Figure. If no epochs were dropped the figure contains a
        text message instead of a chart.
    """
    df = droplog_dataframe(epochs)

    fig, ax = plt.subplots(figsize=(8, 4))
    if df.empty:
        ax.text(0.5, 0.5, "No dropped epochs.", ha="center", va="center")
        ax.axis("off")
        return fig

    # counts(reason, event_type)
    tab = (
        df.groupby(["reason", "event_type"])
          .size()
          .unstack("event_type", fill_value=0)
          .sort_index()
    )

    tab.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Drop reason")
    ax.set_ylabel("# epochs dropped")
    ax.legend(title="Event type", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig
