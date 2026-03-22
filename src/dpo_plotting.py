from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt


def _plot_current_epoch_histogram(
    ax,
    good_distribution: List[float],
    bad_distribution: List[float],
    pearson_value: float,
    title: str,
) -> None:
    """Draw overlaid good/bad sequence-NLL histograms for one split.

    Inputs:
    - ax: Matplotlib axis where the histogram is drawn.
    - good_distribution: List of ``N_good`` per-sequence NLL values.
    - bad_distribution: List of ``N_bad`` per-sequence NLL values.
    - pearson_value: Scalar Pearson correlation for annotation.
    - title: Plot title.

    Output:
    - None. The function mutates ``ax`` in-place.
    """
    # Track whether each class has at least one point to draw.
    has_good = len(good_distribution) > 0
    has_bad = len(bad_distribution) > 0

    if has_good:
        ax.hist(
            good_distribution,
            bins=30,
            alpha=0.45,
            color="tab:blue",
            edgecolor="black",
            linewidth=0.5,
            label="Good NLL",
        )

    if has_bad:
        ax.hist(
            bad_distribution,
            bins=30,
            alpha=0.45,
            color="tab:orange",
            edgecolor="black",
            linewidth=0.5,
            label="Bad NLL",
        )

    if has_good or has_bad:
        # NaN-safe formatting of Pearson annotation.
        if pearson_value == pearson_value:
            pearson_text = f"Pearson = {pearson_value:.3f}"
        else:
            pearson_text = "Pearson = nan"
        ax.text(
            0.02,
            0.98,
            pearson_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_title(title)
    ax.set_xlabel("Per-sequence NLL (model)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    if has_good or has_bad:
        ax.legend(loc="best")


def save_epoch_figures(history: Dict[str, List[Any]], main_path: Path, violin_path: Path) -> None:
    """Generate and save the two standard training figures for the current history.

    Inputs:
    - history: Metric dictionary with per-epoch lists.
        Key size convention: each scalar list has length ``E`` (number of stored epochs).
    - main_path: Output file path for multi-panel epoch curves.
    - violin_path: Output file path for current-epoch histogram panels.

    Output:
    - None. Writes PNG files to disk.
    """
    # X-axis values shared by all time-series panels.
    epochs = history["epoch"]

    # Figure 1: training curves across epochs.
    fig_main, axes_main = plt.subplots(1, 3, figsize=(24, 6))

    ax_nll = axes_main[0]
    nll_train_good_line = ax_nll.plot(epochs, history["nll_train_good_rand"], marker="o", label="NLL Train Good (rand)")[0]
    nll_train_bad_line = ax_nll.plot(epochs, history["nll_train_bad_rand"], marker="o", label="NLL Train Bad (rand)")[0]
    nll_val_good_line = ax_nll.plot(epochs, history["nll_val_good_rand"], marker="o", label="NLL Val Good (rand)")[0]
    nll_val_bad_line = ax_nll.plot(epochs, history["nll_val_bad_rand"], marker="o", label="NLL Val Bad (rand)")[0]
    ax_nll.set_title("NLL Metrics (Random Batches) + Margins")
    ax_nll.set_xlabel("Epoch")
    ax_nll.set_ylabel("NLL")
    ax_nll.grid(alpha=0.3)

    ax_margin = ax_nll.twinx()
    margin_train_line = ax_margin.plot(
        epochs,
        history["nll_margin_train_rand"],
        color="black",
        linestyle="--",
        marker="x",
        label="Margin Train (Bad - Good)",
    )[0]
    margin_val_line = ax_margin.plot(
        epochs,
        history["nll_margin_val_rand"],
        color="gray",
        linestyle="-.",
        marker="d",
        label="Margin Val (Bad - Good)",
    )[0]
    ax_margin.set_ylabel("Margin")

    lines = [
        nll_train_good_line,
        nll_train_bad_line,
        nll_val_good_line,
        nll_val_bad_line,
        margin_train_line,
        margin_val_line,
    ]
    labels = [line.get_label() for line in lines]
    ax_nll.legend(lines, labels, loc="best")

    ax_dn = axes_main[1]
    ax_dn.plot(epochs, history["mean_like_model"], marker="o", label="DN mean token likelihood (model)")
    ax_dn.plot(epochs, history["mean_like_ref"], marker="o", linestyle="--", label="DN mean token likelihood (reference)")
    ax_dn.set_title("DN Mean Token Likelihood")
    ax_dn.set_xlabel("Epoch")
    ax_dn.set_ylabel("Likelihood")
    ax_dn.grid(alpha=0.3)
    ax_dn.legend()

    ax_dpo = axes_main[2]
    ax_dpo.plot(epochs, history["dpo_train_batch_epoch"], marker="o", label="DPO Train Batch Mean")
    ax_dpo.plot(epochs, history["dpo_train_full"], marker="o", label="DPO Full Train")
    ax_dpo.plot(epochs, history["dpo_val_full"], marker="o", label="DPO Full Val")
    ax_dpo.set_title("DPO Loss")
    ax_dpo.set_xlabel("Epoch")
    ax_dpo.set_ylabel("Loss")
    ax_dpo.grid(alpha=0.3)
    ax_dpo.legend()

    fig_main.tight_layout()
    # Persist chart and free figure memory immediately.
    fig_main.savefig(main_path, dpi=140, bbox_inches="tight")
    plt.close(fig_main)

    # Figure 2: per-sequence NLL distributions for the latest epoch only.
    current_epoch = epochs[-1]

    fig_hist, axes_hist = plt.subplots(1, 3, figsize=(24, 6))
    _plot_current_epoch_histogram(
        axes_hist[0],
        history["val_good_seq_nll"][-1],
        history["val_bad_seq_nll"][-1],
        history["val_sep_corr"][-1],
        title=f"Validation NLL Histogram - Epoch {current_epoch}",
    )
    _plot_current_epoch_histogram(
        axes_hist[1],
        history["vae_good_seq_nll"][-1],
        history["vae_bad_seq_nll"][-1],
        history["vae_sep_corr"][-1],
        title=f"VAE Dist 25-30 NLL Histogram - Epoch {current_epoch}",
    )
    _plot_current_epoch_histogram(
        axes_hist[2],
        history["dist2530_good_seq_nll"][-1],
        history["dist2530_bad_seq_nll"][-1],
        history["dist2530_sep_corr"][-1],
        title=f"Validation Dist 25-30 NLL Histogram - Epoch {current_epoch}",
    )

    fig_hist.tight_layout()
    fig_hist.savefig(violin_path, dpi=140, bbox_inches="tight")
    plt.close(fig_hist)
