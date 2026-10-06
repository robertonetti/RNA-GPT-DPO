from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _valid_series(values: List[float]) -> bool:
    return any(value == value for value in values)


def save_main_figure(
    history: Dict[str, List],
    output_path: Path,
    loss_label: str,
    full_tracking: bool,
    full_eval_loss: bool,
) -> None:
    iterations = history["iteration"]
    if not iterations:
        return
    loss_scope = "Full Pair Datasets" if full_eval_loss else "Random Pair Batches"

    if not full_tracking:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(iterations, history["train_loss"], marker="o", label=f"Train {loss_label}")
        if _valid_series(history["val_loss"]):
            ax.plot(iterations, history["val_loss"], marker="o", label=f"Val {loss_label}")
        ax.set_title(f"{loss_label} on {loss_scope}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(loss_label)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    ax_loss = axes[0]
    ax_loss.plot(iterations, history["train_loss"], marker="o", label=f"Train {loss_label}")
    if _valid_series(history["val_loss"]):
        ax_loss.plot(iterations, history["val_loss"], marker="o", label=f"Val {loss_label}")
    ax_loss.set_title(f"{loss_label} on {loss_scope}")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel(loss_label)
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_nll = axes[1]
    ax_nll.plot(iterations, history["train_good_nll_mean"], marker="o", label="Train good NLL")
    ax_nll.plot(iterations, history["train_bad_nll_mean"], marker="o", label="Train bad NLL")
    if _valid_series(history["val_good_nll_mean"]):
        ax_nll.plot(iterations, history["val_good_nll_mean"], marker="o", label="Val good NLL")
        ax_nll.plot(iterations, history["val_bad_nll_mean"], marker="o", label="Val bad NLL")
    if _valid_series(history["val_1_good_nll_mean"]):
        ax_nll.plot(iterations, history["val_1_good_nll_mean"], marker="o", label="Val 1 good NLL")
        ax_nll.plot(iterations, history["val_1_bad_nll_mean"], marker="o", label="Val 1 bad NLL")
    ax_nll.set_title("Mean Sequence NLL")
    ax_nll.set_xlabel("Iteration")
    ax_nll.set_ylabel("NLL")
    ax_nll.grid(alpha=0.3)
    ax_nll.legend(fontsize=8)

    ax_dn = axes[2]
    ax_dn.plot(
        iterations,
        history["dn_mean_token_likelihood"],
        marker="o",
        label="DN mean token likelihood",
    )
    ax_dn.set_title("DN Mean Token Likelihood")
    ax_dn.set_xlabel("Iteration")
    ax_dn.set_ylabel("Likelihood")
    ax_dn.grid(alpha=0.3)
    ax_dn.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_auroc_figure(history: Dict[str, List], output_path: Path) -> None:
    iterations = history["iteration"]
    if not iterations or "train_auroc" not in history:
        return

    panel_specs = [("Train AUROC", history["train_auroc"], "tab:blue")]
    if "val_auroc" in history and _valid_series(history["val_auroc"]):
        panel_specs.append(("Val AUROC", history["val_auroc"], "tab:orange"))
    if "val_1_auroc" in history and _valid_series(history["val_1_auroc"]):
        panel_specs.append(("Val 1 AUROC", history["val_1_auroc"], "tab:green"))

    fig, axes = plt.subplots(
        len(panel_specs),
        1,
        figsize=(10, 4.2 * len(panel_specs)),
        squeeze=False,
        sharex=True,
    )

    for row_idx, (title, values, color) in enumerate(panel_specs):
        ax = axes[row_idx][0]
        ax.plot(iterations, values, marker="o", color=color, label=title)
        valid_values = [float(value) for value in values if value == value]
        if valid_values:
            ymin = min(valid_values)
            ymax = max(valid_values)
            if ymax - ymin < 1e-6:
                pad = 0.02
            else:
                pad = max(0.01, 0.15 * (ymax - ymin))
            ax.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))
        else:
            ax.set_ylim(0.0, 1.0)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        ax.set_ylabel("AUROC")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    axes[-1][0].set_xlabel("Iteration")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _shared_bin_edges(distributions: List[List[float]], n_bins: int = 40) -> np.ndarray:
    """Equal-width bins spanning the range of all the given distributions."""
    all_values = np.concatenate([np.asarray(values, dtype=float) for values in distributions])
    low, high = float(all_values.min()), float(all_values.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    return np.linspace(low, high, n_bins + 1)


def _histogram_violin(
    ax,
    distributions: List[List[float]],
    positions: List[float],
    color: str,
    width: float = 0.75,
    n_bins: int = 40,
    bin_edges: np.ndarray | None = None,
) -> np.ndarray:
    """Draw each distribution as a symmetric vertical histogram ("histogram violin").

    Unlike a violin plot there is no kernel density fit: the shape is the raw
    histogram, mirrored around the x position. All distributions share the same
    bins; each one is scaled so that its tallest bin spans the full width.
    A black tick marks the median. Returns the bin edges used.
    """
    if bin_edges is None:
        bin_edges = _shared_bin_edges(distributions, n_bins)
    bin_heights = np.diff(bin_edges)

    for pos, values in zip(positions, distributions):
        counts, _ = np.histogram(values, bins=bin_edges)
        half_widths = 0.5 * width * counts / max(counts.max(), 1)
        ax.barh(bin_edges[:-1], 2 * half_widths, height=bin_heights, left=pos - half_widths,
                align="edge", color=color, alpha=0.4, edgecolor=color, linewidth=0.3)
        median = float(np.median(values))
        ax.hlines(median, pos - 0.5 * width, pos + 0.5 * width, color="black", linewidth=0.8)
    return bin_edges


def _select_evenly_spaced_indices(n_values: int, max_points: int = 10) -> List[int]:
    if n_values <= 0:
        return []
    if n_values <= max_points:
        return list(range(n_values))

    selected = {0, n_values - 1}
    remaining = max_points - 2
    for step_idx in range(1, remaining + 1):
        position = step_idx * (n_values - 1) / (remaining + 1)
        selected.add(int(round(position)))

    ordered = sorted(selected)
    while len(ordered) < max_points:
        for candidate in range(n_values):
            if candidate not in selected:
                selected.add(candidate)
                ordered = sorted(selected)
                if len(ordered) == max_points:
                    break

    if len(ordered) > max_points:
        ordered = ordered[: max_points - 1] + [n_values - 1]

    return ordered


def _plot_violin_panel(ax, iterations: List[int], good_history: List[List[float]], bad_history: List[List[float]], title: str) -> None:
    if not iterations or all(len(values) == 0 for values in good_history + bad_history):
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Sequence NLL")
        return

    positions = list(range(len(iterations)))
    # Good and bad share the same bins so their histograms are directly comparable.
    bin_edges = _shared_bin_edges(good_history + bad_history)
    _histogram_violin(ax, good_history, positions, "tab:green", bin_edges=bin_edges)
    _histogram_violin(ax, bad_history, positions, "tab:red", bin_edges=bin_edges)

    ax.plot([], [], color="tab:green", linewidth=8, alpha=0.4, label="Good")
    ax.plot([], [], color="tab:red", linewidth=8, alpha=0.4, label="Bad")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sequence NLL")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(value) for value in iterations], rotation=45, ha="right")
    ax.grid(alpha=0.25)
    ax.legend()


def save_violin_history(history: Dict[str, List], output_path: Path) -> None:
    iterations = history["iteration"]
    if not iterations:
        return
    selected_indices = _select_evenly_spaced_indices(len(iterations), max_points=10)
    selected_iterations = [iterations[idx] for idx in selected_indices]

    panel_specs = [
        ("Train", history["train_good_seq_nll"], history["train_bad_seq_nll"]),
    ]
    if any(len(values) > 0 for values in history["val_good_seq_nll"]):
        panel_specs.append(("Validation", history["val_good_seq_nll"], history["val_bad_seq_nll"]))
    if any(len(values) > 0 for values in history["val_1_good_seq_nll"]):
        panel_specs.append(("Validation 1", history["val_1_good_seq_nll"], history["val_1_bad_seq_nll"]))

    fig, axes = plt.subplots(len(panel_specs), 1, figsize=(14, 4.5 * len(panel_specs)), squeeze=False)
    for row_idx, (title, good_history, bad_history) in enumerate(panel_specs):
        _plot_violin_panel(
            axes[row_idx][0],
            selected_iterations,
            [good_history[idx] for idx in selected_indices],
            [bad_history[idx] for idx in selected_indices],
            title,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_distribution_violin(
    iterations: List[int],
    distributions: List[List[float]],
    output_path: Path,
    title: str,
    ylabel: str,
    color: str = "tab:blue",
    max_points: int = 12,
    show_p99: bool = False,
    top_labels: List[str] | None = None,
) -> None:
    """Histogram violin (see _histogram_violin) of one distribution per iteration. Overwrites output_path.

    With show_p99=True a short red bar marks the 99th percentile of each distribution.
    top_labels (one string per iteration, e.g. "AUC=0.81") are written above each histogram.
    """
    labels = top_labels if top_labels is not None else [None] * len(iterations)
    valid = [
        (it, values, label)
        for it, values, label in zip(iterations, distributions, labels)
        if len(values) > 0
    ]
    if not valid:
        return
    selected = _select_evenly_spaced_indices(len(valid), max_points=max_points)
    selected_iterations = [valid[idx][0] for idx in selected]
    selected_values = [np.asarray(valid[idx][1], dtype=np.float32) for idx in selected]
    selected_labels = [valid[idx][2] for idx in selected]

    fig, ax = plt.subplots(figsize=(14, 5))
    positions = list(range(len(selected_iterations)))
    _histogram_violin(ax, selected_values, positions, color)
    means = [float(values.mean()) for values in selected_values]
    ax.plot(positions, means, "o", color="black", markersize=4, label="Mean")
    if show_p99:
        p99 = [float(np.percentile(values, 99)) for values in selected_values]
        ax.hlines(p99, [pos - 0.25 for pos in positions], [pos + 0.25 for pos in positions],
                  color="tab:red", linewidth=2, label="99th percentile")
    if any(label is not None for label in selected_labels):
        # Text in axes-fraction y so it sits just above the plotting area.
        for pos, label in zip(positions, selected_labels):
            if label is not None:
                ax.text(pos, 1.01, label, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8)
        ax.set_title(title, pad=18)
    else:
        ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(value) for value in selected_iterations], rotation=45, ha="right")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
