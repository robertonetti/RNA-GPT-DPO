from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


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


def _style_violin(parts, color: str) -> None:
    for body in parts.get("bodies", []):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.4)
    for key in ("cmins", "cmaxes", "cbars", "cmedians"):
        artist = parts.get(key)
        if artist is not None:
            artist.set_color("black")
            artist.set_linewidth(0.8)


def _plot_violin_panel(ax, iterations: List[int], good_history: List[List[float]], bad_history: List[List[float]], title: str) -> None:
    if not iterations or all(len(values) == 0 for values in good_history + bad_history):
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Sequence NLL")
        return

    positions = list(range(len(iterations)))
    good_parts = ax.violinplot(good_history, positions=positions, widths=0.75, showmedians=True)
    bad_parts = ax.violinplot(bad_history, positions=positions, widths=0.75, showmedians=True)
    _style_violin(good_parts, "tab:green")
    _style_violin(bad_parts, "tab:red")

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

    panel_specs = [
        ("Train", history["train_good_seq_nll"], history["train_bad_seq_nll"]),
    ]
    if any(len(values) > 0 for values in history["val_good_seq_nll"]):
        panel_specs.append(("Validation", history["val_good_seq_nll"], history["val_bad_seq_nll"]))
    if any(len(values) > 0 for values in history["val_1_good_seq_nll"]):
        panel_specs.append(("Validation 1", history["val_1_good_seq_nll"], history["val_1_bad_seq_nll"]))

    fig, axes = plt.subplots(len(panel_specs), 1, figsize=(14, 4.5 * len(panel_specs)), squeeze=False)
    for row_idx, (title, good_history, bad_history) in enumerate(panel_specs):
        _plot_violin_panel(axes[row_idx][0], iterations, good_history, bad_history, title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
