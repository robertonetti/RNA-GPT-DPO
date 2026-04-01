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
