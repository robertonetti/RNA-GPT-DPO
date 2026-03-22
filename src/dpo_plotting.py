from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt


def _plot_violin_panel(
    ax,
    epochs: List[int],
    good_distributions: List[List[float]],
    bad_distributions: List[List[float]],
    pearson_values: List[float],
    title: str,
) -> None:
    good_positions = [e for e in epochs]
    bad_positions = [e for e in epochs]

    if len(good_distributions) > 0 and all(len(d) > 0 for d in good_distributions):
        violin_good = ax.violinplot(
            good_distributions,
            positions=good_positions,
            widths=0.32,
            showmeans=False,
            showmedians=True,
        )
        for body in violin_good["bodies"]:
            body.set_facecolor("tab:blue")
            body.set_edgecolor("black")
            body.set_alpha(0.35)
        violin_good["cmedians"].set_color("tab:blue")

    if len(bad_distributions) > 0 and all(len(d) > 0 for d in bad_distributions):
        violin_bad = ax.violinplot(
            bad_distributions,
            positions=bad_positions,
            widths=0.32,
            showmeans=False,
            showmedians=True,
        )
        for body in violin_bad["bodies"]:
            body.set_facecolor("tab:orange")
            body.set_edgecolor("black")
            body.set_alpha(0.35)
        violin_bad["cmedians"].set_color("tab:orange")

    for e, pearson_r, g_dist, b_dist in zip(epochs, pearson_values, good_distributions, bad_distributions):
        if len(g_dist) == 0 and len(b_dist) == 0:
            continue
        if len(g_dist) > 0 and len(b_dist) > 0:
            local_max = max(g_dist + b_dist)
        elif len(g_dist) > 0:
            local_max = max(g_dist)
        else:
            local_max = max(b_dist)
        label_txt = f"pearson={pearson_r:.3f}" if pearson_r == pearson_r else "pearson=nan"
        ax.text(e, local_max + 0.03, label_txt, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.plot([], [], color="tab:blue", linewidth=8, alpha=0.35, label="Good NLL")
    ax.plot([], [], color="tab:orange", linewidth=8, alpha=0.35, label="Bad NLL")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Per-sequence NLL (model)")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")


def save_epoch_figures(history: Dict[str, List[Any]], main_path: Path, violin_path: Path) -> None:
    epochs = history["epoch"]

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
    fig_main.savefig(main_path, dpi=140, bbox_inches="tight")
    plt.close(fig_main)

    fig_violin, axes_violin = plt.subplots(1, 3, figsize=(24, 6))

    _plot_violin_panel(
        axes_violin[0],
        epochs,
        history["val_good_seq_nll"],
        history["val_bad_seq_nll"],
        history["val_sep_corr"],
        "Validation NLL Distributions by Epoch (Good vs Bad)",
    )
    _plot_violin_panel(
        axes_violin[1],
        epochs,
        history["vae_good_seq_nll"],
        history["vae_bad_seq_nll"],
        history["vae_sep_corr"],
        "VAE Dist 25-30 Split NLL Distributions by Epoch (Good vs Bad)",
    )
    _plot_violin_panel(
        axes_violin[2],
        epochs,
        history["dist2530_good_seq_nll"],
        history["dist2530_bad_seq_nll"],
        history["dist2530_sep_corr"],
        "Validation Dist 25-30 Split NLL Distributions by Epoch (Good vs Bad)",
    )

    fig_violin.tight_layout()
    fig_violin.savefig(violin_path, dpi=140, bbox_inches="tight")
    plt.close(fig_violin)
