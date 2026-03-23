from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Sequence

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


def _with_dataset_description(
    base_title: str,
    dataset_key: str,
    dataset_descriptions: Dict[str, str] | None,
) -> str:
    """Append dataset description (if provided) to a panel title."""
    if dataset_descriptions is None:
        return base_title
    desc = dataset_descriptions.get(dataset_key, "").strip()
    if desc == "":
        return base_title
    return f"{base_title}\n{desc}"


def _percentile(values: Sequence[float], q: float) -> float:
    """Compute percentile from empirical CDF (no interpolation).

    Returns the smallest value x such that the cumulative fraction of samples
    with value <= x is at least q/100.
    """
    if len(values) == 0:
        return float("nan")
    if q <= 0.0:
        return float(min(values))
    if q >= 100.0:
        return float(max(values))

    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    # Nearest-rank on empirical CDF: index of first point where CDF >= q/100.
    rank = (q / 100.0) * n
    idx = int(rank)
    if idx < rank:
        idx += 1
    idx = max(1, min(idx, n)) - 1
    return float(sorted_vals[idx])


def save_epoch_figures(
    history: Dict[str, List[Any]],
    main_path: Path,
    violin_path: Path,
    dataset_descriptions: Dict[str, str] | None = None,
) -> None:
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

    # Figure 1: training curves across epochs (2x2 layout).
    fig_main, axes_main = plt.subplots(2, 2, figsize=(20, 12))

    ax_nll = axes_main[0, 0]
    nll_train_good_line = ax_nll.plot(epochs, history["nll_train_good_rand"], marker="o", label="NLL Train Good (rand)")[0]
    nll_train_bad_line = ax_nll.plot(epochs, history["nll_train_bad_rand"], marker="o", label="NLL Train Bad (rand)")[0]
    has_val_nll = any(v == v for v in history["nll_val_good_rand"]) and any(v == v for v in history["nll_val_bad_rand"])
    if has_val_nll:
        nll_val_good_line = ax_nll.plot(epochs, history["nll_val_good_rand"], marker="o", label="NLL Val Good (rand)")[0]
        nll_val_bad_line = ax_nll.plot(epochs, history["nll_val_bad_rand"], marker="o", label="NLL Val Bad (rand)")[0]
    else:
        nll_val_good_line = None
        nll_val_bad_line = None
    nll_title = "NLL Metrics (Random Batches)"
    train_desc = "" if dataset_descriptions is None else dataset_descriptions.get("train", "").strip()
    val_desc = "" if dataset_descriptions is None else dataset_descriptions.get("val", "").strip()
    if train_desc or val_desc:
        nll_title += f"\nTrain: {train_desc or '-'} | Val: {val_desc or '-'}"
    ax_nll.set_title(nll_title)
    ax_nll.set_xlabel("Epoch")
    ax_nll.set_ylabel("NLL")
    ax_nll.grid(alpha=0.3)

    lines = [
        nll_train_good_line,
        nll_train_bad_line,
    ]
    if nll_val_good_line is not None:
        lines.append(nll_val_good_line)
    if nll_val_bad_line is not None:
        lines.append(nll_val_bad_line)
    labels = [line.get_label() for line in lines]
    ax_nll.legend(lines, labels, loc="best")

    ax_margin = axes_main[0, 1]
    margin_train_line = ax_margin.plot(
        epochs,
        history["nll_margin_train_rand"],
        color="black",
        linestyle="--",
        marker="x",
        label="Margin Train (Bad - Good)",
    )[0]
    has_val_margin = any(v == v for v in history["nll_margin_val_rand"])
    if has_val_margin:
        margin_val_line = ax_margin.plot(
            epochs,
            history["nll_margin_val_rand"],
            color="gray",
            linestyle="-.",
            marker="d",
            label="Margin Val (Bad - Good)",
        )[0]
    else:
        margin_val_line = None
    margin_title = "NLL Margin (Bad - Good)"
    if train_desc or val_desc:
        margin_title += f"\nTrain: {train_desc or '-'} | Val: {val_desc or '-'}"
    ax_margin.set_title(margin_title)
    ax_margin.set_xlabel("Epoch")
    ax_margin.set_ylabel("Margin")
    ax_margin.grid(alpha=0.3)
    margin_lines = [margin_train_line]
    if margin_val_line is not None:
        margin_lines.append(margin_val_line)
    ax_margin.legend(margin_lines, [line.get_label() for line in margin_lines], loc="best")

    ax_dn = axes_main[1, 0]
    ax_dn.plot(epochs, history["mean_like_model"], marker="o", label="DN mean token likelihood (model)")
    ax_dn.plot(epochs, history["mean_like_ref"], marker="o", linestyle="--", label="DN mean token likelihood (reference)")
    ax_dn.set_title("DN Mean Token Likelihood")
    ax_dn.set_xlabel("Epoch")
    ax_dn.set_ylabel("Likelihood")
    ax_dn.grid(alpha=0.3)
    ax_dn.legend()

    ax_dpo = axes_main[1, 1]
    ax_dpo.plot(epochs, history["dpo_train_batch_epoch"], marker="o", label="DPO Train Batch Mean")
    ax_dpo.plot(epochs, history["dpo_train_full"], marker="o", label="DPO Full Train")
    if any(v == v for v in history["dpo_val_full"]):
        ax_dpo.plot(epochs, history["dpo_val_full"], marker="o", label="DPO Full Val")
    dpo_title = "DPO Loss"
    if train_desc or val_desc:
        dpo_title += f"\nTrain: {train_desc or '-'} | Val: {val_desc or '-'}"
    ax_dpo.set_title(dpo_title)
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

    panel_specs = []
    if len(history["val_good_seq_nll"][-1]) > 0 or len(history["val_bad_seq_nll"][-1]) > 0:
        panel_specs.append(
            (
                history["val_good_seq_nll"][-1],
                history["val_bad_seq_nll"][-1],
                history["val_sep_corr"][-1],
                _with_dataset_description(
                    f"Validation NLL Histogram - Epoch {current_epoch}",
                    "val",
                    dataset_descriptions,
                ),
            )
        )
    if len(history["vae_good_seq_nll"][-1]) > 0 or len(history["vae_bad_seq_nll"][-1]) > 0:
        panel_specs.append(
            (
                history["vae_good_seq_nll"][-1],
                history["vae_bad_seq_nll"][-1],
                history["vae_sep_corr"][-1],
                _with_dataset_description(
                    f"VAE Dist 25-30 NLL Histogram - Epoch {current_epoch}",
                    "val_1",
                    dataset_descriptions,
                ),
            )
        )
    if len(history["dist2530_good_seq_nll"][-1]) > 0 or len(history["dist2530_bad_seq_nll"][-1]) > 0:
        panel_specs.append(
            (
                history["dist2530_good_seq_nll"][-1],
                history["dist2530_bad_seq_nll"][-1],
                history["dist2530_sep_corr"][-1],
                _with_dataset_description(
                    f"Validation Dist 25-30 NLL Histogram - Epoch {current_epoch}",
                    "val_2",
                    dataset_descriptions,
                ),
            )
        )

    if len(panel_specs) > 0:
        fig_hist, axes_hist = plt.subplots(1, len(panel_specs), figsize=(8 * len(panel_specs), 6))
        if len(panel_specs) == 1:
            axes = [axes_hist]
        else:
            axes = list(axes_hist)

        for ax, (good_dist, bad_dist, corr_value, panel_title) in zip(axes, panel_specs):
            _plot_current_epoch_histogram(
                ax,
                good_dist,
                bad_dist,
                corr_value,
                title=panel_title,
            )

        fig_hist.tight_layout()
        fig_hist.savefig(violin_path, dpi=140, bbox_inches="tight")
        plt.close(fig_hist)


def _style_violin_parts(parts, color: str) -> None:
    """Apply consistent style to all artists produced by ``ax.violinplot``."""
    for body in parts.get("bodies", []):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.45)

    for key in ["cmeans", "cmins", "cmaxes", "cbars", "cmedians"]:
        artist = parts.get(key)
        if artist is not None:
            artist.set_color("black")
            artist.set_linewidth(0.8)


def _plot_violin_history_panel(
    ax,
    epochs: Sequence[int],
    good_distributions: Sequence[Sequence[float]],
    bad_distributions: Sequence[Sequence[float]],
    pearsons: Sequence[float],
    title: str,
) -> None:
    """Draw one subplot with epoch-wise good/bad NLL violins."""
    if len(epochs) == 0:
        ax.text(
            0.5,
            0.5,
            "No data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Per-sequence NLL", fontsize=14)
        ax.grid(alpha=0.3)
        return

    # Use compact categorical positions to avoid large empty gaps between epochs.
    positions = [float(i) for i in range(len(epochs))]
    width = 0.75

    good_positions = [p for p in positions]
    bad_positions = [p for p in positions]

    good_parts = ax.violinplot(
        dataset=good_distributions,
        positions=good_positions,
        widths=width,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    _style_violin_parts(good_parts, color="tab:green")

    bad_parts = ax.violinplot(
        dataset=bad_distributions,
        positions=bad_positions,
        widths=width,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    _style_violin_parts(bad_parts, color="tab:red")

    # Print metrics near each epoch violin using that epoch's current-model NLLs.
    data_max_values: List[float] = []
    for good_dist, bad_dist in zip(good_distributions, bad_distributions):
        local_max_candidates: List[float] = []
        if len(good_dist) > 0:
            local_max_candidates.append(max(good_dist))
        if len(bad_dist) > 0:
            local_max_candidates.append(max(bad_dist))
        if len(local_max_candidates) > 0:
            data_max_values.append(max(local_max_candidates))

    y_min, y_max = ax.get_ylim()
    if len(data_max_values) > 0:
        global_data_max = max(data_max_values)
        y_span = max(y_max - y_min, 1e-6)
        text_offset = 0.03 * y_span
        target_top = global_data_max + 0.10 * y_span
        if target_top > y_max:
            ax.set_ylim(y_min, target_top)
            y_min, y_max = ax.get_ylim()

        for i, (x_pos, pearson_value) in enumerate(zip(positions, pearsons)):
            local_max = data_max_values[i] if i < len(data_max_values) else global_data_max
            if pearson_value == pearson_value:
                pearson_text = f"r={pearson_value:.2f}"
            else:
                pearson_text = "r=nan"

            good_dist = good_distributions[i] if i < len(good_distributions) else []
            bad_dist = bad_distributions[i] if i < len(bad_distributions) else []
            if len(good_dist) > 0 and len(bad_dist) > 0:
                p10_bad = _percentile(bad_dist, 10.0)
                p90_good = _percentile(good_dist, 90.0)

                # Draw two short bars for percentile landmarks on current epoch violin.
                bar_half_width = 0.22
                ax.hlines(
                    y=p90_good,
                    xmin=x_pos - bar_half_width,
                    xmax=x_pos + bar_half_width,
                    colors="tab:green",
                    linewidth=2.8,
                    alpha=0.95,
                )
                ax.hlines(
                    y=p10_bad,
                    xmin=x_pos - bar_half_width,
                    xmax=x_pos + bar_half_width,
                    colors="tab:red",
                    linewidth=2.8,
                    alpha=0.95,
                )

                if p10_bad == p10_bad and p90_good == p90_good:
                    gap_text = f"dp={p10_bad - p90_good:.2f}"
                else:
                    gap_text = "dp=nan"
            else:
                gap_text = "dp=nan"

            ax.text(
                x_pos,
                local_max + text_offset,
                f"{pearson_text}\n{gap_text}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    ax.plot([], [], color="tab:green", linewidth=8, alpha=0.45, label="Good NLL")
    ax.plot([], [], color="tab:red", linewidth=8, alpha=0.45, label="Bad NLL")
    ax.plot([], [], color="tab:green", linewidth=2.8, label="p90(good)")
    ax.plot([], [], color="tab:red", linewidth=2.8, label="p10(bad)")
    ax.legend(loc="upper right")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Per-sequence NLL")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(e) for e in epochs], rotation=0)
    ax.grid(alpha=0.3)


def save_periodic_violin_history_figure(
    history: Dict[str, List[Any]],
    output_path: Path,
    every_n_epochs: int,
    dataset_descriptions: Dict[str, str] | None = None,
) -> None:
    """Save the cumulative 4-panel violin figure using epochs sampled every N.

    Inputs:
    - history: Metric dictionary containing epoch and per-split NLL distributions.
    - output_path: Destination path of the generated PNG.
    - every_n_epochs: Sampling period for epochs shown on x-axis.

    Output:
    - None. Writes the figure to disk.
    """
    if every_n_epochs <= 0:
        raise ValueError("every_n_epochs must be > 0")

    sampled_indices: List[int] = []
    sampled_epochs: List[int] = []
    for idx, epoch in enumerate(history["epoch"]):
        if epoch % every_n_epochs == 0:
            sampled_indices.append(idx)
            sampled_epochs.append(int(epoch))

    all_panel_specs = [
        (
            "train_good_seq_nll",
            "train_bad_seq_nll",
            "train_dist_nll_corr",
            _with_dataset_description("NLL distributions vs Epoch", "train", dataset_descriptions),
        ),
        (
            "val_good_seq_nll",
            "val_bad_seq_nll",
            "val_dist_nll_corr",
            _with_dataset_description("NLL distributions vs Epoch", "val", dataset_descriptions),
        ),
        (
            "vae_good_seq_nll",
            "vae_bad_seq_nll",
            "vae_dist_nll_corr",
            _with_dataset_description("NLL distributions vs Epoch", "val_1", dataset_descriptions),
        ),
        (
            "dist2530_good_seq_nll",
            "dist2530_bad_seq_nll",
            "dist2530_dist_nll_corr",
            _with_dataset_description(
                "NLL distributions vs Epoch", "val_1", dataset_descriptions,
            ),
        ),
    ]

    panel_specs = []
    for good_key, bad_key, corr_key, title in all_panel_specs:
        has_any_data = any(
            (len(history[good_key][i]) > 0 or len(history[bad_key][i]) > 0)
            for i in sampled_indices
        )
        if has_any_data:
            panel_specs.append((good_key, bad_key, corr_key, title))

    if len(panel_specs) == 0:
        return

    fig, axes = plt.subplots(len(panel_specs), 1, figsize=(22, 5.5 * len(panel_specs)), sharex=True)
    if len(panel_specs) == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes)

    for ax, (good_key, bad_key, corr_key, title) in zip(axes_list, panel_specs):
        good_distributions = [history[good_key][i] for i in sampled_indices]
        bad_distributions = [history[bad_key][i] for i in sampled_indices]
        pearsons = [history[corr_key][i] for i in sampled_indices]
        _plot_violin_history_panel(
            ax=ax,
            epochs=sampled_epochs,
            good_distributions=good_distributions,
            bad_distributions=bad_distributions,
            pearsons=pearsons,
            title=title,
        )

    fig.suptitle(
        (
            f"Per-epoch NLL violins (sampled every {every_n_epochs} epochs)\n"
            "Annotations per epoch:\n r = Pearson(distance from reference, NLL);\n "
            "dp (p10(bad)-p90(good)) = separation margin between lower bad tail and upper good tail; "
        ),
        fontsize=20,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
