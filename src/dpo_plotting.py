from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Dict, List, Any, Sequence

import matplotlib
# Force a non-interactive backend for batch jobs to avoid Tk cleanup crashes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_trapezoid(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    """Compute trapezoidal area for paired points sorted by x."""
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return float("nan")
    area = 0.0
    for i in range(1, len(x_values)):
        dx = float(x_values[i]) - float(x_values[i - 1])
        area += dx * (float(y_values[i]) + float(y_values[i - 1])) * 0.5
    return float(area)


def _build_binary_labels_and_scores(
    good_nll: Sequence[float],
    bad_nll: Sequence[float],
) -> tuple[List[int], List[float]]:
    """Return labels and scores where high score means likely-good sequence."""
    labels: List[int] = []
    scores: List[float] = []

    for value in good_nll:
        val = float(value)
        if val == val:
            labels.append(1)
            scores.append(-val)
    for value in bad_nll:
        val = float(value)
        if val == val:
            labels.append(0)
            scores.append(-val)

    return labels, scores


def _compute_roc_curve(labels: Sequence[int], scores: Sequence[float]) -> tuple[List[float], List[float], float]:
    """Compute ROC points (FPR/TPR) and AUROC without external dependencies."""
    if len(labels) == 0 or len(labels) != len(scores):
        return [0.0, 1.0], [0.0, 1.0], float("nan")

    positives = sum(1 for y in labels if y == 1)
    negatives = sum(1 for y in labels if y == 0)
    if positives == 0 or negatives == 0:
        return [0.0, 1.0], [0.0, 1.0], float("nan")

    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    fpr: List[float] = [0.0]
    tpr: List[float] = [0.0]

    i = 0
    while i < len(ranked):
        score_i = ranked[i][0]
        while i < len(ranked) and ranked[i][0] == score_i:
            if ranked[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        fpr.append(fp / negatives)
        tpr.append(tp / positives)

    auroc = _safe_trapezoid(fpr, tpr)
    return fpr, tpr, auroc


def _compute_pr_curve(labels: Sequence[int], scores: Sequence[float]) -> tuple[List[float], List[float], float]:
    """Compute PR points (Recall/Precision) and AUPRC without external dependencies."""
    if len(labels) == 0 or len(labels) != len(scores):
        return [0.0, 1.0], [1.0, 0.0], float("nan")

    positives = sum(1 for y in labels if y == 1)
    if positives == 0:
        return [0.0, 1.0], [1.0, 0.0], float("nan")

    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    recall: List[float] = [0.0]
    precision: List[float] = [1.0]

    i = 0
    while i < len(ranked):
        score_i = ranked[i][0]
        while i < len(ranked) and ranked[i][0] == score_i:
            if ranked[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        denom = tp + fp
        recall.append(tp / positives)
        precision.append((tp / denom) if denom > 0 else 1.0)

    if recall[-1] < 1.0:
        recall.append(1.0)
        precision.append(precision[-1])

    auprc = _safe_trapezoid(recall, precision)
    return recall, precision, auprc


def _compute_ppv_curve(labels: Sequence[int], scores: Sequence[float]) -> tuple[List[float], List[float]]:
    """Compute PPV among top-k samples ranked by predicted goodness score."""
    if len(labels) == 0 or len(labels) != len(scores):
        return [], []

    ranked_labels = [label for _, label in sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)]
    total = len(ranked_labels)
    cum_tp = 0
    x_fraction: List[float] = []
    y_ppv: List[float] = []

    for idx, label in enumerate(ranked_labels, start=1):
        if label == 1:
            cum_tp += 1
        x_fraction.append(idx / total)
        y_ppv.append(cum_tp / idx)

    return x_fraction, y_ppv


def _format_curve_panel_title(dataset_title: str, metric_title: str) -> str:
    """Build compact multiline titles for ROC/PRC/PPV panels."""
    return f"{fill(dataset_title, width=62)}\n{metric_title}"


def save_validation_roc_prc_ppv_figure(
    output_path: Path,
    eval_iterations: Sequence[int],
    dataset_entries: Sequence[Dict[str, Any]],
) -> None:
    """Save a multi-row (datasets) x 3-column (ROC, PRC, PPV) cumulative figure."""
    if len(eval_iterations) == 0:
        return

    valid_entries = [entry for entry in dataset_entries if str(entry.get("title", "")).strip() != ""]
    if len(valid_entries) == 0:
        return

    n_rows = len(valid_entries)
    fig, axes = plt.subplots(n_rows, 3, figsize=(21, 5.2 * n_rows), squeeze=False)
    cmap = plt.get_cmap("viridis")

    for row_idx, entry in enumerate(valid_entries):
        title = str(entry.get("title", f"val_{row_idx}"))
        good_nll_history = entry.get("good_nll_history", [])
        bad_nll_history = entry.get("bad_nll_history", [])

        ax_roc = axes[row_idx][0]
        ax_prc = axes[row_idx][1]
        ax_ppv = axes[row_idx][2]

        valid_step_indices: List[int] = []
        for step_idx in range(len(eval_iterations)):
            good_nll = good_nll_history[step_idx] if step_idx < len(good_nll_history) else []
            bad_nll = bad_nll_history[step_idx] if step_idx < len(bad_nll_history) else []
            labels, scores = _build_binary_labels_and_scores(good_nll, bad_nll)
            if len(labels) == 0 or sum(labels) == 0 or sum(1 for y in labels if y == 0) == 0:
                continue
            valid_step_indices.append(step_idx)

        if len(valid_step_indices) == 0:
            for ax in (ax_roc, ax_prc, ax_ppv):
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.grid(alpha=0.25)
            ax_roc.set_title(_format_curve_panel_title(title, "ROC"))
            ax_prc.set_title(_format_curve_panel_title(title, "PRC"))
            ax_ppv.set_title(_format_curve_panel_title(title, "PPV"))
            continue

        ax_roc.plot([0.0, 1.0], [0.0, 1.0], color="gray", linestyle="--", linewidth=1.0)

        latest_auroc = float("nan")
        latest_auprc = float("nan")
        for order, step_idx in enumerate(valid_step_indices):
            frac = order / max(len(valid_step_indices) - 1, 1)
            color = cmap(0.15 + 0.8 * frac)
            alpha = 0.22 + 0.72 * frac
            line_w = 1.0 + 1.1 * frac

            good_nll = good_nll_history[step_idx] if step_idx < len(good_nll_history) else []
            bad_nll = bad_nll_history[step_idx] if step_idx < len(bad_nll_history) else []
            labels, scores = _build_binary_labels_and_scores(good_nll, bad_nll)

            fpr, tpr, auroc = _compute_roc_curve(labels, scores)
            recall, precision, auprc = _compute_pr_curve(labels, scores)
            ppv_x, ppv_y = _compute_ppv_curve(labels, scores)

            latest_auroc = auroc
            latest_auprc = auprc
            baseline = sum(labels) / max(len(labels), 1)

            ax_roc.plot(fpr, tpr, color=color, alpha=alpha, linewidth=line_w)
            ax_prc.plot(recall, precision, color=color, alpha=alpha, linewidth=line_w)
            ax_prc.axhline(baseline, color="gray", linestyle="--", linewidth=0.7, alpha=0.2)
            ax_ppv.plot(ppv_x, ppv_y, color=color, alpha=alpha, linewidth=line_w)

        first_it = int(eval_iterations[valid_step_indices[0]])
        last_it = int(eval_iterations[valid_step_indices[-1]])
        ax_roc.plot([], [], color=cmap(0.15), linewidth=2.0, label=f"early it={first_it}")
        ax_roc.plot([], [], color=cmap(0.95), linewidth=2.0, label=f"late it={last_it}")
        ax_roc.legend(loc="lower right", fontsize=8)

        ax_roc.set_xlim(0.0, 1.0)
        ax_roc.set_ylim(0.0, 1.0)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(_format_curve_panel_title(title, "ROC"))
        ax_roc.grid(alpha=0.25)
        ax_roc.text(
            0.04,
            0.06,
            f"AUROC={latest_auroc:.3f}" if latest_auroc == latest_auroc else "AUROC=nan",
            transform=ax_roc.transAxes,
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

        ax_prc.set_xlim(0.0, 1.0)
        ax_prc.set_ylim(0.0, 1.0)
        ax_prc.set_xlabel("Recall")
        ax_prc.set_ylabel("Precision")
        ax_prc.set_title(_format_curve_panel_title(title, "PRC"))
        ax_prc.grid(alpha=0.25)
        ax_prc.text(
            0.04,
            0.06,
            f"AUPRC={latest_auprc:.3f}" if latest_auprc == latest_auprc else "AUPRC=nan",
            transform=ax_prc.transAxes,
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

        ax_ppv.set_xlim(0.0, 1.0)
        ax_ppv.set_ylim(0.0, 1.0)
        ax_ppv.set_xlabel("Top-k Fraction")
        ax_ppv.set_ylabel("PPV (Precision@k)")
        ax_ppv.set_title(_format_curve_panel_title(title, "PPV"))
        ax_ppv.grid(alpha=0.25)

    fig.suptitle(
        f"Train/Validation NLL as Classifier | cumulative curves up to iteration {int(eval_iterations[-1])}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


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


def _format_main_panel_title(base_title: str, train_desc: str, val_desc: str) -> str:
    """Build compact multi-line titles for main.png panels."""
    if not train_desc and not val_desc:
        return base_title

    train_line = fill(f"Train: {train_desc or '-'}", width=85)
    val_line = fill(f"Val: {val_desc or '-'}", width=85)
    return f"{base_title}\n{train_line}\n{val_line}"


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
    dataset_descriptions: Dict[str, str] | None = None,
) -> None:
    """Generate and save the standard multi-panel training figure.

    Inputs:
    - history: Metric dictionary with per-epoch lists.
        Key size convention: each scalar list has length ``E`` (number of stored epochs).
    - main_path: Output file path for multi-panel epoch curves.

    Output:
    - None. Writes PNG files to disk.
    """
    # X-axis values shared by all time-series panels.
    iterations = history["iteration"]

    # Figure 1: training curves across epochs (2x2 layout).
    fig_main, axes_main = plt.subplots(2, 2, figsize=(20, 12))

    ax_nll = axes_main[0, 0]
    nll_train_good_line = ax_nll.plot(iterations, history["nll_train_good_rand"], marker="o", label="NLL Train Good (rand)")[0]
    nll_train_bad_line = ax_nll.plot(iterations, history["nll_train_bad_rand"], marker="o", label="NLL Train Bad (rand)")[0]
    has_val_nll = any(v == v for v in history["nll_val_good_rand"]) and any(v == v for v in history["nll_val_bad_rand"])
    if has_val_nll:
        nll_val_good_line = ax_nll.plot(iterations, history["nll_val_good_rand"], marker="o", label="NLL Val Good (rand)")[0]
        nll_val_bad_line = ax_nll.plot(iterations, history["nll_val_bad_rand"], marker="o", label="NLL Val Bad (rand)")[0]
    else:
        nll_val_good_line = None
        nll_val_bad_line = None
    train_desc = "" if dataset_descriptions is None else dataset_descriptions.get("train", "").strip()
    val_desc = "" if dataset_descriptions is None else dataset_descriptions.get("val", "").strip()
    nll_title = _format_main_panel_title("NLL Metrics (Random Batches)", train_desc, val_desc)
    ax_nll.set_title(nll_title)
    ax_nll.set_xlabel("Iteration")
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
        iterations,
        history["nll_margin_train_rand"],
        color="black",
        linestyle="--",
        marker="x",
        label="Margin Train (Bad - Good)",
    )[0]
    has_val_margin = any(v == v for v in history["nll_margin_val_rand"])
    if has_val_margin:
        margin_val_line = ax_margin.plot(
            iterations,
            history["nll_margin_val_rand"],
            color="gray",
            linestyle="-.",
            marker="d",
            label="Margin Val (Bad - Good)",
        )[0]
    else:
        margin_val_line = None
    margin_title = _format_main_panel_title("NLL Margin (Bad - Good)", train_desc, val_desc)
    ax_margin.set_title(margin_title)
    ax_margin.set_xlabel("Iteration")
    ax_margin.set_ylabel("Margin")
    ax_margin.grid(alpha=0.3)
    margin_lines = [margin_train_line]
    if margin_val_line is not None:
        margin_lines.append(margin_val_line)
    ax_margin.legend(margin_lines, [line.get_label() for line in margin_lines], loc="best")

    ax_dn = axes_main[1, 0]
    ax_dn.plot(iterations, history["mean_like_model"], marker="o", label="DN mean token likelihood (model)")
    ax_dn.plot(iterations, history["mean_like_ref"], marker="o", linestyle="--", label="DN mean token likelihood (reference)")
    ax_dn.set_title("DN Mean Token Likelihood")
    ax_dn.set_xlabel("Iteration")
    ax_dn.set_ylabel("Likelihood")
    ax_dn.grid(alpha=0.3)
    ax_dn.legend()

    ax_dpo = axes_main[1, 1]
    ax_dpo.plot(iterations, history["dpo_train_batch_epoch"], marker="o", label="DPO Train Batch Mean")
    ax_dpo.plot(iterations, history["dpo_train_full"], marker="o", label="DPO Full Train")
    if any(v == v for v in history["dpo_val_full"]):
        ax_dpo.plot(iterations, history["dpo_val_full"], marker="o", label="DPO Full Val")
    dpo_title = _format_main_panel_title("DPO Loss", train_desc, val_desc)
    ax_dpo.set_title(dpo_title)
    ax_dpo.set_xlabel("Iteration")
    ax_dpo.set_ylabel("Loss")
    ax_dpo.grid(alpha=0.3)
    ax_dpo.legend()

    fig_main.tight_layout()
    # Persist chart and free figure memory immediately.
    fig_main.savefig(main_path, dpi=140, bbox_inches="tight")
    plt.close(fig_main)


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
    every_n_iterations: int,
    dataset_descriptions: Dict[str, str] | None = None,
) -> None:
    """Save the cumulative 4-panel violin figure using iterations sampled every N.

    Inputs:
    - history: Metric dictionary containing iteration and per-split NLL distributions.
    - output_path: Destination path of the generated PNG.
    - every_n_iterations: Sampling period for iterations shown on x-axis.

    Output:
    - None. Writes the figure to disk.
    """
    if every_n_iterations <= 0:
        raise ValueError("every_n_iterations must be > 0")

    sampled_indices: List[int] = []
    sampled_iterations: List[int] = []
    for idx, iteration in enumerate(history["iteration"]):
        if iteration % every_n_iterations == 0:
            sampled_indices.append(idx)
            sampled_iterations.append(int(iteration))

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
            "val_1_good_seq_nll",
            "val_1_bad_seq_nll",
            "val_1_dist_nll_corr",
            _with_dataset_description("NLL distributions vs Epoch", "val_1", dataset_descriptions),
        ),
        (
            "val_2_good_seq_nll",
            "val_2_bad_seq_nll",
            "val_2_dist_nll_corr",
            _with_dataset_description(
                "NLL distributions vs Epoch", "val_2", dataset_descriptions,
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
            epochs=sampled_iterations,
            good_distributions=good_distributions,
            bad_distributions=bad_distributions,
            pearsons=pearsons,
            title=title,
        )

    fig.suptitle(
        (
            f"Per-iteration NLL violins (sampled every {every_n_iterations} iterations)\n"
            "Annotations per iteration:\n r = Pearson(distance from reference, NLL);\n "
            "dp (p10(bad)-p90(good)) = separation margin between lower bad tail and upper good tail; "
        ),
        fontsize=20,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _pearson_from_lists(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    """Compute Pearson correlation from two numeric sequences."""
    if len(x_values) == 0 or len(y_values) == 0 or len(x_values) != len(y_values):
        return float("nan")
    if len(x_values) < 2:
        return float("nan")

    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)

    cov = 0.0
    x_var = 0.0
    y_var = 0.0
    for x_val, y_val in zip(x_values, y_values):
        dx = x_val - x_mean
        dy = y_val - y_mean
        cov += dx * dy
        x_var += dx * dx
        y_var += dy * dy

    denom = (x_var * y_var) ** 0.5
    if denom == 0.0:
        return float("nan")
    return cov / denom


def _build_fixed_distance_bins() -> List[tuple[int, int]]:
    """Return fixed integer bins: [1-4], [5-8], ..., [57-60]."""
    bins: List[tuple[int, int]] = []
    for start in range(1, 61, 4):
        bins.append((start, min(start + 3, 60)))
    return bins


def _compute_binned_correlations(
    distances: Sequence[float],
    nll_values: Sequence[float],
) -> tuple[List[str], List[float], List[bool], List[int], List[float]]:
    """Return (labels, correlations, is_empty, counts, mean_nll) over fixed bins."""
    fixed_bins = _build_fixed_distance_bins()

    labels: List[str] = []
    correlations: List[float] = []
    empty_flags: List[bool] = []
    counts: List[int] = []
    mean_nll_values: List[float] = []

    for left_i, right_i in fixed_bins:
        left = float(left_i)
        right = float(right_i)

        x_bin: List[float] = []
        y_bin: List[float] = []
        for dist, nll in zip(distances, nll_values):
            in_bin = left <= dist <= right
            if in_bin:
                x_bin.append(dist)
                y_bin.append(nll)

        label = f"{left_i}-{right_i}"
        labels.append(label)

        if len(x_bin) == 0:
            correlations.append(float("nan"))
            empty_flags.append(True)
            counts.append(0)
            mean_nll_values.append(float("nan"))
        else:
            correlations.append(_pearson_from_lists(x_bin, y_bin))
            empty_flags.append(False)
            counts.append(len(x_bin))
            mean_nll_values.append(sum(y_bin) / len(y_bin))

    return labels, correlations, empty_flags, counts, mean_nll_values


def save_distance_binned_correlation_figure(
    output_path: Path,
    iteration: int,
    dataset_entries: Sequence[Dict[str, Any]],
) -> None:
    """Save distance-bin vs Pearson(distance, NLL) bars for each available dataset.

    Each dataset entry must define:
    - title: Panel title.
    - distances: Sequence with good+bad distances from the reference sequence.
    - good_nll: Per-sequence NLL for good set.
    - bad_nll: Per-sequence NLL for bad set.

    Empty bins are marked with a red "x" at y=0.5 as requested.
    """
    valid_entries: List[Dict[str, Any]] = []
    for entry in dataset_entries:
        distances = [float(v) for v in entry.get("distances", [])]
        good_nll = [float(v) for v in entry.get("good_nll", [])]
        bad_nll = [float(v) for v in entry.get("bad_nll", [])]
        nll_values = good_nll + bad_nll

        usable_len = min(len(distances), len(nll_values))
        valid_entries.append(
            {
                "title": str(entry.get("title", "Dataset")),
                "distances": distances[:usable_len],
                "nll_values": nll_values[:usable_len],
            }
        )

    if len(valid_entries) == 0:
        return

    fig, axes = plt.subplots(len(valid_entries), 1, figsize=(22, 4.6 * len(valid_entries)))
    if len(valid_entries) == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes)

    for ax, entry in zip(axes_list, valid_entries):
        distances = entry["distances"]
        nll_values = entry["nll_values"]
        title = entry["title"]

        if len(distances) == 0 or len(nll_values) == 0:
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
            ax.set_ylim(-1.05, 1.05)
            ax.set_title(title)
            ax.set_ylabel("Pearson r")
            ax.grid(alpha=0.3)
            continue

        labels, correlations, empty_flags, counts, mean_nll_values = _compute_binned_correlations(
            distances,
            nll_values,
        )

        x_positions = list(range(len(labels)))
        bar_heights: List[float] = []
        bar_colors: List[str] = []
        bar_hatches: List[str] = []
        for corr, is_empty in zip(correlations, empty_flags):
            if is_empty:
                bar_heights.append(0.0)
                bar_colors.append("white")
                bar_hatches.append("")
            elif corr == corr:
                bar_heights.append(corr)
                bar_colors.append("tab:blue")
                bar_hatches.append("")
            else:
                # Non-empty bin with undefined correlation (e.g. zero distance variance).
                bar_heights.append(0.0)
                bar_colors.append("lightgray")
                bar_hatches.append("//")

        bars = ax.bar(
            x_positions,
            bar_heights,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, hatch in zip(bars, bar_hatches):
            if hatch:
                bar.set_hatch(hatch)

        empty_x = [x for x, is_empty in zip(x_positions, empty_flags) if is_empty]
        if len(empty_x) > 0:
            ax.scatter(
                empty_x,
                [0.5 for _ in empty_x],
                marker="x",
                color="red",
                s=80,
                linewidths=2.0,
                zorder=5,
                label="Empty bin",
            )

        # Add sample count text above each non-empty bar.
        for x_pos, is_empty, n_count, corr in zip(x_positions, empty_flags, counts, correlations):
            if is_empty:
                continue
            if corr == corr and corr > 0.0:
                y_text = min(corr + 0.05, 1.0)
            else:
                y_text = 0.05
            ax.text(
                x_pos,
                y_text,
                f"n={n_count}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )

        # Plot mean NLL per interval on a secondary (right) y-axis.
        ax_right = ax.twinx()
        line_x: List[float] = []
        line_y: List[float] = []
        for x_pos, is_empty, mean_nll in zip(x_positions, empty_flags, mean_nll_values):
            if is_empty:
                continue
            line_x.append(float(x_pos))
            line_y.append(float(mean_nll))

        if len(line_x) > 0:
            ax_right.plot(
                line_x,
                line_y,
                color="green",
                marker="o",
                linewidth=1.6,
                markersize=4,
                label="Mean NLL",
            )
            ax_right.legend(loc="upper left")
        ax_right.set_ylabel("Mean NLL", color="green")
        ax_right.tick_params(axis="y", colors="green")

        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel("Distance bin from reference (DN first sequence)")
        ax.set_ylabel("Pearson r(distance, NLL)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        if len(empty_x) > 0:
            ax.legend(loc="upper right")

    fig.suptitle(
        f"Distance-binned correlation vs NLL (iteration {iteration})",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_distance_nll_scatter_panel(
    ax,
    title: str,
    good_distances: Sequence[float],
    bad_distances: Sequence[float],
    good_nll: Sequence[float],
    bad_nll: Sequence[float],
    ylim_min: float,
    ylim_max: float,
) -> None:
    """Plot one panel with distance-binned NLL violins by class."""
    good_x = [float(v) for v in good_distances]
    bad_x = [float(v) for v in bad_distances]
    good_y = [float(v) for v in good_nll]
    bad_y = [float(v) for v in bad_nll]

    # Keep arrays aligned if lengths differ for safety.
    good_len = min(len(good_x), len(good_y))
    bad_len = min(len(bad_x), len(bad_y))
    good_x = good_x[:good_len]
    good_y = good_y[:good_len]
    bad_x = bad_x[:bad_len]
    bad_y = bad_y[:bad_len]

    has_points = (len(good_y) > 0) or (len(bad_y) > 0)

    bins = _build_fixed_distance_bins()
    labels = [f"{left}-{right}" for left, right in bins]
    positions = [float(i) for i in range(len(bins))]

    good_by_bin: List[List[float]] = [[] for _ in bins]
    bad_by_bin: List[List[float]] = [[] for _ in bins]

    for dist, nll in zip(good_x, good_y):
        for bin_idx, (left, right) in enumerate(bins):
            if float(left) <= dist <= float(right):
                good_by_bin[bin_idx].append(nll)
                break

    for dist, nll in zip(bad_x, bad_y):
        for bin_idx, (left, right) in enumerate(bins):
            if float(left) <= dist <= float(right):
                bad_by_bin[bin_idx].append(nll)
                break

    good_nonempty_idx = [i for i, values in enumerate(good_by_bin) if len(values) > 0]
    bad_nonempty_idx = [i for i, values in enumerate(bad_by_bin) if len(values) > 0]

    if len(bad_nonempty_idx) > 0:
        bad_parts = ax.violinplot(
            dataset=[bad_by_bin[i] for i in bad_nonempty_idx],
            positions=[positions[i] for i in bad_nonempty_idx],
            widths=0.8,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        _style_violin_parts(bad_parts, color="tab:red")

    if len(good_nonempty_idx) > 0:
        good_parts = ax.violinplot(
            dataset=[good_by_bin[i] for i in good_nonempty_idx],
            positions=[positions[i] for i in good_nonempty_idx],
            widths=0.8,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        _style_violin_parts(good_parts, color="tab:green")

    ax.set_title(title)
    ax.set_xlabel("Distance bin from reference")
    ax.set_ylabel("NLL")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(ylim_min, ylim_max)
    ax.grid(axis="y", alpha=0.25)

    if has_points:
        ax.plot([], [], color="tab:red", linewidth=8, alpha=0.45, label="DT-")
        ax.plot([], [], color="tab:green", linewidth=8, alpha=0.45, label="DT+")
        ax.legend(loc="best")
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


def save_distance_nll_scatter_figure(
    output_path: Path,
    iteration: int,
    train_title: str,
    train_good_distances: Sequence[float],
    train_bad_distances: Sequence[float],
    train_good_nll: Sequence[float],
    train_bad_nll: Sequence[float],
    val_title: str,
    val_good_distances: Sequence[float],
    val_bad_distances: Sequence[float],
    val_good_nll: Sequence[float],
    val_bad_nll: Sequence[float],
    ylim_min: float = 0.45,
    ylim_max: float = 3.0,
) -> None:
    """Save a 2-panel figure (train/val) with distance-binned NLL violins."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=False)

    _plot_distance_nll_scatter_panel(
        ax=axes[0],
        title=train_title,
        good_distances=train_good_distances,
        bad_distances=train_bad_distances,
        good_nll=train_good_nll,
        bad_nll=train_bad_nll,
        ylim_min=ylim_min,
        ylim_max=ylim_max,
    )
    _plot_distance_nll_scatter_panel(
        ax=axes[1],
        title=val_title,
        good_distances=val_good_distances,
        bad_distances=val_bad_distances,
        good_nll=val_good_nll,
        bad_nll=val_bad_nll,
        ylim_min=ylim_min,
        ylim_max=ylim_max,
    )

    fig.suptitle(
        f"Distance-binned NLL violins (iteration {iteration})",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
