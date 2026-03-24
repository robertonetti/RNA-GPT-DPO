from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from torch.utils.data import DataLoader

from src.dpo_config import Config
from src.dpo_logging import print_eval_summary
from src.dpo_metrics import (
    compute_full_dataset_dpo_loss_from_loader,
    compute_full_dataset_reint_loss_from_loader,
    compute_full_dataset_nll,
    compute_mean_token_likelihood,
    compute_pearson_correlation,
    compute_random_batch_nll,
    compute_val_separation_correlation,
)
from src.dpo_plotting import (
    save_distance_binned_correlation_figure,
    save_distance_nll_scatter_figure,
    save_epoch_figures,
    save_periodic_violin_history_figure,
)


def _distance_to_reference(seq: str, reference_seq: str) -> float:
    """Return an edit-like mismatch count against the fixed reference sequence."""
    overlap = min(len(seq), len(reference_seq))
    mismatches = sum(1 for a, b in zip(seq[:overlap], reference_seq[:overlap]) if a != b)
    return float(mismatches + abs(len(seq) - len(reference_seq)))


def _compute_distance_nll_correlation(
    good_distances: torch.Tensor,
    bad_distances: torch.Tensor,
    good_nll: List[float],
    bad_nll: List[float],
) -> float:
    """Compute Pearson(distance, NLL) over good+bad sequences of one split."""
    if len(good_nll) == 0 and len(bad_nll) == 0:
        return float("nan")

    nll_values = torch.tensor(good_nll + bad_nll, dtype=torch.float32)
    distance_values = torch.cat([good_distances, bad_distances], dim=0)
    return compute_pearson_correlation(distance_values, nll_values)


def _has_path(path_value: str | None) -> bool:
    """Return True if a config path is provided (non-empty string)."""
    return path_value is not None and path_value.strip() != ""


def _append_history_row(history: Dict[str, List[Any]], row: Dict[str, Any]) -> None:
    """Append one epoch row to every metric list in history."""
    for key in history:
        history[key].append(row[key])


def _save_eval_artifacts(
    history: Dict[str, List[Any]],
    image_dir: Path,
    epoch: int,
    dataset_descriptions: Dict[str, str],
    eval_period: int,
    train_good_distances: torch.Tensor,
    train_bad_distances: torch.Tensor,
    val_good_distances: torch.Tensor,
    val_bad_distances: torch.Tensor,
    build_distance_binned_entries: Callable[..., List[Dict[str, Any]]],
    distance_nll_ylim_min: float,
    distance_nll_ylim_max: float,
) -> None:
    """Save all figures generated at one evaluation step."""
    save_epoch_figures(
        history,
        main_path=image_dir / f"epoch_{epoch:03d}_main.png",
        dataset_descriptions=dataset_descriptions,
    )
    save_periodic_violin_history_figure(
        history=history,
        output_path=image_dir / f"epoch_{epoch:03d}_violin_history.png",
        every_n_epochs=eval_period,
        dataset_descriptions=dataset_descriptions,
    )
    save_distance_binned_correlation_figure(
        output_path=image_dir / f"epoch_{epoch:03d}_distance_bin_corr.png",
        epoch=epoch,
        dataset_entries=build_distance_binned_entries(
            train_good_nll=history["train_good_seq_nll"][-1],
            train_bad_nll=history["train_bad_seq_nll"][-1],
            val_good_nll=history["val_good_seq_nll"][-1],
            val_bad_nll=history["val_bad_seq_nll"][-1],
            val_1_good_nll=history["val_1_good_seq_nll"][-1],
            val_1_bad_nll=history["val_1_bad_seq_nll"][-1],
            val_2_good_nll=history["val_2_good_seq_nll"][-1],
            val_2_bad_nll=history["val_2_bad_seq_nll"][-1],
        ),
    )
    save_distance_nll_scatter_figure(
        output_path=image_dir / f"epoch_{epoch:03d}_distance_nll_scatter.png",
        epoch=epoch,
        train_title=f"train | {dataset_descriptions.get('train', '').strip()}".strip(" |"),
        train_good_distances=train_good_distances.tolist(),
        train_bad_distances=train_bad_distances.tolist(),
        train_good_nll=history["train_good_seq_nll"][-1],
        train_bad_nll=history["train_bad_seq_nll"][-1],
        val_title=f"val | {dataset_descriptions.get('val', '').strip()}".strip(" |"),
        val_good_distances=val_good_distances.tolist(),
        val_bad_distances=val_bad_distances.tolist(),
        val_good_nll=history["val_good_seq_nll"][-1],
        val_bad_nll=history["val_bad_seq_nll"][-1],
        ylim_min=distance_nll_ylim_min,
        ylim_max=distance_nll_ylim_max,
    )


def _print_eval_block(
    title: str,
    epoch_label: str,
    cfg: Config,
    metrics: Dict[str, Any],
) -> None:
    """Print one eval summary block using a metrics dict."""
    print_eval_summary(
        title=title,
        epoch_label=epoch_label,
        dpo_train_batch_epoch=metrics["dpo_train_batch_epoch"],
        dpo_train_full=metrics["dpo_train_full"],
        dpo_val_full=metrics["dpo_val_full"],
        nll_train_good_rand=metrics["nll_train_good_rand"],
        nll_train_bad_rand=metrics["nll_train_bad_rand"],
        nll_val_good_rand=metrics["nll_val_good_rand"],
        nll_val_bad_rand=metrics["nll_val_bad_rand"],
        nll_margin_train_rand=metrics["nll_margin_train_rand"],
        nll_margin_val_rand=metrics["nll_margin_val_rand"],
        mean_like_model=metrics["mean_like_model"],
        mean_like_ref=metrics["mean_like_ref"],
        compute_val_separation_metric=cfg.compute_val_separation_metric,
        val_sep_corr=metrics["val_sep_corr"],
        val_1_sep_corr=metrics["val_1_sep_corr"],
        val_2_sep_corr=metrics["val_2_sep_corr"],
        compute_full_nll_metrics=cfg.compute_full_nll_metrics,
        nll_train_good_full=metrics["nll_train_good_full"],
        nll_val_good_full=metrics["nll_val_good_full"],
        nll_val_bad_full=metrics["nll_val_bad_full"],
    )


def _evaluate_model_state(
    model,
    ref_model,
    train_eval_loader: DataLoader,
    val_loader: DataLoader | None,
    pad_token: int,
    device: torch.device,
    cfg: Config,
    dpo_dataset: Dict[str, Any],
    val_dpo_dataset: Dict[str, Any] | None,
    val_enabled: bool,
    val_1_enabled: bool,
    val_2_enabled: bool,
    val_1_good_data: List[torch.Tensor],
    val_1_bad_data: List[torch.Tensor],
    val_2_good_data: List[torch.Tensor],
    val_2_bad_data: List[torch.Tensor],
    dn_eval_data: List[torch.Tensor],
    train_good_distances: torch.Tensor,
    train_bad_distances: torch.Tensor,
    val_good_distances: torch.Tensor,
    val_bad_distances: torch.Tensor,
    val_1_good_distances: torch.Tensor,
    val_1_bad_distances: torch.Tensor,
    val_2_good_distances: torch.Tensor,
    val_2_bad_distances: torch.Tensor,
) -> Dict[str, Any]:
    """Compute all evaluation metrics for one model state."""
    if cfg.reint:
        dpo_train_full = compute_full_dataset_reint_loss_from_loader(
            model,
            ref_model,
            train_eval_loader,
            pad_token,
            device,
            beta=cfg.beta,
            lambda_reint=cfg.lambda_reint,
        )
        dpo_val_full = (
            compute_full_dataset_reint_loss_from_loader(
                model,
                ref_model,
                val_loader,
                pad_token,
                device,
                beta=cfg.beta,
                lambda_reint=cfg.lambda_reint,
            )
            if val_loader is not None
            else float("nan")
        )
    else:
        dpo_train_full = compute_full_dataset_dpo_loss_from_loader(
            model, ref_model, train_eval_loader, pad_token, device, beta=cfg.beta
        )
        dpo_val_full = (
            compute_full_dataset_dpo_loss_from_loader(
                model, ref_model, val_loader, pad_token, device, beta=cfg.beta
            )
            if val_loader is not None
            else float("nan")
        )

    eval_bs = min(cfg.batch_size, cfg.nll_rand_eval_batch_size_cap)
    nll_train_good_rand = compute_random_batch_nll(
        model,
        dpo_dataset["good_data"],
        pad_token,
        device,
        n_batches=cfg.nll_rand_n_batches,
        batch_size=eval_bs,
    )
    nll_train_bad_rand = compute_random_batch_nll(
        model,
        dpo_dataset["bad_data"],
        pad_token,
        device,
        n_batches=cfg.nll_rand_n_batches,
        batch_size=eval_bs,
    )
    if val_enabled:
        nll_val_good_rand = compute_random_batch_nll(
            model,
            val_dpo_dataset["good_data"],
            pad_token,
            device,
            n_batches=cfg.nll_rand_n_batches,
            batch_size=eval_bs,
        )
        nll_val_bad_rand = compute_random_batch_nll(
            model,
            val_dpo_dataset["bad_data"],
            pad_token,
            device,
            n_batches=cfg.nll_rand_n_batches,
            batch_size=eval_bs,
        )
    else:
        nll_val_good_rand = float("nan")
        nll_val_bad_rand = float("nan")
    nll_margin_train_rand = nll_train_bad_rand - nll_train_good_rand
    nll_margin_val_rand = nll_val_bad_rand - nll_val_good_rand

    if cfg.compute_full_nll_metrics:
        nll_train_good_full = compute_full_dataset_nll(
            model, dpo_dataset["good_data"], pad_token, device, batch_size=cfg.batch_size
        )
        if val_enabled:
            nll_val_good_full = compute_full_dataset_nll(
                model, val_dpo_dataset["good_data"], pad_token, device, batch_size=cfg.batch_size
            )
            nll_val_bad_full = compute_full_dataset_nll(
                model, val_dpo_dataset["bad_data"], pad_token, device, batch_size=cfg.batch_size
            )
        else:
            nll_val_good_full = float("nan")
            nll_val_bad_full = float("nan")
    else:
        nll_train_good_full = float("nan")
        nll_val_good_full = float("nan")
        nll_val_bad_full = float("nan")

    full_eval_bs = min(cfg.batch_size, cfg.full_eval_batch_size_cap)
    mean_like_model = compute_mean_token_likelihood(
        model, dn_eval_data, pad_token, device, batch_size=full_eval_bs
    )
    mean_like_ref = compute_mean_token_likelihood(
        ref_model, dn_eval_data, pad_token, device, batch_size=full_eval_bs
    )

    if cfg.compute_val_separation_metric:
        _, train_good_seq_nll, train_bad_seq_nll = compute_val_separation_correlation(
            model,
            dpo_dataset["good_data"],
            dpo_dataset["bad_data"],
            pad_token,
            device,
            batch_size=full_eval_bs,
        )

        train_dist_nll_corr = _compute_distance_nll_correlation(
            train_good_distances,
            train_bad_distances,
            train_good_seq_nll,
            train_bad_seq_nll,
        )
        if val_enabled:
            val_sep_corr, val_good_seq_nll, val_bad_seq_nll = compute_val_separation_correlation(
                model,
                val_dpo_dataset["good_data"],
                val_dpo_dataset["bad_data"],
                pad_token,
                device,
                batch_size=full_eval_bs,
            )
            val_dist_nll_corr = _compute_distance_nll_correlation(
                val_good_distances,
                val_bad_distances,
                val_good_seq_nll,
                val_bad_seq_nll,
            )
        else:
            val_sep_corr = float("nan")
            val_good_seq_nll = []
            val_bad_seq_nll = []
            val_dist_nll_corr = float("nan")

        if val_1_enabled:
            val_1_sep_corr, val_1_good_seq_nll, val_1_bad_seq_nll = compute_val_separation_correlation(
                model,
                val_1_good_data,
                val_1_bad_data,
                pad_token,
                device,
                batch_size=full_eval_bs,
            )
            val_1_dist_nll_corr = _compute_distance_nll_correlation(
                val_1_good_distances,
                val_1_bad_distances,
                val_1_good_seq_nll,
                val_1_bad_seq_nll,
            )
        else:
            val_1_sep_corr = float("nan")
            val_1_good_seq_nll = []
            val_1_bad_seq_nll = []
            val_1_dist_nll_corr = float("nan")

        if val_2_enabled:
            val_2_sep_corr, val_2_good_seq_nll, val_2_bad_seq_nll = compute_val_separation_correlation(
                model,
                val_2_good_data,
                val_2_bad_data,
                pad_token,
                device,
                batch_size=full_eval_bs,
            )
            val_2_dist_nll_corr = _compute_distance_nll_correlation(
                val_2_good_distances,
                val_2_bad_distances,
                val_2_good_seq_nll,
                val_2_bad_seq_nll,
            )
        else:
            val_2_sep_corr = float("nan")
            val_2_good_seq_nll = []
            val_2_bad_seq_nll = []
            val_2_dist_nll_corr = float("nan")
    else:
        train_good_seq_nll = []
        train_bad_seq_nll = []
        train_dist_nll_corr = float("nan")
        val_sep_corr = float("nan")
        val_good_seq_nll = []
        val_bad_seq_nll = []
        val_dist_nll_corr = float("nan")
        val_1_sep_corr = float("nan")
        val_1_good_seq_nll = []
        val_1_bad_seq_nll = []
        val_1_dist_nll_corr = float("nan")
        val_2_sep_corr = float("nan")
        val_2_good_seq_nll = []
        val_2_bad_seq_nll = []
        val_2_dist_nll_corr = float("nan")

    return {
        "dpo_train_full": dpo_train_full,
        "dpo_val_full": dpo_val_full,
        "nll_train_good_rand": nll_train_good_rand,
        "nll_train_bad_rand": nll_train_bad_rand,
        "nll_val_good_rand": nll_val_good_rand,
        "nll_val_bad_rand": nll_val_bad_rand,
        "nll_margin_train_rand": nll_margin_train_rand,
        "nll_margin_val_rand": nll_margin_val_rand,
        "nll_train_good_full": nll_train_good_full,
        "nll_val_good_full": nll_val_good_full,
        "nll_val_bad_full": nll_val_bad_full,
        "mean_like_model": mean_like_model,
        "mean_like_ref": mean_like_ref,
        "train_good_seq_nll": train_good_seq_nll,
        "train_bad_seq_nll": train_bad_seq_nll,
        "train_dist_nll_corr": train_dist_nll_corr,
        "val_sep_corr": val_sep_corr,
        "val_good_seq_nll": val_good_seq_nll,
        "val_bad_seq_nll": val_bad_seq_nll,
        "val_dist_nll_corr": val_dist_nll_corr,
        "val_1_sep_corr": val_1_sep_corr,
        "val_1_good_seq_nll": val_1_good_seq_nll,
        "val_1_bad_seq_nll": val_1_bad_seq_nll,
        "val_1_dist_nll_corr": val_1_dist_nll_corr,
        "val_2_sep_corr": val_2_sep_corr,
        "val_2_good_seq_nll": val_2_good_seq_nll,
        "val_2_bad_seq_nll": val_2_bad_seq_nll,
        "val_2_dist_nll_corr": val_2_dist_nll_corr,
    }
