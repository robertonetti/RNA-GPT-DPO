#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src_clean_analysis.dpo_config import Config, CONFIG, analysis_eval_iterations
from src_clean_analysis.dpo_data import (
    PreferencePairDataset,
    build_labels_from_inputs,
    build_tokenizer,
    encode_sequences,
    has_path,
    load_group_dataset,
    load_preference_dataset,
    read_fasta,
    resolve_path,
)
from src_clean_analysis.dpo_metrics import (
    attach_ref_logprobs,
    compute_batched_kl_loss,
    compute_auroc_from_good_bad_nll,
    compute_full_preference_loss_from_loader,
    compute_mean_token_likelihood,
    compute_preference_loss,
    compute_preference_loss_from_batch_in_chunks,
    compute_dpo_prefactors_from_drift,
    compute_sequence_logprobs,
    compute_sequence_nll,
    mean_distance_to_reference,
    mean_distance_within,
    sample_sequences,
)
from src_clean_analysis.dpo_plotting import (
    save_auroc_figure,
    save_distribution_violin,
    save_main_figure,
    save_violin_history,
)
from src_clean_analysis.transformer import GPTTransformer


def _loss_label(cfg: Config) -> str:
    return "Reint Loss" if cfg.reint else "DPO Loss"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate_full_pair_loss(
    model,
    ref_model,
    loader,
    cfg: Config,
    pad_token: int,
    device: torch.device,
) -> float:
    return compute_full_preference_loss_from_loader(
        model,
        ref_model,
        loader,
        pad_token=pad_token,
        device=device,
        beta=cfg.beta,
        reint=cfg.reint,
        lambda_reint=cfg.lambda_reint,
    )


def _evaluate_random_pair_loss(
    model,
    ref_model,
    pair_dataset: PreferencePairDataset,
    cfg: Config,
    pad_token: int,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    batch = pair_dataset.sample_batch(max(1, cfg.eval_loss_batch_size), generator)
    return compute_preference_loss_from_batch_in_chunks(
        model,
        ref_model,
        batch,
        pad_token=pad_token,
        device=device,
        beta=cfg.beta,
        reint=cfg.reint,
        lambda_reint=cfg.lambda_reint,
        chunk_size=max(1, cfg.batch_size),
    )


def _evaluate_split_nll(
    model,
    dataset: Dict[str, torch.Tensor] | None,
    cfg: Config,
    pad_token: int,
    device: torch.device,
) -> tuple[List[float], List[float], float, float]:
    if dataset is None:
        return [], [], float("nan"), float("nan")

    batch_size = max(1, cfg.metrics_batch_size)
    good_nll = compute_sequence_nll(
        model,
        dataset["good_data"],
        dataset["good_labels"],
        pad_token,
        device,
        batch_size=batch_size,
    )
    bad_nll = compute_sequence_nll(
        model,
        dataset["bad_data"],
        dataset["bad_labels"],
        pad_token,
        device,
        batch_size=batch_size,
    )
    return (
        good_nll.tolist(),
        bad_nll.tolist(),
        float(good_nll.mean().item()) if good_nll.numel() > 0 else float("nan"),
        float(bad_nll.mean().item()) if bad_nll.numel() > 0 else float("nan"),
    )


def _evaluate_model(
    model,
    ref_model,
    train_pairs: PreferencePairDataset,
    val_pairs: PreferencePairDataset | None,
    train_eval_loader,
    val_eval_loader,
    val_1_dataset: Dict[str, torch.Tensor] | None,
    dn_eval_data: torch.Tensor,
    dn_eval_labels: torch.Tensor,
    cfg: Config,
    pad_token: int,
    device: torch.device,
    generator: torch.Generator,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "train_loss": (
            _evaluate_full_pair_loss(
                model,
                ref_model,
                train_eval_loader,
                cfg,
                pad_token,
                device,
            )
            if cfg.full_eval_loss
            else _evaluate_random_pair_loss(
                model,
                ref_model,
                train_pairs,
                cfg,
                pad_token,
                device,
                generator,
            )
        ),
        "val_loss": (
            (
                _evaluate_full_pair_loss(
                    model,
                    ref_model,
                    val_eval_loader,
                    cfg,
                    pad_token,
                    device,
                )
                if cfg.full_eval_loss
                else _evaluate_random_pair_loss(
                    model,
                    ref_model,
                    val_pairs,
                    cfg,
                    pad_token,
                    device,
                    generator,
                )
            )
            if val_pairs is not None and val_eval_loader is not None
            else float("nan")
        ),
    }

    need_sequence_metrics = cfg.full_tracking or cfg.compute_auroc
    if not need_sequence_metrics:
        return metrics

    (
        train_good_seq_nll,
        train_bad_seq_nll,
        train_good_nll_mean,
        train_bad_nll_mean,
    ) = _evaluate_split_nll(model, {
        "good_data": train_pairs.good_data,
        "bad_data": train_pairs.bad_data,
        "good_labels": train_pairs.good_labels,
        "bad_labels": train_pairs.bad_labels,
    }, cfg, pad_token, device)

    (
        val_good_seq_nll,
        val_bad_seq_nll,
        val_good_nll_mean,
        val_bad_nll_mean,
    ) = _evaluate_split_nll(model, None if val_pairs is None else {
        "good_data": val_pairs.good_data,
        "bad_data": val_pairs.bad_data,
        "good_labels": val_pairs.good_labels,
        "bad_labels": val_pairs.bad_labels,
    }, cfg, pad_token, device)

    (
        val_1_good_seq_nll,
        val_1_bad_seq_nll,
        val_1_good_nll_mean,
        val_1_bad_nll_mean,
    ) = _evaluate_split_nll(model, val_1_dataset, cfg, pad_token, device)

    if cfg.compute_auroc:
        metrics["train_auroc"] = compute_auroc_from_good_bad_nll(
            train_good_seq_nll,
            train_bad_seq_nll,
        )
        metrics["val_auroc"] = compute_auroc_from_good_bad_nll(
            val_good_seq_nll,
            val_bad_seq_nll,
        )
        metrics["val_1_auroc"] = compute_auroc_from_good_bad_nll(
            val_1_good_seq_nll,
            val_1_bad_seq_nll,
        )

    if not cfg.full_tracking:
        return metrics

    metrics["train_good_seq_nll"] = train_good_seq_nll
    metrics["train_bad_seq_nll"] = train_bad_seq_nll
    metrics["train_good_nll_mean"] = train_good_nll_mean
    metrics["train_bad_nll_mean"] = train_bad_nll_mean
    metrics["val_good_seq_nll"] = val_good_seq_nll
    metrics["val_bad_seq_nll"] = val_bad_seq_nll
    metrics["val_good_nll_mean"] = val_good_nll_mean
    metrics["val_bad_nll_mean"] = val_bad_nll_mean
    metrics["val_1_good_seq_nll"] = val_1_good_seq_nll
    metrics["val_1_bad_seq_nll"] = val_1_bad_seq_nll
    metrics["val_1_good_nll_mean"] = val_1_good_nll_mean
    metrics["val_1_bad_nll_mean"] = val_1_bad_nll_mean

    metrics["dn_mean_token_likelihood"] = compute_mean_token_likelihood(
        model,
        dn_eval_data,
        dn_eval_labels,
        pad_token,
        device,
        batch_size=max(1, cfg.metrics_batch_size),
    )

    return metrics


ANALYSIS_KEYS = (
    "prefactor_mean",
    "prefactor_median",
    "prefactor_p99",
    "prefactor_train_auroc",
    "sample_pairwise_dist",
    "sample_pairwise_dist_mean",
    "sample_pairwise_dist_std",
    "sample_to_positive_dist",
    "sample_to_positive_dist_mean",
    "sample_to_positive_dist_std",
)


def _run_drift_analysis(
    model,
    train_pairs: PreferencePairDataset,
    cfg: Config,
    pad_token: int,
    device: torch.device,
    iteration: int,
    drift_dir: Path,
) -> Dict[str, Any]:
    """Per-sequence drift on the whole train set, saved to disk, and DPO prefactor of every pair.

    For every train sequence i:   d_i = log pi_theta(seq_i) - log pi_ref(seq_i)
    For every train pair (w, l):  sigma_wl = sigma(-beta * (d_w - d_l))

    The drift of all good and bad sequences is saved in one file per evaluation
    iteration, so the prefactor of any pair can be rebuilt offline.
    """
    batch_size = max(1, cfg.analysis_logp_batch_size)
    logp_good = compute_sequence_logprobs(
        model, train_pairs.good_data, train_pairs.good_labels, pad_token, device, batch_size
    )
    logp_bad = compute_sequence_logprobs(
        model, train_pairs.bad_data, train_pairs.bad_labels, pad_token, device, batch_size
    )
    d_good = logp_good - train_pairs.ref_good_logps
    d_bad = logp_bad - train_pairs.ref_bad_logps

    np.savez(
        drift_dir / f"drift_iter{iteration}.npz",
        iteration=iteration,
        beta=cfg.beta,
        d_good=d_good.numpy(),
        d_bad=d_bad.numpy(),
        logp_good=logp_good.numpy(),
        logp_bad=logp_bad.numpy(),
    )

    prefactor = compute_dpo_prefactors_from_drift(
        d_good, d_bad, train_pairs.pair_good_idx, train_pairs.pair_bad_idx, cfg.beta
    )
    # Train AUROC: how well log pi_theta separates good from bad train sequences.
    train_auroc = compute_auroc_from_good_bad_nll((-logp_good).tolist(), (-logp_bad).tolist())

    return {
        # Full distribution (one value per pair): kept in memory for plotting only,
        # never written to history.json (millions of values).
        "prefactor_all": prefactor.numpy().astype(np.float16),
        "prefactor_mean": float(prefactor.mean()),
        "prefactor_median": float(prefactor.median()),
        "prefactor_p99": float(np.percentile(prefactor.numpy(), 99)),
        "prefactor_train_auroc": train_auroc,
    }


def _run_analysis(
    model,
    positive_data: torch.Tensor,
    cfg: Config,
    start_token: int,
    vocab_size: int,
    device: torch.device,
    iteration: int,
) -> Dict[str, Any]:
    """Diversity/distance of model samples."""
    metrics: Dict[str, Any] = {}

    generator = torch.Generator(device=device).manual_seed(cfg.seed + iteration)
    samples = sample_sequences(
        model,
        n_samples=cfg.analysis_n_samples,
        seq_len=positive_data.size(1),
        start_token=start_token,
        device=device,
        batch_size=max(1, cfg.analysis_sample_batch_size),
        temperature=cfg.analysis_temperature,
        generator=generator,
    )

    within = mean_distance_within(samples, vocab_size, device)
    to_positive = mean_distance_to_reference(samples, positive_data, vocab_size, device)
    metrics["sample_pairwise_dist"] = within.tolist()
    metrics["sample_pairwise_dist_mean"] = float(within.mean())
    metrics["sample_pairwise_dist_std"] = float(within.std())
    metrics["sample_to_positive_dist"] = to_positive.tolist()
    metrics["sample_to_positive_dist_mean"] = float(to_positive.mean())
    metrics["sample_to_positive_dist_std"] = float(to_positive.std())
    return metrics


def _init_history(full_tracking: bool, compute_auroc: bool) -> Dict[str, List]:
    history: Dict[str, List] = {
        "iteration": [],
        "train_loss": [],
        "val_loss": [],
    }
    history.update({key: [] for key in ANALYSIS_KEYS})
    if compute_auroc:
        history.update(
            {
                "train_auroc": [],
                "val_auroc": [],
                "val_1_auroc": [],
            }
        )
    if not full_tracking:
        return history

    history.update(
        {
            "train_good_nll_mean": [],
            "train_bad_nll_mean": [],
            "val_good_nll_mean": [],
            "val_bad_nll_mean": [],
            "val_1_good_nll_mean": [],
            "val_1_bad_nll_mean": [],
            "dn_mean_token_likelihood": [],
            "train_good_seq_nll": [],
            "train_bad_seq_nll": [],
            "val_good_seq_nll": [],
            "val_bad_seq_nll": [],
            "val_1_good_seq_nll": [],
            "val_1_bad_seq_nll": [],
        }
    )
    return history


def _append_history(history: Dict[str, List], iteration: int, metrics: Dict[str, Any]) -> None:
    history["iteration"].append(iteration)
    history["train_loss"].append(metrics["train_loss"])
    history["val_loss"].append(metrics["val_loss"])
    for key in ANALYSIS_KEYS:
        history[key].append(metrics[key])

    if "train_auroc" in history:
        history["train_auroc"].append(metrics["train_auroc"])
        history["val_auroc"].append(metrics["val_auroc"])
        history["val_1_auroc"].append(metrics["val_1_auroc"])

    if "train_good_nll_mean" not in history:
        return

    for key in (
        "train_good_nll_mean",
        "train_bad_nll_mean",
        "val_good_nll_mean",
        "val_bad_nll_mean",
        "val_1_good_nll_mean",
        "val_1_bad_nll_mean",
        "dn_mean_token_likelihood",
        "train_good_seq_nll",
        "train_bad_seq_nll",
        "val_good_seq_nll",
        "val_bad_seq_nll",
        "val_1_good_seq_nll",
        "val_1_bad_seq_nll",
    ):
        history[key].append(metrics[key])


def _save_artifacts(
    history: Dict[str, List],
    image_dir: Path,
    history_json_path: Path,
    cfg: Config,
    prefactor_history: List[np.ndarray],
    n_train_pairs: int,
) -> None:
    save_main_figure(
        history,
        image_dir / "main.png",
        loss_label=_loss_label(cfg),
        full_tracking=cfg.full_tracking,
        full_eval_loss=cfg.full_eval_loss,
    )
    if cfg.compute_auroc:
        save_auroc_figure(history, image_dir / "auroc_history.png")
    if cfg.full_tracking:
        save_violin_history(history, image_dir / "violin_history.png")
    iterations = history["iteration"]
    if not cfg.reint:
        save_distribution_violin(
            iterations,
            prefactor_history,
            image_dir / "prefactor_violin.png",
            title=f"DPO prefactor sigma(beta*(r_l - r_w)) on all {n_train_pairs} train pairs (beta={cfg.beta})",
            ylabel="Prefactor",
            show_p99=True,
            top_labels=[f"AUC={value:.3f}" for value in history["prefactor_train_auroc"]],
            color="tab:purple",
            max_points=cfg.analysis_max_violins,
        )
    save_distribution_violin(
        iterations,
        history["sample_pairwise_dist"],
        image_dir / "sample_pairwise_dist_violin.png",
        title=f"Mean divergence within {cfg.analysis_n_samples} model samples",
        ylabel="Divergence (Hamming / length)",
        color="tab:blue",
        max_points=cfg.analysis_max_violins,
    )
    save_distribution_violin(
        iterations,
        history["sample_to_positive_dist"],
        image_dir / "sample_to_positive_dist_violin.png",
        title="Mean divergence between model samples and DT+ sequences",
        ylabel="Divergence (Hamming / length)",
        color="tab:green",
        max_points=cfg.analysis_max_violins,
    )
    history_json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def main(cfg: Config) -> None:
    project_dir = Path(__file__).resolve().parent.parent

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = _device()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if cfg.suppress_dynamo_errors:
        torch._dynamo.config.suppress_errors = True

    dn_path = resolve_path(project_dir, cfg.dn_path)
    ckpt_path = resolve_path(project_dir, cfg.pretrained_ckpt_path)
    if dn_path is None or ckpt_path is None:
        raise ValueError("Both dn_path and pretrained_ckpt_path must be provided.")

    stoi, pad_token = build_tokenizer(dn_path)
    vocab_size = len(stoi)

    def encode(sequence: str) -> List[int]:
        return [stoi[char] for char in sequence]

    model = GPTTransformer(
        vocab_size=vocab_size,
        embed_dim=cfg.n_embd,
        num_heads=cfg.n_head,
        num_layers=cfg.n_layer,
        dropout_p=0.1,
        mlp_ratio=4,
        pad_id=pad_token,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device)

    if cfg.layers_to_freeze > 0:
        for parameter in model.token_emb.parameters():
            parameter.requires_grad = False
        for block in model.layers[: cfg.layers_to_freeze]:
            for parameter in block.parameters():
                parameter.requires_grad = False

    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad = False

    if cfg.use_torch_compile:
        try:
            model = torch.compile(model)
            ref_model = torch.compile(ref_model)
        except Exception as exc:
            print(f"torch.compile disabled: {exc}")

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler_step_size,
        gamma=cfg.scheduler_gamma,
    )

    train_good_path = resolve_path(project_dir, cfg.train_good_fasta_path)
    train_bad_path = resolve_path(project_dir, cfg.train_bad_fasta_path)
    train_csv_path = resolve_path(project_dir, cfg.train_csv_mapping_path)
    val_good_path = resolve_path(project_dir, cfg.val_good_fasta_path)
    val_bad_path = resolve_path(project_dir, cfg.val_bad_fasta_path)
    val_csv_path = resolve_path(project_dir, cfg.val_csv_mapping_path)
    val_1_good_path = resolve_path(project_dir, cfg.val_1_good_fasta_path)
    val_1_bad_path = resolve_path(project_dir, cfg.val_1_bad_fasta_path)

    if train_good_path is None or train_bad_path is None or train_csv_path is None:
        raise ValueError("Train good/bad FASTA paths and train CSV mapping are required.")

    train_dataset = load_preference_dataset(
        train_good_path,
        train_bad_path,
        train_csv_path,
        cfg.block_size,
        pad_token,
        encode,
    )

    val_enabled = has_path(cfg.val_good_fasta_path) and has_path(cfg.val_bad_fasta_path) and has_path(cfg.val_csv_mapping_path)
    val_dataset = (
        load_preference_dataset(
            val_good_path,
            val_bad_path,
            val_csv_path,
            cfg.block_size,
            pad_token,
            encode,
        )
        if val_enabled and val_good_path is not None and val_bad_path is not None and val_csv_path is not None
        else None
    )

    val_1_enabled = has_path(cfg.val_1_good_fasta_path) and has_path(cfg.val_1_bad_fasta_path)
    val_1_dataset = (
        load_group_dataset(
            val_1_good_path,
            val_1_bad_path,
            cfg.block_size,
            pad_token,
            encode,
        )
        if val_1_enabled and val_1_good_path is not None and val_1_bad_path is not None
        else None
    )

    ref_batch_size = max(1, cfg.metrics_batch_size)
    if not cfg.reint:
        attach_ref_logprobs(train_dataset, ref_model, pad_token, device, ref_batch_size)
        attach_ref_logprobs(val_dataset, ref_model, pad_token, device, ref_batch_size)

    train_pairs = PreferencePairDataset(train_dataset)
    val_pairs = PreferencePairDataset(val_dataset) if val_dataset is not None else None

    train_loader = DataLoader(
        train_pairs,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=train_pairs.collate_fn,
    )
    train_eval_loader = DataLoader(
        train_pairs,
        batch_size=max(1, cfg.eval_pair_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=train_pairs.collate_fn,
    )
    val_eval_loader = (
        DataLoader(
            val_pairs,
            batch_size=max(1, cfg.eval_pair_batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=val_pairs.collate_fn,
        )
        if val_pairs is not None
        else None
    )

    dn_sequences = read_fasta(dn_path)
    if not dn_sequences:
        raise ValueError(f"DN dataset is empty: {dn_path}")
    dn_eval_sequences = dn_sequences
    if cfg.dn_eval_subset_size is not None:
        subset_size = min(cfg.dn_eval_subset_size, len(dn_sequences))
        rng = random.Random(cfg.seed)
        picked = rng.sample(range(len(dn_sequences)), subset_size)
        dn_eval_sequences = [dn_sequences[index] for index in picked]
    dn_eval_data = encode_sequences(dn_eval_sequences, cfg.block_size, pad_token, encode)
    dn_eval_labels = build_labels_from_inputs(dn_eval_data, pad_token)

    checkpoint_dir = resolve_path(project_dir, cfg.checkpoint_dir)
    image_dir = resolve_path(project_dir, cfg.image_dir)
    history_json_path = resolve_path(project_dir, cfg.history_json_path)
    if checkpoint_dir is None or image_dir is None or history_json_path is None:
        raise ValueError("checkpoint_dir, image_dir and history_json_path are required.")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    history_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-sequence drift files (one per evaluation iteration) live next to the images.
    # The reference log-probs and the pair list are saved once, so that the
    # prefactor of every pair can be rebuilt offline from the drift files.
    drift_dir = image_dir / "sequence_drift"
    drift_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.reint:
        np.savez(
            drift_dir / "reference_and_pairs.npz",
            ref_logp_good=train_pairs.ref_good_logps.numpy(),
            ref_logp_bad=train_pairs.ref_bad_logps.numpy(),
            pair_good_idx=train_pairs.pair_good_idx.numpy(),
            pair_bad_idx=train_pairs.pair_bad_idx.numpy(),
            beta=cfg.beta,
        )
    # Full prefactor distribution at each evaluation iteration (float16, plotting only).
    prefactor_history: List[np.ndarray] = []

    positive_path = resolve_path(
        project_dir,
        cfg.analysis_positive_fasta_path if has_path(cfg.analysis_positive_fasta_path) else cfg.train_good_fasta_path,
    )
    positive_sequences = read_fasta(positive_path)
    if len({len(sequence) for sequence in positive_sequences}) != 1:
        raise ValueError(f"Positive sequences must be aligned (same length): {positive_path}")
    positive_data = torch.tensor([encode(sequence) for sequence in positive_sequences], dtype=torch.long)
    start_token = stoi["\n"]

    def evaluate(iteration: int) -> Dict[str, Any]:
        metrics = _evaluate_model(
            model,
            ref_model,
            train_pairs,
            val_pairs,
            train_eval_loader,
            val_eval_loader,
            val_1_dataset,
            dn_eval_data,
            dn_eval_labels,
            cfg,
            pad_token,
            device,
            eval_generator,
        )
        metrics.update(
            _run_analysis(
                model,
                positive_data,
                cfg,
                start_token,
                vocab_size,
                device,
                iteration,
            )
        )
        if cfg.reint:
            # Reint has no reference model in the loss: no drift / prefactor.
            metrics.update({"prefactor_mean": float("nan"), "prefactor_median": float("nan"),
                            "prefactor_p99": float("nan"), "prefactor_train_auroc": float("nan")})
        else:
            drift_metrics = _run_drift_analysis(
                model, train_pairs, cfg, pad_token, device, iteration, drift_dir
            )
            prefactor_history.append(drift_metrics.pop("prefactor_all"))
            metrics.update(drift_metrics)
        return metrics

    eval_iterations = set(analysis_eval_iterations(cfg.max_iterations))

    history = _init_history(cfg.full_tracking, cfg.compute_auroc)
    eval_generator = torch.Generator().manual_seed(cfg.seed)
    use_kl_loss = cfg.reint and float(cfg.lambda_kl) != 0.0

    model.eval()
    baseline = evaluate(0)
    _append_history(history, 0, baseline)
    _save_artifacts(history, image_dir, history_json_path, cfg, prefactor_history, len(train_pairs))

    progress_bar = tqdm(total=cfg.max_iterations, desc="Iterations")
    global_iteration = 0
    epoch = 0

    while global_iteration < cfg.max_iterations:
        epoch += 1
        completed_epoch = True
        for batch in train_loader:
            if global_iteration >= cfg.max_iterations:
                completed_epoch = False
                break

            if len(batch) == 6:
                x_w, x_l, y_w, y_l, ref_logps_w, ref_logps_l = batch
                ref_logps_w = ref_logps_w.to(device)
                ref_logps_l = ref_logps_l.to(device)
            else:
                x_w, x_l, y_w, y_l = batch
                ref_logps_w = None
                ref_logps_l = None

            x_w = x_w.to(device)
            x_l = x_l.to(device)
            y_w = y_w.to(device)
            y_l = y_l.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = compute_preference_loss(
                model,
                ref_model,
                x_w,
                y_w,
                x_l,
                y_l,
                pad_token=pad_token,
                beta=cfg.beta,
                reint=cfg.reint,
                lambda_reint=cfg.lambda_reint,
                ref_logps_w=ref_logps_w,
                ref_logps_l=ref_logps_l,
            )

            if use_kl_loss:
                x_l_unique = torch.unique(x_l, dim=0)
                y_l_unique = build_labels_from_inputs(x_l_unique, pad_token)
                x_kl = torch.cat([x_w, x_l_unique], dim=0)
                y_kl = torch.cat([y_w, y_l_unique], dim=0)
                loss = loss + cfg.lambda_kl * compute_batched_kl_loss(
                    model,
                    ref_model,
                    x_kl,
                    y_kl,
                    pad_token,
                    batch_size=max(1, cfg.metrics_batch_size),
                )

            loss.backward()
            optimizer.step()

            global_iteration += 1
            progress_bar.update(1)

            if global_iteration not in eval_iterations:
                continue

            model.eval()
            metrics = evaluate(global_iteration)
            _append_history(history, global_iteration, metrics)
            _save_artifacts(history, image_dir, history_json_path, cfg, prefactor_history, len(train_pairs))

            checkpoint_path = checkpoint_dir / f"model_iter{global_iteration}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"iter={global_iteration} epoch={epoch} "
                f"train={metrics['train_loss']:.6f} val={metrics['val_loss']:.6f} "
                f"within={metrics['sample_pairwise_dist_mean']:.3f}+-{metrics['sample_pairwise_dist_std']:.3f} "
                f"to_pos={metrics['sample_to_positive_dist_mean']:.3f}+-{metrics['sample_to_positive_dist_std']:.3f} "
                f"prefactor_mean={metrics['prefactor_mean']:.3f} train_auc={metrics['prefactor_train_auroc']:.3f}"
            )
            model.train()

        if completed_epoch:
            scheduler.step()

    progress_bar.close()

    torch.save(model.state_dict(), checkpoint_dir / "final_model.pt")
    (history_json_path.parent / "config_used.json").write_text(
        json.dumps(asdict(cfg), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(CONFIG)
