#!/usr/bin/env python3

"""
DPO training script extracted from DPO_train_epochs.ipynb.

Edit the CONFIG section below to control all parameters, hyperparameters,
input paths, checkpoint path, and image output path.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.transformer import GPTTransformer
from src.Transformer_Reint import load_dataset

from src.dpo_config import Config, CONFIG
from src.dpo_data import (
    resolve_path,
    read_fasta,
    pad_encode,
    load_dpo_dataset,
    PreferencePairDataset,
    _build_labels_from_inputs,
)
from src.dpo_metrics import (
    compute_dpo_loss_from_tensors,
    compute_reint_loss_from_tensors,
)
from src.dpo_logging import print_run_configuration
from src.dpo_train_utils import (
    _append_history_row,
    _distance_to_reference,
    _evaluate_model_state,
    _has_path,
    _print_eval_block,
    _save_eval_artifacts,
)


# ###########################################################################
# ## Main Training Entry                                                   ##
# ###########################################################################
def main(cfg: Config) -> None:
    """Run the complete DPO pipeline with iteration-based evaluation cadence.

    Input:
    - cfg: ``Config`` dataclass with paths, model hyperparameters, and metric switches.

    Main stages:
    - Build vocabulary and tokenizer from DN data.
    - Load pretrained model and create frozen reference copy.
    - Build preference datasets and dataloaders.
    - Evaluate baseline metrics (iteration 0).
    - Train for up to ``cfg.max_iterations`` optimizer steps and evaluate every fixed number of iterations.
    - Save checkpoints, plots, metric history JSON, and resolved config JSON.

    Output:
    - None (side effects only: files on disk + console logs).
    """
    # #######################################################################
    # ## Runtime Setup                                                      ##
    # #######################################################################

    # Project root is used to resolve all relative paths from config.
    project_dir = Path(__file__).resolve().parent.parent

    # Seed Python and PyTorch RNGs for reproducible runs.
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Prefer CUDA, then Apple MPS, otherwise CPU fallback.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if torch.cuda.is_available():
        # Enable TensorFloat32 acceleration on supported GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.suppress_dynamo_errors:
        # Avoid hard failures when torch.compile graph capture hits edge cases.
        torch._dynamo.config.suppress_errors = True

    # #######################################################################
    # ## Vocabulary + Tokenizer Construction                               ##
    # #######################################################################

    # Resolve dataset/checkpoint paths against project root.
    dn_path = resolve_path(project_dir, cfg.dn_path)
    ckpt_path_gpt = resolve_path(project_dir, cfg.pretrained_ckpt_path)

    dn_sequences = read_fasta(dn_path)
    if len(dn_sequences) == 0:
        raise ValueError(f"DN dataset is empty: {dn_path}")
    reference_seq = dn_sequences[0]

    # Build tokenizer alphabet directly from DN file symbols.
    with dn_path.open("r", encoding="utf-8") as f:
        temp = f.read().splitlines()[1::2]
        temp = "\n".join(temp)

    chars = sorted(list(set(temp)))
    # Reserve one explicit padding symbol.
    pad_symbol = "?"
    chars = chars + [pad_symbol]
    stoi = {ch: i for i, ch in enumerate(chars)}

    pad_token = stoi[pad_symbol]
    vocab_size = len(stoi)

    def encode(s: str) -> List[int]:
        """Encode a string into token ids using the run-local ``stoi`` mapping.

        Input:
        - s: Raw text sequence.

        Output:
        - List of length ``len(s)`` with integer token ids.
        """
        # Character-level encoding used by all data loaders.
        return [stoi[c] for c in s]

    dn_dataset = load_dataset(
        seq_path=str(dn_path),
        label_path=".",
        block_size=cfg.max_context,
        train_size=2000, # dataset is 817 sequences, so train_size=2000 means we use the whole dataset for training
        encode_fn=encode,
        pad_token=pad_token,
    )

    # #######################################################################
    # ## Model/Reference Initialization                                    ##
    # #######################################################################

    legacy_model = GPTTransformer(
        vocab_size=vocab_size,
        embed_dim=cfg.n_embd,
        num_layers=cfg.n_layer,
        num_heads=cfg.n_head,
        mlp_ratio=4,
        dropout_p=0.1,
        pad_id=pad_token,
    )
    legacy_model.load_state_dict(torch.load(ckpt_path_gpt, map_location="cpu"))
    legacy_model = legacy_model.to(device)

    # Fine-tuned model starts from pretrained GPT checkpoint.
    model = legacy_model

    if cfg.layers_to_freeze > 0:
        print(f"Freezing embeddings and first {cfg.layers_to_freeze} blocks...")
        # Freeze token embedding table first.
        for p in model.token_emb.parameters():
            p.requires_grad = False
        # Freeze first N transformer blocks.
        for block in model.layers[: cfg.layers_to_freeze]:
            for p in block.parameters():
                p.requires_grad = False

    # Reference model is a frozen copy used in DPO comparisons.
    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()

    if cfg.use_torch_compile:
        try:
            model = torch.compile(model)
            ref_model = torch.compile(ref_model)
            print("torch.compile enabled for both model and reference model.")
        except Exception as e:
            print(f"torch.compile failed, falling back to eager mode: {e}")

    optimizer = torch.optim.AdamW(
        # Exclude frozen parameters from optimizer state.
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler_step_size,
        gamma=cfg.scheduler_gamma,
    )

    # #######################################################################
    # ## Paths + Dataset Descriptions                                      ##
    # #######################################################################

    # Load DPO datasets and create DataLoaders for train/val pairs and extra eval sets.
    train_good_fasta_path = resolve_path(project_dir, cfg.train_good_fasta_path)
    train_bad_fasta_path = resolve_path(project_dir, cfg.train_bad_fasta_path)
    train_csv_mapping_path = resolve_path(project_dir, cfg.train_csv_mapping_path)

    val_enabled = (
        _has_path(cfg.val_good_fasta_path)
        and _has_path(cfg.val_bad_fasta_path)
        and _has_path(cfg.val_csv_mapping_path)
    )
    val_1_enabled = _has_path(cfg.val_1_good_fasta_path) and _has_path(cfg.val_1_bad_fasta_path)
    val_2_enabled = _has_path(cfg.val_2_good_fasta_path) and _has_path(cfg.val_2_bad_fasta_path)

    val_good_fasta_path = resolve_path(project_dir, cfg.val_good_fasta_path) if val_enabled else None
    val_bad_fasta_path = resolve_path(project_dir, cfg.val_bad_fasta_path) if val_enabled else None
    val_csv_mapping_path = resolve_path(project_dir, cfg.val_csv_mapping_path) if val_enabled else None

    val_1_good_fasta_path = resolve_path(project_dir, cfg.val_1_good_fasta_path) if val_1_enabled else None
    val_1_bad_fasta_path = resolve_path(project_dir, cfg.val_1_bad_fasta_path) if val_1_enabled else None

    val_2_good_fasta_path = (
        resolve_path(project_dir, cfg.val_2_good_fasta_path) if val_2_enabled else None
    )
    val_2_bad_fasta_path = (
        resolve_path(project_dir, cfg.val_2_bad_fasta_path) if val_2_enabled else None
    )

    checkpoint_dir = resolve_path(project_dir, cfg.checkpoint_dir)
    image_dir = resolve_path(project_dir, cfg.image_dir)
    history_json_path = resolve_path(project_dir, cfg.history_json_path)
    dataset_descriptions = {
        "train": cfg.train_dataset_description,
        "val": cfg.val_dataset_description,
        "val_1": cfg.val_1_dataset_description,
        "val_2": cfg.val_2_dataset_description,
    }

    # Ensure output directories always exist before writing files.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    history_json_path.parent.mkdir(parents=True, exist_ok=True)

    # #######################################################################
    # ## Sequence Loading + Distance Features                              ##
    # #######################################################################

    # Load raw sequences once to compute distance-to-reference metrics.
    train_good_sequences = read_fasta(train_good_fasta_path)
    train_bad_sequences = read_fasta(train_bad_fasta_path)
    val_good_sequences = read_fasta(val_good_fasta_path) if val_enabled else []
    val_bad_sequences = read_fasta(val_bad_fasta_path) if val_enabled else []
    val_1_good_sequences = read_fasta(val_1_good_fasta_path) if val_1_enabled else []
    val_1_bad_sequences = read_fasta(val_1_bad_fasta_path) if val_1_enabled else []
    val_2_good_sequences = read_fasta(val_2_good_fasta_path) if val_2_enabled else []
    val_2_bad_sequences = read_fasta(val_2_bad_fasta_path) if val_2_enabled else []

    train_good_distances = torch.tensor(
        [_distance_to_reference(seq, reference_seq) for seq in train_good_sequences],
        dtype=torch.float32,
    )
    train_bad_distances = torch.tensor(
        [_distance_to_reference(seq, reference_seq) for seq in train_bad_sequences],
        dtype=torch.float32,
    )
    val_good_distances = torch.tensor(
        [_distance_to_reference(seq, reference_seq) for seq in val_good_sequences],
        dtype=torch.float32,
    )
    val_bad_distances = torch.tensor(
        [_distance_to_reference(seq, reference_seq) for seq in val_bad_sequences],
        dtype=torch.float32,
    )
    val_1_good_distances = torch.tensor(
        [_distance_to_reference(seq, reference_seq) for seq in val_1_good_sequences],
        dtype=torch.float32,
    )
    val_1_bad_distances = torch.tensor(
        [_distance_to_reference(seq, reference_seq) for seq in val_1_bad_sequences],
        dtype=torch.float32,
    )
    val_2_good_distances = torch.tensor(
        [_distance_to_reference(seq, reference_seq) for seq in val_2_good_sequences],
        dtype=torch.float32,
    )
    val_2_bad_distances = torch.tensor(
        [_distance_to_reference(seq, reference_seq) for seq in val_2_bad_sequences],
        dtype=torch.float32,
    )

    train_all_distances = torch.cat([train_good_distances, train_bad_distances], dim=0).tolist()
    val_all_distances = torch.cat([val_good_distances, val_bad_distances], dim=0).tolist()
    val_1_all_distances = torch.cat([val_1_good_distances, val_1_bad_distances], dim=0).tolist()
    val_2_all_distances = torch.cat([val_2_good_distances, val_2_bad_distances], dim=0).tolist()

    def _build_distance_binned_entries(
        train_good_nll: List[float],
        train_bad_nll: List[float],
        val_good_nll: List[float],
        val_bad_nll: List[float],
        val_1_good_nll: List[float],
        val_1_bad_nll: List[float],
        val_2_good_nll: List[float],
        val_2_bad_nll: List[float],
    ) -> List[Dict[str, Any]]:
        """Build plotting payload for train/val/val_1/val_2 distance-binned correlation figure."""
        entries: List[Dict[str, Any]] = []

        train_desc = dataset_descriptions.get("train", "").strip()
        entries.append(
            {
                "title": f"train | {train_desc}" if train_desc else "train",
                "distances": train_all_distances,
                "good_nll": train_good_nll,
                "bad_nll": train_bad_nll,
            }
        )

        if val_enabled:
            val_desc = dataset_descriptions.get("val", "").strip()
            entries.append(
                {
                    "title": f"val | {val_desc}" if val_desc else "val",
                    "distances": val_all_distances,
                    "good_nll": val_good_nll,
                    "bad_nll": val_bad_nll,
                }
            )

        if val_1_enabled:
            val_1_desc = dataset_descriptions.get("val_1", "").strip()
            entries.append(
                {
                    "title": f"val_1 | {val_1_desc}" if val_1_desc else "val_1",
                    "distances": val_1_all_distances,
                    "good_nll": val_1_good_nll,
                    "bad_nll": val_1_bad_nll,
                }
            )

        if val_2_enabled:
            val_2_desc = dataset_descriptions.get("val_2", "").strip()
            entries.append(
                {
                    "title": f"val_2 | {val_2_desc}" if val_2_desc else "val_2",
                    "distances": val_2_all_distances,
                    "good_nll": val_2_good_nll,
                    "bad_nll": val_2_bad_nll,
                }
            )

        return entries

    # #######################################################################
    # ## Encoded Datasets + DataLoaders                                    ##
    # #######################################################################

    # Load datasets
    dpo_dataset = load_dpo_dataset(
        good_fasta_path=train_good_fasta_path,
        bad_fasta_path=train_bad_fasta_path,
        csv_path=train_csv_mapping_path,
        block_size=cfg.block_size,
        pad_token=pad_token,
        encode_fn=encode,
    )

    if val_enabled:
        val_dpo_dataset = load_dpo_dataset(
            good_fasta_path=val_good_fasta_path,
            bad_fasta_path=val_bad_fasta_path,
            csv_path=val_csv_mapping_path,
            block_size=cfg.block_size,
            pad_token=pad_token,
            encode_fn=encode,
        )
    else:
        val_dpo_dataset = None

    val_1_good_data = [pad_encode(s, cfg.block_size, pad_token, encode) for s in val_1_good_sequences] # shape (N_val_1_good, block_size)
    val_1_bad_data = [pad_encode(s, cfg.block_size, pad_token, encode) for s in val_1_bad_sequences] # shape (N_val_1_bad, block_size)

    val_2_good_data = [
        pad_encode(s, cfg.block_size, pad_token, encode) for s in val_2_good_sequences
    ] # shape (N_val_2_good, block_size)
    val_2_bad_data = [
        pad_encode(s, cfg.block_size, pad_token, encode) for s in val_2_bad_sequences
    ] # shape (N_val_2_bad, block_size)

    # Create DataLoaders for training/validation pairs and extra eval sets.
    train_pair_dataset = PreferencePairDataset(dpo_dataset) # shape (N_train_pairs, 2) where each item is (good_idx, bad_idx)
    val_pair_dataset = PreferencePairDataset(val_dpo_dataset) if val_enabled else None # shape (N_val_pairs, 2) where each item is (good_idx, bad_idx)

    # Train loader is shuffled; eval loaders are deterministic.
    train_loader = DataLoader(train_pair_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False) # shape (N_train_pairs // batch_size, batch_size, 2)
    train_eval_loader = DataLoader(train_pair_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False) # shape (N_train_pairs // batch_size, batch_size, 2)
    val_loader = (
        DataLoader(val_pair_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        if val_enabled
        else None
    ) # shape (N_val_pairs // batch_size, batch_size, 2)

    # #######################################################################
    # ## DN Evaluation Subset Selection                                    ##
    # #######################################################################

    dn_full_data = dn_dataset["val"]["data"]
    if cfg.dn_likelihood_mode == "full":
        # Evaluate likelihood on all DN validation sequences.
        dn_eval_data = dn_full_data
    elif cfg.dn_likelihood_mode == "fixed_subsample":
        # Evaluate on a fixed random subset for cheaper/stable tracking.
        k = min(cfg.dn_fixed_subsample_size, len(dn_full_data))
        rng = random.Random(cfg.dn_fixed_subsample_seed)
        dn_eval_indices = rng.sample(range(len(dn_full_data)), k)
        dn_eval_data = [dn_full_data[i] for i in dn_eval_indices]
    else:
        raise ValueError("dn_likelihood_mode must be either 'full' or 'fixed_subsample'.")

    # #######################################################################
    # ## Run Summary + History Initialization                              ##
    # #######################################################################

    # Print run configuration summary before starting training.
    print_run_configuration(
        cfg=cfg,
        device=device,
        train_pair_count=len(train_pair_dataset),
        val_pair_count=len(val_pair_dataset) if val_pair_dataset is not None else 0,
        val_1_good_count=len(val_1_good_data),
        val_1_bad_count=len(val_1_bad_data),
        val_2_good_count=len(val_2_good_data),
        val_2_bad_count=len(val_2_bad_data),
        dn_eval_count=len(dn_eval_data),
        train_good_fasta_path=train_good_fasta_path,
        train_bad_fasta_path=train_bad_fasta_path,
        train_csv_mapping_path=train_csv_mapping_path,
        val_good_fasta_path=val_good_fasta_path,
        val_bad_fasta_path=val_bad_fasta_path,
        val_csv_mapping_path=val_csv_mapping_path,
    )

    # Initialize history dictionary to store metrics across evaluation iterations.
    history: Dict[str, List[Any]] = {
        # Store one value per evaluation step for each metric key.
        "iteration": [],
        "dpo_train_batch_epoch": [],
        "dpo_train_full": [],
        "dpo_val_full": [],
        "nll_train_good_rand": [],
        "nll_train_bad_rand": [],
        "nll_val_good_rand": [],
        "nll_val_bad_rand": [],
        "nll_margin_train_rand": [],
        "nll_margin_val_rand": [],
        "nll_train_good_full": [],
        "nll_val_good_full": [],
        "nll_val_bad_full": [],
        "mean_like_model": [],
        "mean_like_ref": [],
        "train_good_seq_nll": [],
        "train_bad_seq_nll": [],
        "train_dist_nll_corr": [],
        "val_sep_corr": [],
        "val_good_seq_nll": [],
        "val_bad_seq_nll": [],
        "val_dist_nll_corr": [],
        "val_1_sep_corr": [],
        "val_1_good_seq_nll": [],
        "val_1_bad_seq_nll": [],
        "val_1_dist_nll_corr": [],
        "val_2_sep_corr": [],
        "val_2_good_seq_nll": [],
        "val_2_bad_seq_nll": [],
        "val_2_dist_nll_corr": [],
    }

    if cfg.eval_every_n_iterations <= 0:
        raise ValueError("eval_every_n_iterations must be > 0.")
    eval_period_iterations = cfg.eval_every_n_iterations

    # #######################################################################
    # ## Baseline Evaluation (Iteration 0)                                 ##
    # #######################################################################

    # Baseline (epoch 0)
    model.eval()
    with torch.inference_mode():
        baseline_metrics = _evaluate_model_state(
            model=model,
            ref_model=ref_model,
            train_eval_loader=train_eval_loader,
            val_loader=val_loader,
            pad_token=pad_token,
            device=device,
            cfg=cfg,
            dpo_dataset=dpo_dataset,
            val_dpo_dataset=val_dpo_dataset,
            val_enabled=val_enabled,
            val_1_enabled=val_1_enabled,
            val_2_enabled=val_2_enabled,
            val_1_good_data=val_1_good_data,
            val_1_bad_data=val_1_bad_data,
            val_2_good_data=val_2_good_data,
            val_2_bad_data=val_2_bad_data,
            dn_eval_data=dn_eval_data,
            train_good_distances=train_good_distances,
            train_bad_distances=train_bad_distances,
            val_good_distances=val_good_distances,
            val_bad_distances=val_bad_distances,
            val_1_good_distances=val_1_good_distances,
            val_1_bad_distances=val_1_bad_distances,
            val_2_good_distances=val_2_good_distances,
            val_2_bad_distances=val_2_bad_distances,
        )

    # No train batches were run yet.
    baseline_metrics["dpo_train_batch_epoch"] = float("nan")

    _append_history_row(history, {"iteration": 0, **baseline_metrics})

    # #######################################################################
    # ## Baseline Artifacts + Logs                                         ##
    # #######################################################################

    _save_eval_artifacts(
        history=history,
        image_dir=image_dir,
        iteration=0,
        dataset_descriptions=dataset_descriptions,
        eval_period_iterations=eval_period_iterations,
        train_good_distances=train_good_distances,
        train_bad_distances=train_bad_distances,
        val_good_distances=val_good_distances,
        val_bad_distances=val_bad_distances,
        build_distance_binned_entries=_build_distance_binned_entries,
        distance_nll_ylim_min=cfg.distance_nll_ylim_min,
        distance_nll_ylim_max=cfg.distance_nll_ylim_max,
        compute_val_roc_prc_ppv=cfg.compute_val_roc_prc_ppv,
    )
    _print_eval_block(
        title="BASELINE EVALUATION",
        epoch_label="Iteration 0 / before training",
        cfg=cfg,
        metrics=baseline_metrics,
    )

    # #######################################################################
    # ## Training Loop                                                     ##
    # #######################################################################

    if cfg.max_iterations <= 0:
        raise ValueError("max_iterations must be > 0.")

    global_iteration = 0
    epoch = 0
    pending_train_batch_losses: List[float] = []

    progress_bar = tqdm(total=cfg.max_iterations, desc="Iterations")
    while global_iteration < cfg.max_iterations:
        epoch += 1
        # ---- Training phase ----
        model.train()
        completed_epoch = True

        for x_w, x_l in train_loader:
            if global_iteration >= cfg.max_iterations:
                completed_epoch = False
                break
            # Move preferred/dispreferred sequences to target device.
            x_w = x_w.to(device)
            x_l = x_l.to(device)
            # Build next-token labels by shifting inputs.
            y_w = _build_labels_from_inputs(x_w, pad_token)
            y_l = _build_labels_from_inputs(x_l, pad_token)

            # Standard optimization step.
            optimizer.zero_grad(set_to_none=True)
            if cfg.reint:
                loss = compute_reint_loss_from_tensors(
                    model,
                    ref_model,
                    x_w,
                    y_w,
                    x_l,
                    y_l,
                    pad_token=pad_token,
                    beta=cfg.beta,
                    lambda_reint=cfg.lambda_reint,
                )
            else:
                loss = compute_dpo_loss_from_tensors(
                    model,
                    ref_model,
                    x_w,
                    y_w,
                    x_l,
                    y_l,
                    pad_token=pad_token,
                    beta=cfg.beta,
                )
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            pending_train_batch_losses.append(loss_value)
            global_iteration += 1
            progress_bar.update(1)

            should_evaluate = (global_iteration % eval_period_iterations) == 0
            if not should_evaluate:
                continue

            # ---- Evaluation phase ----
            model.eval()
            with torch.inference_mode():
                # Average training loss over batches since previous evaluation.
                dpo_train_batch_epoch = sum(pending_train_batch_losses) / max(len(pending_train_batch_losses), 1)
                iteration_metrics = _evaluate_model_state(
                    model=model,
                    ref_model=ref_model,
                    train_eval_loader=train_eval_loader,
                    val_loader=val_loader,
                    pad_token=pad_token,
                    device=device,
                    cfg=cfg,
                    dpo_dataset=dpo_dataset,
                    val_dpo_dataset=val_dpo_dataset,
                    val_enabled=val_enabled,
                    val_1_enabled=val_1_enabled,
                    val_2_enabled=val_2_enabled,
                    val_1_good_data=val_1_good_data,
                    val_1_bad_data=val_1_bad_data,
                    val_2_good_data=val_2_good_data,
                    val_2_bad_data=val_2_bad_data,
                    dn_eval_data=dn_eval_data,
                    train_good_distances=train_good_distances,
                    train_bad_distances=train_bad_distances,
                    val_good_distances=val_good_distances,
                    val_bad_distances=val_bad_distances,
                    val_1_good_distances=val_1_good_distances,
                    val_1_bad_distances=val_1_bad_distances,
                    val_2_good_distances=val_2_good_distances,
                    val_2_bad_distances=val_2_bad_distances,
                )
                iteration_metrics["dpo_train_batch_epoch"] = dpo_train_batch_epoch

            _append_history_row(history, {"iteration": global_iteration, **iteration_metrics})

            _save_eval_artifacts(
                history=history,
                image_dir=image_dir,
                iteration=global_iteration,
                dataset_descriptions=dataset_descriptions,
                eval_period_iterations=eval_period_iterations,
                train_good_distances=train_good_distances,
                train_bad_distances=train_bad_distances,
                val_good_distances=val_good_distances,
                val_bad_distances=val_bad_distances,
                build_distance_binned_entries=_build_distance_binned_entries,
                distance_nll_ylim_min=cfg.distance_nll_ylim_min,
                distance_nll_ylim_max=cfg.distance_nll_ylim_max,
                compute_val_roc_prc_ppv=cfg.compute_val_roc_prc_ppv,
            )
            _print_eval_block(
                title="ITERATION EVALUATION",
                epoch_label=f"Iteration {global_iteration} / epoch {epoch}",
                cfg=cfg,
                metrics=iteration_metrics,
            )

            ckpt_path = checkpoint_dir / f"dpo_model_iter{global_iteration}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

            # Save cumulative history after each evaluation for crash safety.
            history_json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            pending_train_batch_losses = []
            model.train()

        # Epoch-level LR update only after a full dataloader pass.
        if completed_epoch:
            scheduler.step()

    progress_bar.close()

    # #######################################################################
    # ## Final Run Metadata                                                 ##
    # #######################################################################

    config_dump_path = history_json_path.parent / "config_used.json"
    # Save exact run configuration for reproducibility.
    config_dump_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"Training completed. History saved to {history_json_path}")
    print(f"Config saved to {config_dump_path}")


if __name__ == "__main__":
    main(CONFIG)
