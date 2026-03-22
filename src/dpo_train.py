#!/usr/bin/env python3

"""
Epoch-based DPO training script extracted from DPO_train_epochs.ipynb.

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
    compute_random_batch_nll,
    compute_full_dataset_nll,
    compute_full_dataset_dpo_loss_from_loader,
    compute_mean_token_likelihood,
    compute_val_separation_correlation,
)
from src.dpo_plotting import save_epoch_figures
from src.dpo_logging import print_run_configuration, print_eval_summary


# ============================
# Main training
# ============================
def main(cfg: Config) -> None:
    """Run the complete epoch-based DPO pipeline.

    Input:
    - cfg: ``Config`` dataclass with paths, model hyperparameters, and metric switches.

    Main stages:
    - Build vocabulary and tokenizer from DN data.
    - Load pretrained model and create frozen reference copy.
    - Build preference datasets and dataloaders.
    - Evaluate baseline metrics (epoch 0).
    - Train for ``cfg.num_epochs`` and evaluate after each epoch.
    - Save checkpoints, plots, metric history JSON, and resolved config JSON.

    Output:
    - None (side effects only: files on disk + console logs).
    """
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

    # Resolve dataset/checkpoint paths against project root.
    dn_path = resolve_path(project_dir, cfg.dn_path)
    ckpt_path_gpt = resolve_path(project_dir, cfg.pretrained_ckpt_path)

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

    # Load DPO datasets and create DataLoaders for train/val pairs and extra eval sets.
    train_good_fasta_path = resolve_path(project_dir, cfg.train_good_fasta_path)
    train_bad_fasta_path = resolve_path(project_dir, cfg.train_bad_fasta_path)
    train_csv_mapping_path = resolve_path(project_dir, cfg.train_csv_mapping_path)

    val_good_fasta_path = resolve_path(project_dir, cfg.val_good_fasta_path)
    val_bad_fasta_path = resolve_path(project_dir, cfg.val_bad_fasta_path)
    val_csv_mapping_path = resolve_path(project_dir, cfg.val_csv_mapping_path)

    vae_good_fasta_path = resolve_path(project_dir, cfg.vae_good_fasta_path)
    vae_bad_fasta_path = resolve_path(project_dir, cfg.vae_bad_fasta_path)

    dist2530_good_fasta_path = resolve_path(project_dir, cfg.dist2530_good_fasta_path)
    dist2530_bad_fasta_path = resolve_path(project_dir, cfg.dist2530_bad_fasta_path)

    checkpoint_dir = resolve_path(project_dir, cfg.checkpoint_dir)
    image_dir = resolve_path(project_dir, cfg.image_dir)
    history_json_path = resolve_path(project_dir, cfg.history_json_path)

    # Ensure output directories always exist before writing files.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    history_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Load datasets 
    dpo_dataset = load_dpo_dataset(
        good_fasta_path=train_good_fasta_path,
        bad_fasta_path=train_bad_fasta_path,
        csv_path=train_csv_mapping_path,
        block_size=cfg.block_size,
        pad_token=pad_token,
        encode_fn=encode,
    )

    val_dpo_dataset = load_dpo_dataset( 
        good_fasta_path=val_good_fasta_path,
        bad_fasta_path=val_bad_fasta_path,
        csv_path=val_csv_mapping_path,
        block_size=cfg.block_size,
        pad_token=pad_token,
        encode_fn=encode,
    )

    vae_good_data = [pad_encode(s, cfg.block_size, pad_token, encode) for s in read_fasta(vae_good_fasta_path)] # shape (N_vae_good, block_size)
    vae_bad_data = [pad_encode(s, cfg.block_size, pad_token, encode) for s in read_fasta(vae_bad_fasta_path)] # shape (N_vae_bad, block_size)

    dist2530_good_data = [
        pad_encode(s, cfg.block_size, pad_token, encode) for s in read_fasta(dist2530_good_fasta_path)
    ] # shape (N_dist2530_good, block_size)
    dist2530_bad_data = [
        pad_encode(s, cfg.block_size, pad_token, encode) for s in read_fasta(dist2530_bad_fasta_path)
    ] # shape (N_dist2530_bad, block_size)

    # Create DataLoaders for training/validation pairs and extra eval sets.
    train_pair_dataset = PreferencePairDataset(dpo_dataset) # shape (N_train_pairs, 2) where each item is (good_idx, bad_idx)
    val_pair_dataset = PreferencePairDataset(val_dpo_dataset) # shape (N_val_pairs, 2) where each item is (good_idx, bad_idx)

    # Train loader is shuffled; eval loaders are deterministic.
    train_loader = DataLoader(train_pair_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False) # shape (N_train_pairs // batch_size, batch_size, 2)
    train_eval_loader = DataLoader(train_pair_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False) # shape (N_train_pairs // batch_size, batch_size, 2)
    val_loader = DataLoader(val_pair_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False) # shape (N_val_pairs // batch_size, batch_size, 2)

    # 
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

    # Print run configuration summary before starting training.
    print_run_configuration(
        cfg=cfg,
        device=device,
        train_pair_count=len(train_pair_dataset),
        val_pair_count=len(val_pair_dataset),
        vae_good_count=len(vae_good_data),
        vae_bad_count=len(vae_bad_data),
        dist2530_good_count=len(dist2530_good_data),
        dist2530_bad_count=len(dist2530_bad_data),
        dn_eval_count=len(dn_eval_data),
        train_good_fasta_path=train_good_fasta_path,
        train_bad_fasta_path=train_bad_fasta_path,
        train_csv_mapping_path=train_csv_mapping_path,
        val_good_fasta_path=val_good_fasta_path,
        val_bad_fasta_path=val_bad_fasta_path,
        val_csv_mapping_path=val_csv_mapping_path,
    )

    # Initialize history dictionary to store metrics across epochs for plotting and JSON export.
    history: Dict[str, List[Any]] = {
        # Store one value per epoch for each metric key.
        "epoch": [],
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
        "val_sep_corr": [],
        "val_good_seq_nll": [],
        "val_bad_seq_nll": [],
        "vae_sep_corr": [],
        "vae_good_seq_nll": [],
        "vae_bad_seq_nll": [],
        "dist2530_sep_corr": [],
        "dist2530_good_seq_nll": [],
        "dist2530_bad_seq_nll": [],
    }

    # Baseline (epoch 0)
    model.eval()
    with torch.inference_mode():
        # No train batches were run yet.
        dpo_train_batch_epoch = float("nan")

        dpo_train_full = compute_full_dataset_dpo_loss_from_loader(
            model, ref_model, train_eval_loader, pad_token, device, beta=cfg.beta
        )
        dpo_val_full = compute_full_dataset_dpo_loss_from_loader(
            model, ref_model, val_loader, pad_token, device, beta=cfg.beta
        )

        eval_bs = min(cfg.batch_size, cfg.nll_rand_eval_batch_size_cap)
        # Random-batch NLL estimates for fast monitoring.
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
        nll_margin_train_rand = nll_train_bad_rand - nll_train_good_rand
        nll_margin_val_rand = nll_val_bad_rand - nll_val_good_rand

        if cfg.compute_full_nll_metrics:
            # Optional full-dataset metrics (more expensive).
            nll_train_good_full = compute_full_dataset_nll(
                model, dpo_dataset["good_data"], pad_token, device, batch_size=cfg.batch_size
            )
            nll_val_good_full = compute_full_dataset_nll(
                model, val_dpo_dataset["good_data"], pad_token, device, batch_size=cfg.batch_size
            )
            nll_val_bad_full = compute_full_dataset_nll(
                model, val_dpo_dataset["bad_data"], pad_token, device, batch_size=cfg.batch_size
            )
        else:
            nll_train_good_full = float("nan")
            nll_val_good_full = float("nan")
            nll_val_bad_full = float("nan")

        full_eval_bs = min(cfg.batch_size, cfg.full_eval_batch_size_cap)
        # Compare language likelihood of current model vs frozen reference.
        mean_like_model = compute_mean_token_likelihood(
            model, dn_eval_data, pad_token, device, batch_size=full_eval_bs
        )
        mean_like_ref = compute_mean_token_likelihood(
            ref_model, dn_eval_data, pad_token, device, batch_size=full_eval_bs
        )

        if cfg.compute_val_separation_metric:
            # Compute class-separation correlation + per-sequence NLL lists.
            val_sep_corr, val_good_seq_nll, val_bad_seq_nll = compute_val_separation_correlation(
                model,
                val_dpo_dataset["good_data"],
                val_dpo_dataset["bad_data"],
                pad_token,
                device,
                batch_size=full_eval_bs,
            )
            vae_sep_corr, vae_good_seq_nll, vae_bad_seq_nll = compute_val_separation_correlation(
                model,
                vae_good_data,
                vae_bad_data,
                pad_token,
                device,
                batch_size=full_eval_bs,
            )
            dist2530_sep_corr, dist2530_good_seq_nll, dist2530_bad_seq_nll = compute_val_separation_correlation(
                model,
                dist2530_good_data,
                dist2530_bad_data,
                pad_token,
                device,
                batch_size=full_eval_bs,
            )
        else:
            val_sep_corr = float("nan")
            val_good_seq_nll = []
            val_bad_seq_nll = []
            vae_sep_corr = float("nan")
            vae_good_seq_nll = []
            vae_bad_seq_nll = []
            dist2530_sep_corr = float("nan")
            dist2530_good_seq_nll = []
            dist2530_bad_seq_nll = []

    for key, value in [
        # Persist baseline metrics as epoch 0 in history.
        ("epoch", 0),
        ("dpo_train_batch_epoch", dpo_train_batch_epoch),
        ("dpo_train_full", dpo_train_full),
        ("dpo_val_full", dpo_val_full),
        ("nll_train_good_rand", nll_train_good_rand),
        ("nll_train_bad_rand", nll_train_bad_rand),
        ("nll_val_good_rand", nll_val_good_rand),
        ("nll_val_bad_rand", nll_val_bad_rand),
        ("nll_margin_train_rand", nll_margin_train_rand),
        ("nll_margin_val_rand", nll_margin_val_rand),
        ("nll_train_good_full", nll_train_good_full),
        ("nll_val_good_full", nll_val_good_full),
        ("nll_val_bad_full", nll_val_bad_full),
        ("mean_like_model", mean_like_model),
        ("mean_like_ref", mean_like_ref),
        ("val_sep_corr", val_sep_corr),
        ("val_good_seq_nll", val_good_seq_nll),
        ("val_bad_seq_nll", val_bad_seq_nll),
        ("vae_sep_corr", vae_sep_corr),
        ("vae_good_seq_nll", vae_good_seq_nll),
        ("vae_bad_seq_nll", vae_bad_seq_nll),
        ("dist2530_sep_corr", dist2530_sep_corr),
        ("dist2530_good_seq_nll", dist2530_good_seq_nll),
        ("dist2530_bad_seq_nll", dist2530_bad_seq_nll),
    ]:
        history[key].append(value)

    save_epoch_figures(
        history,
        main_path=image_dir / "epoch_000_main.png",
        violin_path=image_dir / "epoch_000_violin.png",
    )

    print_eval_summary(
        title="BASELINE EVALUATION",
        epoch_label="Epoch 0 / before training",
        dpo_train_batch_epoch=dpo_train_batch_epoch,
        dpo_train_full=dpo_train_full,
        dpo_val_full=dpo_val_full,
        nll_train_good_rand=nll_train_good_rand,
        nll_train_bad_rand=nll_train_bad_rand,
        nll_val_good_rand=nll_val_good_rand,
        nll_val_bad_rand=nll_val_bad_rand,
        nll_margin_train_rand=nll_margin_train_rand,
        nll_margin_val_rand=nll_margin_val_rand,
        mean_like_model=mean_like_model,
        mean_like_ref=mean_like_ref,
        compute_val_separation_metric=cfg.compute_val_separation_metric,
        val_sep_corr=val_sep_corr,
        vae_sep_corr=vae_sep_corr,
        dist2530_sep_corr=dist2530_sep_corr,
        compute_full_nll_metrics=cfg.compute_full_nll_metrics,
        nll_train_good_full=nll_train_good_full,
        nll_val_good_full=nll_val_good_full,
        nll_val_bad_full=nll_val_bad_full,
    )

    for epoch in tqdm(range(1, cfg.num_epochs + 1), desc="Epochs"):
        # ---- Training phase ----
        model.train()
        epoch_batch_losses: List[float] = []

        for x_w, x_l in train_loader:
            # Move preferred/dispreferred sequences to target device.
            x_w = x_w.to(device)
            x_l = x_l.to(device)
            # Build next-token labels by shifting inputs.
            y_w = _build_labels_from_inputs(x_w, pad_token)
            y_l = _build_labels_from_inputs(x_l, pad_token)

            # Standard optimization step.
            optimizer.zero_grad(set_to_none=True)
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

            epoch_batch_losses.append(loss.item())

        # Epoch-level LR update.
        scheduler.step()

        # ---- Evaluation phase ----
        model.eval()
        with torch.inference_mode():
            # Average training loss over all batches in this epoch.
            dpo_train_batch_epoch = sum(epoch_batch_losses) / max(len(epoch_batch_losses), 1)

            dpo_train_full = compute_full_dataset_dpo_loss_from_loader(
                model, ref_model, train_eval_loader, pad_token, device, beta=cfg.beta
            )
            dpo_val_full = compute_full_dataset_dpo_loss_from_loader(
                model, ref_model, val_loader, pad_token, device, beta=cfg.beta
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
            nll_margin_train_rand = nll_train_bad_rand - nll_train_good_rand
            nll_margin_val_rand = nll_val_bad_rand - nll_val_good_rand

            if cfg.compute_full_nll_metrics:
                nll_train_good_full = compute_full_dataset_nll(
                    model, dpo_dataset["good_data"], pad_token, device, batch_size=cfg.batch_size
                )
                nll_val_good_full = compute_full_dataset_nll(
                    model, val_dpo_dataset["good_data"], pad_token, device, batch_size=cfg.batch_size
                )
                nll_val_bad_full = compute_full_dataset_nll(
                    model, val_dpo_dataset["bad_data"], pad_token, device, batch_size=cfg.batch_size
                )
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
                val_sep_corr, val_good_seq_nll, val_bad_seq_nll = compute_val_separation_correlation(
                    model,
                    val_dpo_dataset["good_data"],
                    val_dpo_dataset["bad_data"],
                    pad_token,
                    device,
                    batch_size=full_eval_bs,
                )
                vae_sep_corr, vae_good_seq_nll, vae_bad_seq_nll = compute_val_separation_correlation(
                    model,
                    vae_good_data,
                    vae_bad_data,
                    pad_token,
                    device,
                    batch_size=full_eval_bs,
                )
                dist2530_sep_corr, dist2530_good_seq_nll, dist2530_bad_seq_nll = compute_val_separation_correlation(
                    model,
                    dist2530_good_data,
                    dist2530_bad_data,
                    pad_token,
                    device,
                    batch_size=full_eval_bs,
                )
            else:
                val_sep_corr = float("nan")
                val_good_seq_nll = []
                val_bad_seq_nll = []
                vae_sep_corr = float("nan")
                vae_good_seq_nll = []
                vae_bad_seq_nll = []
                dist2530_sep_corr = float("nan")
                dist2530_good_seq_nll = []
                dist2530_bad_seq_nll = []

        for key, value in [
            # Append current epoch metrics to history.
            ("epoch", epoch),
            ("dpo_train_batch_epoch", dpo_train_batch_epoch),
            ("dpo_train_full", dpo_train_full),
            ("dpo_val_full", dpo_val_full),
            ("nll_train_good_rand", nll_train_good_rand),
            ("nll_train_bad_rand", nll_train_bad_rand),
            ("nll_val_good_rand", nll_val_good_rand),
            ("nll_val_bad_rand", nll_val_bad_rand),
            ("nll_margin_train_rand", nll_margin_train_rand),
            ("nll_margin_val_rand", nll_margin_val_rand),
            ("nll_train_good_full", nll_train_good_full),
            ("nll_val_good_full", nll_val_good_full),
            ("nll_val_bad_full", nll_val_bad_full),
            ("mean_like_model", mean_like_model),
            ("mean_like_ref", mean_like_ref),
            ("val_sep_corr", val_sep_corr),
            ("val_good_seq_nll", val_good_seq_nll),
            ("val_bad_seq_nll", val_bad_seq_nll),
            ("vae_sep_corr", vae_sep_corr),
            ("vae_good_seq_nll", vae_good_seq_nll),
            ("vae_bad_seq_nll", vae_bad_seq_nll),
            ("dist2530_sep_corr", dist2530_sep_corr),
            ("dist2530_good_seq_nll", dist2530_good_seq_nll),
            ("dist2530_bad_seq_nll", dist2530_bad_seq_nll),
        ]:
            history[key].append(value)

        save_epoch_figures(
            history,
            main_path=image_dir / f"epoch_{epoch:03d}_main.png",
            violin_path=image_dir / f"epoch_{epoch:03d}_violin.png",
        )

        print_eval_summary(
            title="EPOCH EVALUATION",
            epoch_label=f"Epoch {epoch} / {cfg.num_epochs}",
            dpo_train_batch_epoch=dpo_train_batch_epoch,
            dpo_train_full=dpo_train_full,
            dpo_val_full=dpo_val_full,
            nll_train_good_rand=nll_train_good_rand,
            nll_train_bad_rand=nll_train_bad_rand,
            nll_val_good_rand=nll_val_good_rand,
            nll_val_bad_rand=nll_val_bad_rand,
            nll_margin_train_rand=nll_margin_train_rand,
            nll_margin_val_rand=nll_margin_val_rand,
            mean_like_model=mean_like_model,
            mean_like_ref=mean_like_ref,
            compute_val_separation_metric=cfg.compute_val_separation_metric,
            val_sep_corr=val_sep_corr,
            vae_sep_corr=vae_sep_corr,
            dist2530_sep_corr=dist2530_sep_corr,
            compute_full_nll_metrics=cfg.compute_full_nll_metrics,
            nll_train_good_full=nll_train_good_full,
            nll_val_good_full=nll_val_good_full,
            nll_val_bad_full=nll_val_bad_full,
        )

        ckpt_path = checkpoint_dir / f"dpo_model_epoch{epoch}.pt"
        # Save model checkpoint every epoch.
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        # Save cumulative history after each epoch for crash safety.
        history_json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    config_dump_path = history_json_path.parent / "config_used.json"
    # Save exact run configuration for reproducibility.
    config_dump_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"Training completed. History saved to {history_json_path}")
    print(f"Config saved to {config_dump_path}")


if __name__ == "__main__":
    main(CONFIG)
