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
    project_dir = Path(__file__).resolve().parent.parent

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.suppress_dynamo_errors:
        torch._dynamo.config.suppress_errors = True

    dn_path = resolve_path(project_dir, cfg.dn_path)
    ckpt_path_gpt = resolve_path(project_dir, cfg.pretrained_ckpt_path)

    with dn_path.open("r", encoding="utf-8") as f:
        temp = f.read().splitlines()[1::2]
        temp = "\n".join(temp)

    chars = sorted(list(set(temp)))
    pad_symbol = "?"
    chars = chars + [pad_symbol]
    stoi = {ch: i for i, ch in enumerate(chars)}

    pad_token = stoi[pad_symbol]
    vocab_size = len(stoi)

    def encode(s: str) -> List[int]:
        return [stoi[c] for c in s]

    dn_dataset = load_dataset(
        seq_path=str(dn_path),
        label_path=".",
        block_size=cfg.max_context,
        train_size=2000,
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

    model = legacy_model

    if cfg.layers_to_freeze > 0:
        print(f"Freezing embeddings and first {cfg.layers_to_freeze} blocks...")
        for p in model.token_emb.parameters():
            p.requires_grad = False
        for block in model.layers[: cfg.layers_to_freeze]:
            for p in block.parameters():
                p.requires_grad = False

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
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler_step_size,
        gamma=cfg.scheduler_gamma,
    )

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

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    history_json_path.parent.mkdir(parents=True, exist_ok=True)

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

    vae_good_data = [pad_encode(s, cfg.block_size, pad_token, encode) for s in read_fasta(vae_good_fasta_path)]
    vae_bad_data = [pad_encode(s, cfg.block_size, pad_token, encode) for s in read_fasta(vae_bad_fasta_path)]

    dist2530_good_data = [
        pad_encode(s, cfg.block_size, pad_token, encode) for s in read_fasta(dist2530_good_fasta_path)
    ]
    dist2530_bad_data = [
        pad_encode(s, cfg.block_size, pad_token, encode) for s in read_fasta(dist2530_bad_fasta_path)
    ]

    train_pair_dataset = PreferencePairDataset(dpo_dataset)
    val_pair_dataset = PreferencePairDataset(val_dpo_dataset)

    train_loader = DataLoader(train_pair_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    train_eval_loader = DataLoader(train_pair_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_pair_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    dn_full_data = dn_dataset["val"]["data"]
    if cfg.dn_likelihood_mode == "full":
        dn_eval_data = dn_full_data
    elif cfg.dn_likelihood_mode == "fixed_subsample":
        k = min(cfg.dn_fixed_subsample_size, len(dn_full_data))
        rng = random.Random(cfg.dn_fixed_subsample_seed)
        dn_eval_indices = rng.sample(range(len(dn_full_data)), k)
        dn_eval_data = [dn_full_data[i] for i in dn_eval_indices]
    else:
        raise ValueError("dn_likelihood_mode must be either 'full' or 'fixed_subsample'.")

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

    history: Dict[str, List[Any]] = {
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
        dpo_train_batch_epoch = float("nan")

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
        model.train()
        epoch_batch_losses: List[float] = []

        for x_w, x_l in train_loader:
            x_w = x_w.to(device)
            x_l = x_l.to(device)
            y_w = _build_labels_from_inputs(x_w, pad_token)
            y_l = _build_labels_from_inputs(x_l, pad_token)

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

        scheduler.step()

        model.eval()
        with torch.inference_mode():
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
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        history_json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    config_dump_path = history_json_path.parent / "config_used.json"
    config_dump_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"Training completed. History saved to {history_json_path}")
    print(f"Config saved to {config_dump_path}")


if __name__ == "__main__":
    main(CONFIG)
