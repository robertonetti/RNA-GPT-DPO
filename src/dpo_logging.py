from __future__ import annotations

from pathlib import Path

import torch

from src.dpo_config import Config


def _fmt_float(value: float, digits: int = 4) -> str:
    """Format floats consistently in logs, handling NaN values."""
    if value == value:
        return f"{value:.{digits}f}"
    return "nan"


def _print_section(title: str) -> None:
    """Print a visually clear section header for terminal logs."""
    bar = "=" * 96
    print(f"\n{bar}")
    print(title)
    print(bar)


def print_run_configuration(
    cfg: Config,
    device: torch.device,
    train_pair_count: int,
    val_pair_count: int,
    vae_good_count: int,
    vae_bad_count: int,
    dist2530_good_count: int,
    dist2530_bad_count: int,
    dn_eval_count: int,
    train_good_fasta_path: Path,
    train_bad_fasta_path: Path,
    train_csv_mapping_path: Path,
    val_good_fasta_path: Path,
    val_bad_fasta_path: Path,
    val_csv_mapping_path: Path,
) -> None:
    """Print the run setup in a stable, readable order before training."""
    _print_section("RUN CONFIGURATION")
    print("Model:")
    print(f"  Device                          : {device}")
    print(f"  Layers frozen                   : {cfg.layers_to_freeze}")
    print(f"  torch.compile                   : {cfg.use_torch_compile}")
    print(f"  Suppress torch._dynamo errors   : {cfg.suppress_dynamo_errors}")
    print("\nTraining hyperparameters:")
    print(f"  Epochs                          : {cfg.num_epochs}")
    print(f"  Batch size                      : {cfg.batch_size}")
    print(f"  Learning rate                   : {cfg.learning_rate}")
    print(f"  Beta (DPO)                      : {cfg.beta}")
    print(f"  Block size                      : {cfg.block_size}")
    print("\nMetrics configuration:")
    print(f"  Compute full NLL metrics        : {cfg.compute_full_nll_metrics}")
    print(f"  Compute separation metric       : {cfg.compute_val_separation_metric}")
    print(f"  DN likelihood mode              : {cfg.dn_likelihood_mode}")
    print(f"  DN eval size                    : {dn_eval_count}")
    print("\nData paths:")
    print(f"  Train good FASTA                : {train_good_fasta_path}")
    print(f"  Train bad FASTA                 : {train_bad_fasta_path}")
    print(f"  Train mapping CSV               : {train_csv_mapping_path}")
    print(f"  Val good FASTA                  : {val_good_fasta_path}")
    print(f"  Val bad FASTA                   : {val_bad_fasta_path}")
    print(f"  Val mapping CSV                 : {val_csv_mapping_path}")
    print("\nDataset sizes:")
    print(f"  Train pairs                     : {train_pair_count}")
    print(f"  Validation pairs                : {val_pair_count}")
    print(f"  VAE split (good / bad)          : {vae_good_count} / {vae_bad_count}")
    print(f"  Dist25-30 split (good / bad)    : {dist2530_good_count} / {dist2530_bad_count}")


def print_eval_summary(
    title: str,
    epoch_label: str,
    dpo_train_batch_epoch: float,
    dpo_train_full: float,
    dpo_val_full: float,
    nll_train_good_rand: float,
    nll_train_bad_rand: float,
    nll_val_good_rand: float,
    nll_val_bad_rand: float,
    nll_margin_train_rand: float,
    nll_margin_val_rand: float,
    mean_like_model: float,
    mean_like_ref: float,
    compute_val_separation_metric: bool,
    val_sep_corr: float,
    vae_sep_corr: float,
    dist2530_sep_corr: float,
    compute_full_nll_metrics: bool,
    nll_train_good_full: float,
    nll_val_good_full: float,
    nll_val_bad_full: float,
) -> None:
    """Print one compact but explicit metrics report (baseline or epoch)."""
    _print_section(title)
    print(f"{epoch_label}")
    print("\nDPO losses:")
    print(
        "  Train batch mean               : "
        f"{_fmt_float(dpo_train_batch_epoch, 6)}"
    )
    print(
        "  Train full                     : "
        f"{_fmt_float(dpo_train_full, 6)}"
    )
    print(
        "  Validation full                : "
        f"{_fmt_float(dpo_val_full, 6)}"
    )
    print("\nRandom-batch NLL:")
    print(
        "  Train good / bad               : "
        f"{_fmt_float(nll_train_good_rand)} / {_fmt_float(nll_train_bad_rand)}"
    )
    print(
        "  Validation good / bad          : "
        f"{_fmt_float(nll_val_good_rand)} / {_fmt_float(nll_val_bad_rand)}"
    )
    print(
        "  Margin train / val (bad-good) : "
        f"{_fmt_float(nll_margin_train_rand)} / {_fmt_float(nll_margin_val_rand)}"
    )
    print("\nDN likelihood:")
    print(
        "  Model / reference              : "
        f"{_fmt_float(mean_like_model, 6)} / {_fmt_float(mean_like_ref, 6)}"
    )

    if compute_val_separation_metric:
        print("\nSeparation metric (Pearson r, NLL vs label 0/1):")
        print(
            "  Validation / VAE / Dist25-30   : "
            f"{_fmt_float(val_sep_corr)} / {_fmt_float(vae_sep_corr)} / {_fmt_float(dist2530_sep_corr)}"
        )
    else:
        print("\nSeparation metric:")
        print("  Disabled")

    if compute_full_nll_metrics:
        print("\nFull-split NLL:")
        print(
            "  Train good                     : "
            f"{_fmt_float(nll_train_good_full)}"
        )
        print(
            "  Validation good / bad          : "
            f"{_fmt_float(nll_val_good_full)} / {_fmt_float(nll_val_bad_full)}"
        )
        print(
            "  Validation margin (bad-good)   : "
            f"{_fmt_float(nll_val_bad_full - nll_val_good_full)}"
        )
    else:
        print("\nFull-split NLL:")
        print("  Skipped (compute_full_nll_metrics=False)")
