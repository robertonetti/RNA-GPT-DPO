from __future__ import annotations

from pathlib import Path

import torch

from src.dpo_config import Config


def _fmt_float(value: float, digits: int = 4) -> str:
    """Format a float for logs with NaN-safe behavior.

    Inputs:
    - value: Numeric value to print.
    - digits: Number of decimal digits for finite values.

    Output:
    - String representation of value (or "nan" for NaN).
    """
    # NaN is the only float for which (x == x) is False.
    if value == value:
        return f"{value:.{digits}f}"
    return "nan"


def _print_section(title: str) -> None:
    """Print a visual section separator and title.

    Input:
    - title: Section name displayed between separator lines.

    Output:
    - None (console side effect only).
    """
    # Build a fixed-width separator line for readability.
    bar = "=" * 96
    print(f"\n{bar}")
    print(title)
    print(bar)


def print_run_configuration(
    cfg: Config,
    device: torch.device,
    train_pair_count: int,
    val_pair_count: int,
    val_1_good_count: int,
    val_1_bad_count: int,
    val_2_good_count: int,
    val_2_bad_count: int,
    dn_eval_count: int,
    train_good_fasta_path: Path,
    train_bad_fasta_path: Path,
    train_csv_mapping_path: Path,
    val_good_fasta_path: Path,
    val_bad_fasta_path: Path,
    val_csv_mapping_path: Path,
) -> None:
    """Print a structured summary of training configuration and dataset sizes.

    Inputs:
    - cfg: Runtime config dataclass.
    - device: Active compute device.
    - train_pair_count, val_pair_count: Number of preference pairs.
    - val_1_good_count, val_1_bad_count: VAE evaluation split sizes.
    - val_2_good_count, val_2_bad_count: Dist25-30 split sizes.
    - dn_eval_count: Number of DN sequences used for likelihood monitoring.
    - *_path arguments: Resolved filesystem paths for train/validation sources.

    Output:
    - None (console side effect only).
    """
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
    print(f"  Reint mode                      : {cfg.reint}")
    print(f"  Lambda Reint                    : {cfg.lambda_reint}")
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
    print(f"  VAE split (good / bad)          : {val_1_good_count} / {val_1_bad_count}")
    print(f"  Dist25-30 split (good / bad)    : {val_2_good_count} / {val_2_bad_count}")


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
    val_1_sep_corr: float,
    val_2_sep_corr: float,
    compute_full_nll_metrics: bool,
    nll_train_good_full: float,
    nll_val_good_full: float,
    nll_val_bad_full: float,
) -> None:
    """Print one metrics report block for baseline or epoch evaluation.

    Inputs:
    - title: Section title (for example BASELINE EVALUATION).
    - epoch_label: Human-readable epoch label.
    - dpo_*: DPO losses (batch/full train/full validation).
    - nll_*: Random-batch and optional full-split NLL metrics.
    - mean_like_model/ref: DN mean token likelihood values.
    - compute_val_separation_metric: Toggle for separation output.
    - val_sep_corr, val_1_sep_corr, val_2_sep_corr: Correlation scores.
    - compute_full_nll_metrics: Toggle for full-split NLL output.

    Output:
    - None (console side effect only).
    """
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
            f"{_fmt_float(val_sep_corr)} / {_fmt_float(val_1_sep_corr)} / {_fmt_float(val_2_sep_corr)}"
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
