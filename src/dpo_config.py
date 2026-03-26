from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


# This dataclass centralizes every runtime setting used by the DPO training pipeline.
# Keeping all defaults here makes experiments reproducible and easy to compare.

@dataclass
class Config:
    # Reproducibility
    seed: int = 1234

    # Data/vocabulary
    max_context: int = 198
    dn_path: str = "data/RF00028_aligned_no_inserts_reweighted_d10.fa"

    # Model checkpoint initialization
    pretrained_ckpt_path: str = "checkpoints/checkpoints_giovanni/RF00028_aligned_GPTTransformer_th10_lr5e-4_batch16_embd64_nhead8_nlayer4_longer_best.pt"

    # DPO train/val pair datasets
    train_good_fasta_path: str = "data/split_train_validation/split_fraction-0.15_add-vae-25-30-and-close-to-validation_nll-filtered_100/DTp_train.fasta"   # split train in funzionali
    train_bad_fasta_path: str = "data/split_train_validation/split_fraction-0.15_add-vae-25-30-and-close-to-validation_nll-filtered_100/DTm_train_after-nll-filter.fasta"    # split train in non funzionali
    train_csv_mapping_path: str = "data/pairings/pair_split_fraction-0.15_add-vae-25-30-and-close-to-validation_nll-filtered_100/pair_train/allpairs_bin_refbin_reciprocal.csv"                            # pairing csv file
    train_dataset_description: str = "Train Dataset (without VAE sequences in 25-30 distance range) filtered by NLL (70 percentile)"                              # titolo del plot che descrive questo dataset

    val_good_fasta_path: str = "data/split_train_validation/split_fraction-0.15_add-vae-25-30-and-close-to-validation_nll-filtered_100/DTp_val_plus-vae25-30-and-close.fasta"  # split test in funzionali
    val_bad_fasta_path: str  = "data/split_train_validation/split_fraction-0.15_add-vae-25-30-and-close-to-validation_nll-filtered_100/DTm_val_plus-vae25-30-and-close.fasta"  # split test in non funzionali
    val_csv_mapping_path: str = "data/pairings/pair_split_fraction-0.15_add-vae-25-30-and-close-to-validation_nll-filtered_100/pair_validation/allpairs_bin_refbin_reciprocal.csv"                               # pairing csv file                           
    val_dataset_description: str = "Validation Dataset (with VAE sequences in 25-30 distance range) filtered by NLL (70 percentile"                              # titolo del plot che descrive questo dataset

    # OPTIONAL Extra datasets used for violin-based separation analysis

    # primo validation set opzionale
    val_1_good_fasta_path: str | None = "data/split_train_validation/split_fraction-0.15_add-vae-25-30-and-close-to-validation_nll-filtered_100/DTp_vae25-30_plus-close-close.fasta"  # funzionali primo validation ulteriore (OPZIONALE)
    val_1_bad_fasta_path: str | None =  "data/split_train_validation/split_fraction-0.15_add-vae-25-30-and-close-to-validation_nll-filtered_100/DTm_vae25-30_plus-close-close.fasta"  # non funzionali primo validation ulteriore (OPZIONALE)
    val_1_dataset_description: str = "VAE sequences in 25-30 distance range and close to them"                                                          # titolo del plot che descrive questo dataset

    # secondo validation set opzionale
    val_2_good_fasta_path: str | None = None # "data/split_data_197/vae_plus_split/DTp_val_VAE_plus_25_30.fasta"
    val_2_bad_fasta_path: str | None = None  # "data/split_data_197/vae_plus_split/DTm_val_VAE_plus_25_30.fasta"
    val_2_dataset_description: str = "Dist 25-30 validation split"  # titolo del plot che descrive questo dataset

    # Output paths
    checkpoint_dir: str = "checkpoints/checkpoints_roberto/split_0.15_vae25-30-and-close-in-test_nll-filter-100/allpairs_bin_refbin_reciprocal/"  # dove salvare i checkpoint del modello durante il training
    image_dir: str = "images_roberto/split_0.15_vae25-30-and-close-in-test_nll-filter-100/allpairs_bin_refbin_reciprocal/"                        # dove salvare le immagini generate durante il training
    history_json_path: str = "images_roberto/split_0.15_vae25-30-and-close-in-test_nll-filter-100/allpairs_bin_refbin_reciprocal/history.json"    # dove salvare il json con la storia di training (loss e metriche ad ogni epoch)

    # Training hyperparameters
    block_size: int = 198
    batch_size: int = 1024
    learning_rate: float = 1e-4
    num_epochs: int = 500
    # Beta coefficient for DPO loss: higher values put more emphasis on correctly classifying the worse sequence in each pair.
    beta: float = 0.3

    # Loss selection: if True use Reint loss instead of DPO loss.
    reint: bool = False
    # Reint coefficient applied to loser reward term.
    lambda_reint: float = 1.0

    # Model architecture
    n_embd: int = 64
    n_head: int = 8
    n_layer: int = 4

    # freeze layers
    layers_to_freeze: int = 0   # how many of the lower (closer to input) transformer layers to freeze during training; 0 means no freezing
    








    ######### NON ESSENTIAL CONFIGS BELOW (mostly for metrics and evaluation settings) - can keep defaults for most experiments, but good to have them here if we want to edit without digging through the code

    # Optimization/acceleration (lascia su True)
    use_torch_compile: bool = True  
    suppress_dynamo_errors: bool = True

    # Metrics switches
    # se True calcola la loss da plottare su tutto i training set e non solo su una batch random
    compute_full_nll_metrics: bool = False
    # Random-batch NLL estimation settings
    nll_rand_n_batches: int = 10
    nll_rand_eval_batch_size_cap: int = 128

    # Pearson correlation between sequence NLL and labels (good=0, bad=1)
    compute_val_separation_metric: bool = True

    # DN likelihood monitoring
    # If "full", compute NLL on the full DN eval set every eval epoch. If "fixed_subsample", compute NLL on a fixed random subsample of the DN eval set every eval epoch. The fixed subsample mode is much faster and should correlate well with the full NLL, as long as the subsample size is large enough.
    dn_likelihood_mode: str = "fixed_subsample"  # "full" or "fixed_subsample"
    dn_fixed_subsample_size: int = 800
    dn_fixed_subsample_seed: int = 1234

    

    # Full-metric batch sizes
    full_eval_batch_size_cap: int = 256

    # Run evaluation and save plots every N training epochs (plus baseline epoch 0).
    eval_every_n_epochs: int = 10

    # Y-limits for distance-binned NLL violin figure.
    distance_nll_ylim_min: float = 0.45
    distance_nll_ylim_max: float = 3.0

    # Scheduler (mirrors notebook behavior)
    scheduler_step_size: int = 10000
    scheduler_gamma: float = 0.1


def config_from_dict(overrides: Mapping[str, Any], base: Config | None = None) -> Config:
    """Build a Config by merging ``overrides`` on top of ``base`` (or defaults)."""
    if not isinstance(overrides, Mapping):
        raise TypeError("Config overrides must be a mapping (dict-like object).")

    field_names = {f.name for f in fields(Config)}
    unknown_keys = sorted(set(overrides) - field_names)
    if unknown_keys:
        raise ValueError(
            "Unknown config keys: "
            + ", ".join(unknown_keys)
            + ". Check field names in src/dpo_config.py"
        )

    base_cfg = base if base is not None else Config()
    merged = {**asdict(base_cfg), **dict(overrides)}
    return Config(**merged)


def load_config_from_json(config_path: str | Path, base: Config | None = None) -> Config:
    """Load a JSON config file and merge it with ``base`` (or defaults)."""
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config JSON must contain an object at top-level.")

    return config_from_dict(raw, base=base)


# Single global configuration instance imported by the training entrypoint.
CONFIG = Config()
