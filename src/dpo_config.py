from __future__ import annotations

from dataclasses import dataclass


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
    train_good_fasta_path: str = "data/split_data_197/split_0.15_vae25_30_in_test/DTp_train_split_85_excluding_vae_25_30.fasta"
    train_bad_fasta_path: str = "data/split_data_197/split_0.15_vae25_30_in_test/DTm_train_split_85_excluding_vae_25_30.fasta"
    train_csv_mapping_path: str = "data/clustering_methods/split_0.15_vae25_30_in_train/top5_bin.csv"
    train_dataset_description: str = "Train Dataset (without VAE sequences in 25-30 distance range)"

    val_good_fasta_path: str | None = "data/split_data_197/split_0.15_vae25_30_in_test/DTp_test_split_15_plus_excluded_vae_25_30.fasta"
    val_bad_fasta_path: str | None = "data/split_data_197/split_0.15_vae25_30_in_test/DTm_test_split_15_plus_excluded_vae_25_30.fasta"
    val_csv_mapping_path: str | None = "data/clustering_methods/split_0.15_vae25_30_in_test/top5_bin.csv"
    val_dataset_description: str = "Validation Dataset (with VAE sequences in 25-30 distance range)"

    # Extra datasets used for violin-based separation analysis
    val_1_good_fasta_path: str | None = "data/split_data_197/split_0.15_vae25_30_in_test/DTp_excluded_vae_25_30_added_to_test.fasta"
    val_1_bad_fasta_path: str | None =  "data/split_data_197/split_0.15_vae25_30_in_test/DTm_excluded_vae_25_30_added_to_test.fasta"
    val_1_dataset_description: str = "VAE sequences in 25-30 distance range"

    val_2_good_fasta_path: str | None = None #"data/split_data_197/vae_plus_split/DTp_val_VAE_plus_25_30.fasta"
    val_2_bad_fasta_path: str | None = None #"data/split_data_197/vae_plus_split/DTm_val_VAE_plus_25_30.fasta"
    val_2_dataset_description: str = "Dist 25-30 validation split"

    # Output paths
    checkpoint_dir: str = "checkpoints/checkpoints_roberto/split_0.15_vae25_30_in_test/top5_bin_b03/"
    image_dir: str = "images_roberto/split_0.15_vae25_30_in_test/top5_bin_b03/"
    history_json_path: str = "images_roberto/split_0.15_vae25_30_in_test/top5_bin_b03/history.json"

    # Training hyperparameters
    block_size: int = 198
    batch_size: int = 1024
    learning_rate: float = 1e-4
    num_epochs: int = 500
    beta: float = 0.3

    # Loss selection: if True use Reint loss instead of DPO loss.
    reint: bool = False
    # Reint coefficient applied to loser reward term.
    lambda_reint: float = 1.0

    # Model architecture
    n_embd: int = 64
    n_head: int = 8
    n_layer: int = 4

    # Optimization/acceleration
    layers_to_freeze: int = 0
    use_torch_compile: bool = True
    suppress_dynamo_errors: bool = True

    # Metrics switches
    compute_full_nll_metrics: bool = False
    # Pearson correlation between sequence NLL and labels (good=0, bad=1)
    compute_val_separation_metric: bool = True

    # DN likelihood monitoring
    dn_likelihood_mode: str = "fixed_subsample"  # "full" or "fixed_subsample"
    dn_fixed_subsample_size: int = 800
    dn_fixed_subsample_seed: int = 1234

    # Random-batch NLL estimation settings
    nll_rand_n_batches: int = 10
    nll_rand_eval_batch_size_cap: int = 128

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


# Single global configuration instance imported by the training entrypoint.
CONFIG = Config()
