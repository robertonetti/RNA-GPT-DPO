from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # Reproducibility
    seed: int = 1234

    # Data/vocabulary
    max_context: int = 198
    dn_path: str = "DATA/RF00028_aligned_no_inserts_reweighted_d10.fa"

    # Model checkpoint initialization
    pretrained_ckpt_path: str = "../Transformer_giovanni/checkpoints/RF00028_aligned_GPTTransformer_th10_lr5e-4_batch16_embd64_nhead8_nlayer4_longer_best.pt"

    # DPO train/val pair datasets
    train_good_fasta_path: str = "../Group_I_intron-2/DN&DT/split_data_197/DTp_train_split_85_197.fasta"
    train_bad_fasta_path: str = "../Group_I_intron-2/DN&DT/split_data_197/DTm_train_split_85_197.fasta"
    train_csv_mapping_path: str = "../Group_I_intron-2/DN&DT/clustering_methods/train_split_85/top20_bin.csv"

    val_good_fasta_path: str = "../Group_I_intron-2/DN&DT/split_data_197/DTp_test_split_15_197.fasta"
    val_bad_fasta_path: str = "../Group_I_intron-2/DN&DT/split_data_197/DTm_test_split_15_197.fasta"
    val_csv_mapping_path: str = "../Group_I_intron-2/DN&DT/clustering_methods/test_split_15/top20_bin.csv"

    # Extra datasets used for violin-based separation analysis
    vae_good_fasta_path: str = "../Group_I_intron-2/DN&DT/split_data_197/vae_split_25_30/DTp_vae_dist_25_30_197.fasta"
    vae_bad_fasta_path: str = "../Group_I_intron-2/DN&DT/split_data_197/vae_split_25_30/DTm_vae_dist_25_30_197.fasta"

    dist2530_good_fasta_path: str = "../Group_I_intron-2/DN&DT/split_data_197/DTp_test_split_15_dist_25_30_197.fasta"
    dist2530_bad_fasta_path: str = "../Group_I_intron-2/DN&DT/split_data_197/DTm_test_split_15_dist_25_30_197.fasta"

    # Output paths
    checkpoint_dir: str = "checkpoints_dpo_roberto_epochs"
    image_dir: str = "images_dpo_roberto_epochs"
    history_json_path: str = "images_dpo_roberto_epochs/history.json"

    # Training hyperparameters
    block_size: int = 198
    batch_size: int = 1024
    learning_rate: float = 1e-4
    num_epochs: int = 20
    beta: float = 0.3

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

    # Scheduler (mirrors notebook behavior)
    scheduler_step_size: int = 10000
    scheduler_gamma: float = 0.1


CONFIG = Config()
