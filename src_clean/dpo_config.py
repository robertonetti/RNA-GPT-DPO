from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass
class Config:
    seed: int = 1234

    max_context: int = 198
    block_size: int = 198
    dn_path: str = "data/RF00028_aligned_no_inserts_reweighted_d10.fa"
    pretrained_ckpt_path: str = (
        "checkpoints/checkpoints_giovanni/"
        "RF00028_aligned_GPTTransformer_th10_lr5e-4_batch16_embd64_nhead8_nlayer4_longer_best.pt"
    )

    train_good_fasta_path: str = ""
    train_bad_fasta_path: str = ""
    train_csv_mapping_path: str = ""
    val_good_fasta_path: str | None = None
    val_bad_fasta_path: str | None = None
    val_csv_mapping_path: str | None = None
    val_1_good_fasta_path: str | None = None
    val_1_bad_fasta_path: str | None = None

    checkpoint_dir: str = "checkpoints/checkpoints_clean/default"
    image_dir: str = "images_clean/default"
    history_json_path: str = "images_clean/default/history.json"

    batch_size: int = 128
    learning_rate: float = 1e-4
    max_iterations: int = 1000
    beta: float = 0.3

    reint: bool = False
    lambda_reint: float = 1.0
    lambda_kl: float = 0.0

    n_embd: int = 64
    n_head: int = 8
    n_layer: int = 4
    layers_to_freeze: int = 0

    use_torch_compile: bool = True
    suppress_dynamo_errors: bool = True
    eval_every_n_iterations: int = 100
    scheduler_step_size: int = 10000
    scheduler_gamma: float = 0.1

    full_tracking: bool = False
    full_eval_loss: bool = True
    compute_auroc: bool = False
    eval_pair_batch_size: int = 256
    eval_loss_batch_size: int = 256
    metrics_batch_size: int = 256
    dn_eval_subset_size: int | None = 800


def _infer_full_tracking(overrides: Mapping[str, Any]) -> bool:
    if "full_tracking" in overrides:
        return bool(overrides["full_tracking"])

    legacy_flags = [
        "compute_full_nll_metrics",
        "compute_val_separation_metric",
        "compute_val_roc_prc_ppv",
    ]
    return any(bool(overrides.get(key, False)) for key in legacy_flags)


def config_from_dict(overrides: Mapping[str, Any], base: Config | None = None) -> Config:
    if not isinstance(overrides, Mapping):
        raise TypeError("Config overrides must be a mapping (dict-like object).")

    base_cfg = base if base is not None else Config()
    field_names = {field.name for field in fields(Config)}
    merged = asdict(base_cfg)
    merged.update({key: value for key, value in overrides.items() if key in field_names})

    if "full_tracking" not in overrides:
        merged["full_tracking"] = _infer_full_tracking(overrides)

    if "metrics_batch_size" not in overrides and "full_eval_batch_size_cap" in overrides:
        merged["metrics_batch_size"] = int(overrides["full_eval_batch_size_cap"])

    if "eval_pair_batch_size" not in overrides and "nll_rand_eval_batch_size_cap" in overrides:
        merged["eval_pair_batch_size"] = int(overrides["nll_rand_eval_batch_size_cap"])
    if "eval_loss_batch_size" not in overrides:
        if "eval_pair_batch_size" in overrides:
            merged["eval_loss_batch_size"] = int(overrides["eval_pair_batch_size"])
        elif "nll_rand_eval_batch_size_cap" in overrides:
            merged["eval_loss_batch_size"] = int(overrides["nll_rand_eval_batch_size_cap"])

    return Config(**merged)


def load_config_from_json(config_path: str | Path, base: Config | None = None) -> Config:
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config JSON must contain an object at top-level.")

    return config_from_dict(raw, base=base)


CONFIG = Config()
