#!/usr/bin/env python3

"""Compatibility launcher for the precomputed-reference training entrypoint."""

import argparse

from src_precomputed.dpo_train import main
from src_precomputed.dpo_config import CONFIG, load_config_from_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch precomputed-reference DPO training with optional JSON config overrides."
    )
    parser.add_argument(
        "-config",
        "--config",
        dest="config_path",
        type=str,
        default=None,
        help="Path to a JSON config file (overrides defaults in src_precomputed/dpo_config.py).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config_from_json(args.config_path, base=CONFIG) if args.config_path else CONFIG
    if not args.config_path:
        print("Using default configuration.")
    else:
        print(f"Loaded configuration from {args.config_path}:")
        for field in cfg.__dataclass_fields__:
            print(f"  {field}: {getattr(cfg, field)}")

    main(cfg)
