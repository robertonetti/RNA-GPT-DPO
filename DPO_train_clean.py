#!/usr/bin/env python3

import argparse

from src_clean.dpo_config import CONFIG, load_config_from_json
from src_clean.dpo_train import main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the minimal clean DPO/Reint trainer with optional JSON config."
    )
    parser.add_argument(
        "-config",
        "--config",
        dest="config_path",
        type=str,
        default=None,
        help="Path to a JSON config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config_from_json(args.config_path, base=CONFIG) if args.config_path else CONFIG
    main(cfg)
