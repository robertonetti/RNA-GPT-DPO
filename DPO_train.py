#!/usr/bin/env python3

"""Compatibility launcher for the src-based training entrypoint."""

from src.dpo_train import main
from src.dpo_config import CONFIG


if __name__ == "__main__":
    main(CONFIG)
