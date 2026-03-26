# src layout

This folder contains the modular source code for DPO training and all imported model/utilities modules.

## Structure

- `dpo_train.py`: training/evaluation orchestrator.
- `dpo_config.py`: centralized runtime configuration (`Config`, `CONFIG`).
- `dpo_data.py`: FASTA loading, padding/encoding, dataset and pair dataset utilities.
- `dpo_metrics.py`: DPO loss and NLL/likelihood/correlation metrics.
- `dpo_plotting.py`: per-epoch plot generation.
- `dpo_logging.py`: structured stdout logging helpers.
- `transformer.py`: `GPTTransformer` model definition.
- `Transformer_Reint.py`: legacy dataset/training utilities (`load_dataset`, etc.).

## Run

From project root you can run one of the two commands:

- `python3 DPO_train.py -config config.json` (recommended: load parameters from JSON)
- `python3 DPO_train.py` (fallback: use defaults from `src/dpo_config.py`)
- `python3 -m src.DPO_train`

Both use the same configuration in `src/dpo_config.py`.
