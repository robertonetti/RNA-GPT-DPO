#!/bin/bash
#SBATCH --job-name=dpo_clean.py
#SBATCH --output=dpo_clean.py_%j.out
#SBATCH --error=dpo_clean.py_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --time=1-10:00:00

CONFIG_PATH="configs_clean/config_precomputed/configs_split_0.0_vae25-30-in-val_nll-filtered_70/config_allpairs_nt_refbin_reciprocal.json"

python DPO_train_clean.py -config "$CONFIG_PATH"
