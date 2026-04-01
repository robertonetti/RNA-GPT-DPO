#!/bin/bash
#SBATCH --job-name=dpo_precomputed.py
#SBATCH --output=dpo_precomputed.py_%j.out
#SBATCH --error=dpo_precomputed.py_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --time=1-10:00:00
#SBATCH --cpus-per-task=2

CONFIG_PATH="configs_clean/config_precomputed/configs_split_0.0_vae25-30-in-val_nll-filtered_70/config_allpairs_nt_refbin_reciprocal.json"

python DPO_train_clean.py -config "$CONFIG_PATH"
