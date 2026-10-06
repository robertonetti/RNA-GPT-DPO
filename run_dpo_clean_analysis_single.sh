#!/bin/bash
#SBATCH --job-name=dpo_analysis
#SBATCH --output=dpo_analysis_%j.out
#SBATCH --error=dpo_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --time=3-00:00:00

# Usage: sbatch run_dpo_clean_analysis_single.sh [config_path]
CONFIG_PATH="${1:-configs_clean/configs_new_analysis/configs_split_0.0_vae25-30-in-val_nll-filtered_70/config_allpairs_nt_refbin_reciprocal.json}"

python DPO_train_clean_analysis.py -config "$CONFIG_PATH"
