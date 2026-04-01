#!/bin/bash
#SBATCH --job-name=dpo.py
#SBATCH --output=dpo.py_%j.out
#SBATCH --error=dpo.py_%j.err
#SBATCH --nodes=1                         
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100_3g.40gb:1 
#SBATCH --time=1-10:00:00
#SBATCH --cpus-per-task=2

CONFIG_PATH="configs/configs_reint_dkl/config_l=0.json"

python DPO_train.py -config "$CONFIG_PATH"