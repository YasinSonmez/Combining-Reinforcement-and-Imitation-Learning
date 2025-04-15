#!/bin/bash
#SBATCH --job-name=rlil-train
#SBATCH --account=fc_control
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=2:00:00
#SBATCH --array=0-0
# SBATCH --qos=a5k_gpu4_normal

#SBATCH --output=hopper_expert_1M_TD3BC_%A_%a.out
# SBATCH --mail-type=ALL
# SBATCH --mail-user=yasin_sonmez@berkeley.edu

# minari
python train.py --dataset "mujoco/hopper/expert-v0"

# D4RL
# python train.py --dataset "hopper-expert-v2"