#!/bin/bash
#SBATCH --job-name=rlil-train
#SBATCH --account=fc_control
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=12:00:00
#SBATCH --array=1-1
# SBATCH --qos=savio_lowprio

#SBATCH --output=ant_v5_dataset_sweep_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yasin_sonmez@berkeley.edu


RATIOS=(0 0.25 0.5 0.75 1)
TOTAL=25

run_for_index() {
  local idx=$1
  local ratio_idx=$((idx / 5))
  local seed=$((idx % 5))
  local mix_ratio=${RATIOS[$ratio_idx]}

  echo "[Job $SLURM_ARRAY_TASK_ID] Running idx=$idx mix_ratio=$mix_ratio seed=$seed"
  python train_from_buffer.py \
    --algo bc \
    --buffer-path replay_buffers/buffer_base_ant_1.pkl \
    --buffer-path-2 replay_buffers/buffer_base_ant_3.pkl \
    --mix-ratio ${mix_ratio} \
    --env Ant-v5 \
    --seed ${seed}
}

IDX1=$((SLURM_ARRAY_TASK_ID * 2))
IDX2=$((IDX1 + 1))

if [ ${IDX1} -lt ${TOTAL} ]; then
  run_for_index ${IDX1} &
fi

if [ ${IDX2} -lt ${TOTAL} ]; then
  run_for_index ${IDX2} &
fi

wait