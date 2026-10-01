#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -p gpu4
#SBATCH -t 08:00:00
#SBATCH -A <HPC_ALLOCATION>
#SBATCH -J SystemB_Final
#SBATCH --array=0-2
#SBATCH -o project/hpc/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913/logs/final_%A_%a.out
#SBATCH -e project/hpc/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913/logs/final_%A_%a.err

set -euo pipefail

EXPERIMENT_DIRECTORY=project/hpc/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913
CONTAINER_IMAGE=/home/admin/singularity/tensorflow.sif
SEEDS=(101 202 303)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

if command -v apptainer >/dev/null 2>&1
then
    CONTAINER_RUNTIME=apptainer
else
    CONTAINER_RUNTIME=singularity
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_NUM_INTEROP_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[JOB] System B final fit seed ${SEED} started on $(hostname) at $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv

"${CONTAINER_RUNTIME}" exec --nv --bind /project,/work,/scratch "${CONTAINER_IMAGE}" \
    python -u "${EXPERIMENT_DIRECTORY}/scripts/train.py" \
    --stage final \
    --model-kind stgnn \
    --graph-key obs_pre2026 \
    --seed "${SEED}"

echo "SYSTEM_B_FINAL_COMPLETE seed=${SEED}"
