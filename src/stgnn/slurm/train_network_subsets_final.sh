#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -p gpu4
#SBATCH -t 08:00:00
#SBATCH -A <HPC_ALLOCATION>
#SBATCH -J SysB_BndFinal
#SBATCH --array=0-38%6
#SBATCH -o project/hpc/Experiments/SYSTEM_B_BOUNDARY_INPARISH_20260914/logs/final_%A_%a.out
#SBATCH -e project/hpc/Experiments/SYSTEM_B_BOUNDARY_INPARISH_20260914/logs/final_%A_%a.err
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

set -euo pipefail

EXPERIMENT_DIRECTORY=project/hpc/Experiments/SYSTEM_B_BOUNDARY_INPARISH_20260914
SYSTEM_B_DIRECTORY=project/hpc/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913
CONTAINER_IMAGE=/home/admin/singularity/tensorflow.sif
SEEDS=(101 202 303)
ARM_KEYS=(inparish51 ipsub40_d1 ipsub40_d2 ipsub40_d3 ipsub30_d1 ipsub30_d2 ipsub30_d3 ipsub20_d1 ipsub20_d2 ipsub20_d3 ipsub10_d1 ipsub10_d2 ipsub10_d3)
NODE_COUNTS=(51 40 40 40 30 30 30 20 20 20 10 10 10)

ARM_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
ARM_KEY=${ARM_KEYS[$ARM_INDEX]}
NODE_COUNT=${NODE_COUNTS[$ARM_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

if command -v apptainer >/dev/null 2>&1
then
    CONTAINER_RUNTIME=apptainer
else
    CONTAINER_RUNTIME=singularity
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_NUM_INTEROP_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[JOB] System B boundary final fitting"
echo "[JOB] Arm: ${ARM_KEY}"
echo "[JOB] Nodes: ${NODE_COUNT}"
echo "[JOB] Seed: ${SEED}"
echo "[JOB] Host: $(hostname)"
echo "[JOB] Start: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv

"${CONTAINER_RUNTIME}" exec --nv --bind /project,/work,/scratch "${CONTAINER_IMAGE}" \
    python -u "${REPO_ROOT}/src/stgnn/train_network_subsets.py" \
    --stage final \
    --model-kind stgnn \
    --graph-key "${ARM_KEY}" \
    --development-graph "${EXPERIMENT_DIRECTORY}/frozen_assets/graph_${ARM_KEY}_development.npz" \
    --final-graph "${EXPERIMENT_DIRECTORY}/frozen_assets/graph_${ARM_KEY}_final.npz" \
    --stage-to-rain "${SYSTEM_B_DIRECTORY}/frozen_assets/stage_to_rain_gauge.csv" \
    --models-root "${EXPERIMENT_DIRECTORY}/models" \
    --experiment-name SYSTEM_B_BOUNDARY_INPARISH_20260914 \
    --expected-node-count "${NODE_COUNT}" \
    --seed "${SEED}"

echo "SYSTEM_B_BOUNDARY_FINAL_COMPLETE arm=${ARM_KEY} seed=${SEED}"
