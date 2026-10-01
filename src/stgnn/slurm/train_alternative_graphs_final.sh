#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -p gpu4
#SBATCH -t 08:00:00
#SBATCH -A <HPC_ALLOCATION>
#SBATCH -J SystemB_CmpFinal
#SBATCH --array=0-17%6
#SBATCH -o project/hpc/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913/logs/comparison_final_%A_%a.out
#SBATCH -e project/hpc/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913/logs/comparison_final_%A_%a.err

set -euo pipefail

EXPERIMENT_DIRECTORY=project/hpc/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913
GRAPH_DIRECTORY=${EXPERIMENT_DIRECTORY}/frozen_assets/comparisons
CONTAINER_IMAGE=/home/admin/singularity/tensorflow.sif
SEEDS=(101 202 303)
MODEL_KINDS=(stgnn stgnn stgnn stgnn stgnn lstm)
GRAPH_KEYS=(obs_pre2026_basin hecras dem dem_basin identity nodewise)
DEVELOPMENT_FILES=(graph_obs_pre2026_basin_development.npz graph_hecras_development.npz graph_dem_development.npz graph_dem_basin_development.npz graph_identity_development.npz graph_identity_development.npz)
FINAL_FILES=(graph_obs_pre2026_basin_final.npz graph_hecras_final.npz graph_dem_final.npz graph_dem_basin_final.npz graph_identity_final.npz graph_identity_final.npz)

ARM_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
SEED=${SEEDS[$SEED_INDEX]}
MODEL_KIND=${MODEL_KINDS[$ARM_INDEX]}
GRAPH_KEY=${GRAPH_KEYS[$ARM_INDEX]}
DEVELOPMENT_GRAPH=${GRAPH_DIRECTORY}/${DEVELOPMENT_FILES[$ARM_INDEX]}
FINAL_GRAPH=${GRAPH_DIRECTORY}/${FINAL_FILES[$ARM_INDEX]}

if command -v apptainer >/dev/null 2>&1
then
    CONTAINER_RUNTIME=apptainer
else
    CONTAINER_RUNTIME=singularity
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_NUM_INTEROP_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[JOB] System B comparison final fitting started"
echo "[JOB] Model kind: ${MODEL_KIND}"
echo "[JOB] Graph key: ${GRAPH_KEY}"
echo "[JOB] Seed: ${SEED}"
echo "[JOB] Host: $(hostname)"
echo "[JOB] Start: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv

"${CONTAINER_RUNTIME}" exec --nv --bind /project,/work,/scratch "${CONTAINER_IMAGE}" \
    python -u "${EXPERIMENT_DIRECTORY}/scripts/train.py" \
    --stage final \
    --model-kind "${MODEL_KIND}" \
    --graph-key "${GRAPH_KEY}" \
    --development-graph "${DEVELOPMENT_GRAPH}" \
    --final-graph "${FINAL_GRAPH}" \
    --seed "${SEED}"

echo "SYSTEM_B_COMPARISON_FINAL_COMPLETE kind=${MODEL_KIND} graph=${GRAPH_KEY} seed=${SEED}"
