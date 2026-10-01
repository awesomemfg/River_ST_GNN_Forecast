#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -p gpu4
#SBATCH -t 08:00:00
#SBATCH -A <HPC_ALLOCATION>
#SBATCH -J LSTM_Train
#SBATCH --array=0-5
#SBATCH -o project/hpc/Experiments/HRRR_FORCING_AND_LSTM_20260911/logs/train_%A_%a.out
#SBATCH -e project/hpc/Experiments/HRRR_FORCING_AND_LSTM_20260911/logs/train_%A_%a.err
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

# Six trainings from ONE frozen matrix: the ST-GNN and the gauge-by-gauge LSTM, seeds 101, 202, 303.
# Research only: nothing is uploaded to the forecast server.
set -euo pipefail

EXP=project/hpc/Experiments/HRRR_FORCING_AND_LSTM_20260911
SIF=/home/admin/singularity/tensorflow.sif
KINDS=(stgnn stgnn stgnn lstm lstm lstm)
SEEDS=(101 202 303 101 202 303)
KIND=${KINDS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

if command -v apptainer >/dev/null 2>&1; then
    CTR=apptainer
else
    CTR=singularity
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_NUM_INTEROP_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[JOB] host $(hostname) task ${SLURM_ARRAY_TASK_ID} kind ${KIND} seed ${SEED} start $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv

"${CTR}" exec --nv --bind /project,/work,/scratch "${SIF}" \
    python -u "${REPO_ROOT}/src/lstm/train.py" "${KIND}" "${SEED}"

echo "[JOB] finished $(date --iso-8601=seconds)"
echo "TRAIN_TASK_DONE ${KIND} ${SEED}"
