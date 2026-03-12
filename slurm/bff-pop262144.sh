#!/bin/bash
#SBATCH --job-name=bff-pop262144
#SBATCH --partition=all
#SBATCH --array=1-100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --gpus=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/pop262144_%A_%a.out

cd ~/bff-emergent-complexity
source .venv/bin/activate
export LD_LIBRARY_PATH=$HOME/bff-emergent-complexity:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 run_fast.py \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --population 262144 \
    --prog-len 64 \
    --epochs 50000 \
    --sample-every 50 \
    --batch 10 \
    --output-dir results/sweep2_pop262144_${SLURM_ARRAY_JOB_ID}
