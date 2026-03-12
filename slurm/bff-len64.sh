#!/bin/bash
#SBATCH --job-name=bff-len64
#SBATCH --partition=all
#SBATCH --array=1-500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gpus=0
#SBATCH --time=00:15:00
#SBATCH --output=logs/len64_%A_%a.out

cd ~/bff-emergent-complexity
export LD_LIBRARY_PATH=$HOME/bff-emergent-complexity:$LD_LIBRARY_PATH
source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEED=$SLURM_ARRAY_TASK_ID
OUTDIR="results/sweep_len64_${SLURM_ARRAY_JOB_ID}"

python3 run_fast.py \
    --seed $SEED \
    --epochs 50000 \
    --population 1024 \
    --prog-len 64 \
    --sample-every 50 \
    --batch 50 \
    --output-dir $OUTDIR
