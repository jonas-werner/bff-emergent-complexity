#!/bin/bash
#SBATCH --job-name=bff-pop65536
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --gpus=0
#SBATCH --time=01:00:00
#SBATCH --array=1-100
#SBATCH --output=logs/pop65536_%A_%a.out

cd ~/bff-emergent-complexity
source .venv/bin/activate
export LD_LIBRARY_PATH=$HOME/bff-emergent-complexity:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEED=$SLURM_ARRAY_TASK_ID
OUTDIR="results/sweep2_pop65536_${SLURM_ARRAY_JOB_ID}"

python3 run_fast.py \
    --epochs 50000 \
    --population 65536 \
    --prog-len 64 \
    --sample-every 50 \
    --batch 10 \
    --seed $SEED \
    --output-dir $OUTDIR
