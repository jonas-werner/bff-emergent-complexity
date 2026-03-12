#!/bin/bash
#SBATCH --job-name=bff-len256
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gpus=0
#SBATCH --time=00:20:00
#SBATCH --array=1-500
#SBATCH --output=logs/len256_%A_%a.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=$HOME/bff-emergent-complexity:$LD_LIBRARY_PATH

cd ~/bff-emergent-complexity
source .venv/bin/activate

SEED=$SLURM_ARRAY_TASK_ID
OUTDIR=results/sweep_len256_${SLURM_ARRAY_JOB_ID}

python3 run_fast.py \
    --seed $SEED \
    --epochs 50000 \
    --population 1024 \
    --prog-len 256 \
    --sample-every 50 \
    --batch 50 \
    --output-dir $OUTDIR
