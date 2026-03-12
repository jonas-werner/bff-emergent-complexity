#!/bin/bash
#SBATCH --job-name=bff-pop1024
#SBATCH --partition=all
#SBATCH --array=1-500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --gpus=0
#SBATCH --time=00:15:00
#SBATCH --output=logs/pop1024_%A_%a.out

cd ~/bff-emergent-complexity
source .venv/bin/activate

export LD_LIBRARY_PATH=$HOME/bff-emergent-complexity:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

OUTDIR="results/sweep2_pop1024_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUTDIR"

python3 run_fast.py \
    --epochs 50000 \
    --population 1024 \
    --prog-len 64 \
    --sample-every 50 \
    --batch 50 \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --output-dir "$OUTDIR"
