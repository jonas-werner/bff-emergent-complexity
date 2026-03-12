#!/bin/bash
#SBATCH --job-name=bff-pop16384
#SBATCH --partition=all
#SBATCH --array=1-100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=8G
#SBATCH --gpus=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/pop16384_%A_%a.out

cd ~/bff-emergent-complexity
source .venv/bin/activate
export LD_LIBRARY_PATH=$HOME/bff-emergent-complexity:$LD_LIBRARY_PATH

SEED=${SLURM_ARRAY_TASK_ID}
OUTDIR=results/sweep2_pop16384_${SLURM_ARRAY_JOB_ID}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 run_fast.py \
    --population 16384 \
    --prog-len 64 \
    --epochs 50000 \
    --sample-every 50 \
    --batch 50 \
    --seed ${SEED} \
    --output-dir ${OUTDIR}
