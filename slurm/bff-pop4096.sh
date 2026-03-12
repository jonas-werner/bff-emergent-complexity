#!/bin/bash
#SBATCH --job-name=bff-pop4096
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --gpus=0
#SBATCH --time=00:20:00
#SBATCH --array=1-200
#SBATCH --output=logs/pop4096_%A_%a.out

cd ~/bff-emergent-complexity
source .venv/bin/activate
export LD_LIBRARY_PATH=$HOME/bff-emergent-complexity:$LD_LIBRARY_PATH

SEED=${SLURM_ARRAY_TASK_ID}
OUTDIR=results/sweep2_pop4096_${SLURM_ARRAY_JOB_ID}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 run_fast.py \
    --population 4096 \
    --prog-len 64 \
    --epochs 50000 \
    --sample-every 50 \
    --batch 50 \
    --seed ${SEED} \
    --output-dir ${OUTDIR}
