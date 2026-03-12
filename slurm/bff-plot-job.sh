#!/bin/bash
#SBATCH --job-name=bff-plot
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/plot_%j.out

# Usage: sbatch bff-plot-job.sh <results_dir>
# Example: sbatch bff-plot-job.sh results/sweep_pop1024_12345

RESULTS_DIR=${1:?Usage: sbatch bff-plot-job.sh <results_dir>}

cd ~/bff-emergent-complexity
export LD_LIBRARY_PATH=$HOME/bff-emergent-complexity:$LD_LIBRARY_PATH
source .venv/bin/activate

echo "=== Plotting: ${RESULTS_DIR} ==="
echo "Node: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start: $(date)"

python3 plot_results.py --input-dir "${RESULTS_DIR}" --parallel 16

echo "Done: $(date)"
