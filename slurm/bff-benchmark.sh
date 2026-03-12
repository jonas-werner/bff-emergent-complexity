#!/bin/bash
#SBATCH --job-name=bff-bench
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=32G
#SBATCH --gpus=0
#SBATCH --time=00:15:00
#SBATCH --output=logs/benchmark_%j.out

cd ~/bff-emergent-complexity
export LD_LIBRARY_PATH=$HOME/bff-emergent-complexity:$LD_LIBRARY_PATH
source .venv/bin/activate

echo "=== BFF Benchmark on $(hostname) ==="
echo "Date: $(date)"
echo "CPU: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "Cores: $(nproc)"
echo ""

echo "POP     THREADS  EPOCHS/S  EST_50K_EPOCHS"
echo "------  -------  --------  --------------"

for POP in 1024 4096 16384 65536; do
    for THREADS in 4 16 48 96; do
        OMP_NUM_THREADS=$THREADS python3 -c "
import time
from engine import CPopulation
p = CPopulation(size=$POP, seed=1, prog_len=64)
# Warmup
p.run_epochs(10)
# Timed run
t0 = time.monotonic()
p.run_epochs(100)
dt = time.monotonic() - t0
rate = 100 / dt
est_hours = 50000 / rate / 3600
print(f'$POP\t$THREADS\t{rate:.1f}\t\t{est_hours:.2f}h')
"
    done
done

echo ""
echo "=== Program length benchmark (pop=1024, threads=4) ==="
echo "PROGLEN  EPOCHS/S  EST_50K_EPOCHS"
echo "-------  --------  --------------"

for PROGLEN in 32 64 128 256; do
    OMP_NUM_THREADS=4 python3 -c "
import time
from engine import CPopulation
p = CPopulation(size=1024, seed=1, prog_len=$PROGLEN)
p.run_epochs(10)
t0 = time.monotonic()
p.run_epochs(100)
dt = time.monotonic() - t0
rate = 100 / dt
est_hours = 50000 / rate / 3600
print(f'$PROGLEN\t\t{rate:.1f}\t\t{est_hours:.2f}h')
"
done

echo ""
echo "Benchmark complete: $(date)"
