# BFF Emergent Complexity

A replicable implementation of the **Computational Life** experiment by Blaise Agüera y Arcas et al., where self-replicating programs spontaneously emerge from random noise with no fitness function.

**Blog posts:**

This code repo is referenced in two of my blog posts. They contain a lot more detail so please refer to them if you're interested: 

- [BFF - Emergent Complexity experiment](https://jonamiki.com/posts/bff-emergent-complexity-experiment/) — initial results on a desktop PC (5 seeds)
- [BFF at Scale - 3,850 Runs Across Two HPC Clusters](https://jonamiki.com/posts/bff-emergent-complexity-cluster-testing/) — 3,850 experiments across two Slurm clusters with population and program length sweeps

**Paper:** [Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction](https://arxiv.org/abs/2406.19108) (arXiv:2406.19108)


## What this does

A population of programs (default 1,024, each 64 bytes) is initialized with random bytes. Each "interaction" picks two programs, concatenates them into a shared tape, and runs them as Extended Brainfuck (BFF). The tape is then split back into two programs which replace the originals.

After enough interactions, **self-replicating programs emerge** — programs that copy themselves onto their interaction partner. This creates a sharp phase transition visible as:

- A sudden spike in **computational intensity** (operations per interaction)
- A collapse in **unique tokens** (program ancestry diversity)
- A drop in **compressibility** (the soup becomes structured rather than random)

## Setup

Requires Python 3.10+, a C compiler with OpenMP support, and the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Compile the C engine:

```bash
gcc -O3 -march=native -fopenmp -shared -fPIC -o bff_engine.so bff_engine.c
```

Or just use the Makefile:

```bash
make
```

## Quick start

Run a single experiment:

```bash
python run_fast.py --epochs 50000 --seed 42
```

Run multiple experiments in parallel:

```bash
python run_multi.py --runs 5 --epochs 50000
```

Try different program lengths (new):

```bash
python run_fast.py --epochs 50000 --prog-len 32 --seed 1
python run_multi.py --runs 100 --epochs 50000 --prog-len 32
```

Results are saved to timestamped folders under `results/`.

## Plotting

Generate plots from saved run data:

```bash
python plot_results.py --input-dir results/YYYY-MM-DD_HH-MM
```

For large datasets, limit parallel workers to avoid OOM:

```bash
python plot_results.py --input-dir results/my_batch --parallel 8
```

## Cluster results

The `results_summary.csv` file contains per-seed metrics for 3,000 experiments run on a 123-node AMD EPYC 9655P Slurm cluster. The `slurm/` directory has the sbatch scripts used to run them.

To regenerate the CSV from tape dump files:

```bash
python extract_summary.py
```

Key findings from the cluster runs (gelation = dominant lineage controls >= 50% of the population):

**Program length sweep** (pop=1,024, 50K epochs, 500 seeds each):

| Length | Gelation rate | Median dominant % |
|--------|--------------|-------------------|
| 32B    | 95%          | 85.8%             |
| 64B    | 81%          | 72.8%             |
| 128B   | 64%          | 60.4%             |
| 256B   | 31%          | 37.4%             |

**Population size sweep** (prog_len=64, 50K epochs, 100-500 seeds each):

| Pop size | Gelation rate | Median dominant % | Interactions  |
|----------|--------------|-------------------|---------------|
| 1,024    | 78%          | 69.6%             | 51.2M         |
| 4,096    | 31%          | 37.7%             | 204.8M        |
| 16,384   | 47%          | 23.4%             | 819.2M        |
| 65,536   | 98%          | 100.0%            | 3.28B         |
| 262,144  | 96%          | 100.0%            | 13.1B         |

## Files

| File | Description |
|------|-------------|
| `bff_engine.c` | C implementation of the BFF interpreter and simulation loop (OpenMP parallelized) |
| `engine.py` | Python ctypes wrapper around the C engine |
| `run_fast.py` | Run a single experiment with one seed |
| `run_multi.py` | Run multiple experiments in parallel with different seeds |
| `plot_results.py` | Generate matplotlib plots from saved run data |
| `extract_summary.py` | Extract per-seed metrics from tape dump files into CSV |
| `results_summary.csv` | Pre-computed metrics for 3,000 cluster experiment runs |
| `slurm/` | Sbatch scripts used for the HPC cluster experiments |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 6,000 (single) / 16,000 (multi) | Number of epochs (1 epoch = population_size interactions) |
| `--population` | 1,024 | Number of programs in the soup |
| `--prog-len` | 64 | Length of each program in bytes (new) |
| `--max-steps` | 16,384 | Max BFF execution steps per interaction |
| `--seed` | random | Random seed for reproducibility |
| `--sample-every` | 50 (single) / 100 (multi) | Collect metrics every N epochs |
| `--batch` | 50 (single) / 100 (multi) | Epochs per C engine call |
| `--runs` | 10 | Number of independent runs (multi mode) |
| `--parallel` | CPU count | Max concurrent processes (multi mode) |

For a clear phase transition, use at least 30,000-50,000 epochs. With the default settings (pop=1024, prog_len=64), about 78% of runs will produce a dominant replicator within 50,000 epochs.

## Reproducibility

The simulation is deterministic per seed. The C engine uses a xoshiro128** PRNG seeded per run. Given the same seed and single-threaded execution (`OMP_NUM_THREADS=1`), results are identical on any x86 machine. Multi-threaded runs may differ across architectures due to OpenMP scheduling order.

## References

- Agüera y Arcas et al., *Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction*, arXiv:2406.19108 (2024)
- Smoluchowski, M., *Versuch einer mathematischen Theorie der Koagulationskinetik kolloider Lösungen*, Zeitschrift für physikalische Chemie (1917)
- [Reproducing the results of a computational life experiment](https://blog.ricky0123.com/blog/complexity_v2/) (ricky0123)
