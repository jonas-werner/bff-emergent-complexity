#!/usr/bin/env python3
"""
Run multiple BFF experiments with different seeds in parallel.
Each batch gets a timestamped folder under results/.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def run_one_seed(seed: int, args_dict: dict) -> tuple[int, int, str]:
    cmd = [
        sys.executable, "run_fast.py",
        "--epochs", str(args_dict["epochs"]),
        "--population", str(args_dict["population"]),
        "--sample-every", str(args_dict["sample_every"]),
        "--max-steps", str(args_dict["max_steps"]),
        "--batch", str(args_dict["batch"]),
        "--output-dir", str(args_dict["output_dir"]),
        "--seed", str(seed),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args_dict["threads_per_run"])
    result = subprocess.run(
        cmd, cwd=str(Path(__file__).parent), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return seed, result.returncode, result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple BFF seeds in parallel")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=16000)
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: results/YYYY-MM-DD_HH-MM)")
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=None,
                        help="Max concurrent runs (default: CPU count)")
    args = parser.parse_args()

    n_cpus = os.cpu_count() or 1
    parallel = args.parallel or n_cpus
    threads_per_run = max(1, n_cpus // parallel)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_dir = Path("results") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out_dir}")
    print(f"Config: epochs={args.epochs}, pop={args.population}, "
          f"max_steps={args.max_steps}, seeds={args.start_seed}..{args.start_seed + args.runs - 1}")

    seeds_to_run = []
    for i in range(args.runs):
        seed = args.start_seed + i
        npz = out_dir / f"run_s{seed}.npz"
        if npz.exists():
            print(f"  seed {seed}: already exists, skipping")
        else:
            seeds_to_run.append(seed)

    if not seeds_to_run:
        print("All runs already exist.")
        return

    print(f"Running {len(seeds_to_run)} seed(s), {parallel} in parallel, "
          f"{threads_per_run} OMP thread(s) per run")

    args_dict = {
        "epochs": args.epochs, "population": args.population,
        "sample_every": args.sample_every, "max_steps": args.max_steps,
        "batch": args.batch, "output_dir": str(out_dir),
        "threads_per_run": threads_per_run,
    }

    with ProcessPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(run_one_seed, seed, args_dict): seed
            for seed in seeds_to_run
        }
        for future in as_completed(futures):
            seed, rc, output = future.result()
            status = "done" if rc == 0 else f"FAILED (exit {rc})"
            last_lines = [l for l in output.strip().split("\n") if l][-3:]
            print(f"  seed {seed}: {status}")
            for line in last_lines:
                print(f"    {line}")

    print(f"\nAll runs complete.")
    print(f"Plot with: python plot_results.py --input-dir {out_dir}")


if __name__ == "__main__":
    main()
