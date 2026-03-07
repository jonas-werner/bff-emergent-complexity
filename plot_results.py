#!/usr/bin/env python3
"""
Generate Blaise-style plots from saved .npz run data.

1. Heatmap scatter: ops per interaction vs interaction number (density view)
2. Compressibility (compressed/raw size) vs interaction number
3. Multi-run overlays
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def load_runs(input_dir: str):
    runs = []
    for f in sorted(Path(input_dir).glob("run_s*.npz")):
        d = dict(np.load(f, allow_pickle=True))
        d["path"] = f
        runs.append(d)
    return runs


def plot_single_scatter(run, out_dir: Path):
    """Blaise-style scatter: 2D density heatmap of ops per interaction."""
    steps = run["steps"]  # (epochs, pop_size)
    seed = int(run["seed"])
    pop = int(run["population"])
    max_steps = int(run["max_steps"])
    n_epochs, pop_size = steps.shape
    total = n_epochs * pop_size

    flat = steps.ravel().astype(np.float64)
    x = np.arange(total, dtype=np.float64)

    # 2D histogram: x = interaction number, y = ops
    nx_bins = min(2000, n_epochs)
    ny_bins = 200

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Main heatmap — log-scale color to show the transition from sparse to dense
    h, xedges, yedges = np.histogram2d(
        x, flat,
        bins=[nx_bins, ny_bins],
        range=[[0, total], [0, max_steps]],
    )
    h_masked = np.ma.masked_where(h.T == 0, h.T)
    ax1.pcolormesh(
        xedges, yedges, h_masked,
        norm=mcolors.LogNorm(vmin=1, vmax=h.max()),
        cmap="inferno",
        rasterized=True,
    )
    ax1.set_ylabel("Operations per interaction")
    ax1.set_title(f"BFF Emergent Complexity — seed {seed}, pop {pop}")
    ax1.set_ylim(0, max_steps)

    # Compressibility
    sample_epochs = run["sample_epochs"]
    compress = run["compressibility"]
    sample_x = sample_epochs.astype(float) * pop_size
    ax2.plot(sample_x, compress, color="C3", linewidth=1.5)
    ax2.set_ylabel("Compressibility\n(zip/raw)")
    ax2.set_xlabel("Interaction number")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    path = out_dir / f"scatter_s{seed}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    {path}")


def plot_single_scatter_raw(run, out_dir: Path):
    """Blaise-style scatter: white dots on black background."""
    steps = run["steps"]
    seed = int(run["seed"])
    pop = int(run["population"])
    max_steps = int(run["max_steps"])
    n_epochs, pop_size = steps.shape
    total = n_epochs * pop_size

    max_dots = 2_000_000
    stride = max(1, total // max_dots)

    flat = steps.ravel()[::stride]
    x = np.arange(0, total, stride)

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.scatter(
        x, flat,
        s=0.8, alpha=0.4, color="white",
        rasterized=True, edgecolors="none",
    )
    ax.set_ylabel("Ops", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of interactions", color="white", fontsize=12)
    ax.set_title(f"Computational intensity", color="white",
                 fontsize=16, fontweight="bold", loc="left", pad=12)
    ax.set_ylim(0, max_steps)
    ax.set_xlim(0, total)
    ax.tick_params(colors="white", labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(False)

    plt.tight_layout()
    path = out_dir / f"dots_s{seed}.png"
    plt.savefig(path, dpi=200, facecolor="black")
    plt.close()
    print(f"    {path}")


def plot_single_metrics(run, out_dir: Path):
    """Three-panel line plot: mean ops, HOE, unique tokens."""
    steps = run["steps"]
    seed = int(run["seed"])
    pop = int(run["population"])
    sample_epochs = run["sample_epochs"]
    n_epochs, pop_size = steps.shape

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    mean_per_epoch = steps.mean(axis=1)
    x_interactions = np.arange(n_epochs) * pop_size
    ax1.plot(x_interactions, mean_per_epoch, color="C2", linewidth=0.5, alpha=0.7)
    ax1.set_ylabel("Mean ops/interaction")
    ax1.set_title(f"BFF seed {seed}, pop {pop}")
    ax1.grid(True, alpha=0.2)

    sample_x = sample_epochs.astype(float) * pop_size
    ax2.plot(sample_x, run["hoe"], color="C0", linewidth=1.2)
    ax2.set_ylabel("Higher-order entropy")
    ax2.grid(True, alpha=0.2)

    ax3.plot(sample_x, run["tokens"], color="C1", linewidth=1.2)
    ax3.set_ylabel("Unique tokens")
    ax3.set_xlabel("Interaction number")
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    path = out_dir / f"metrics_s{seed}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    {path}")


def plot_multi_compressibility(runs, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for run in runs:
        seed = int(run["seed"])
        pop = int(run["population"])
        sample_x = run["sample_epochs"].astype(float) * pop
        ax.plot(sample_x, run["compressibility"], linewidth=0.8, alpha=0.6, label=f"s{seed}")
    ax.set_ylabel("Compressibility (zip / raw)")
    ax.set_xlabel("Interaction number")
    ax.set_title(f"BFF compressibility — {len(runs)} runs")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2)
    if len(runs) <= 20:
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = out_dir / "multi_compressibility.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def plot_multi_ops(runs, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for run in runs:
        seed = int(run["seed"])
        pop = int(run["population"])
        steps = run["steps"]
        mean_per_epoch = steps.mean(axis=1)
        x = np.arange(len(mean_per_epoch)) * pop
        ax.plot(x, mean_per_epoch, linewidth=0.5, alpha=0.5, label=f"s{seed}")
    ax.set_ylabel("Mean ops per interaction")
    ax.set_xlabel("Interaction number")
    ax.set_title(f"BFF mean ops — {len(runs)} runs")
    ax.grid(True, alpha=0.2)
    if len(runs) <= 20:
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = out_dir / "multi_ops.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def plot_multi_tokens(runs, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for run in runs:
        seed = int(run["seed"])
        pop = int(run["population"])
        sample_x = run["sample_epochs"].astype(float) * pop
        ax.plot(sample_x, run["tokens"], linewidth=0.8, alpha=0.6, label=f"s{seed}")
    ax.set_ylabel("Unique tokens")
    ax.set_xlabel("Interaction number")
    ax.set_title(f"BFF unique tokens — {len(runs)} runs")
    ax.grid(True, alpha=0.2)
    if len(runs) <= 20:
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = out_dir / "multi_tokens.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def _plot_one_seed(args_tuple):
    """Worker: generate all plots for a single seed."""
    npz_path, out_dir = args_tuple
    run = dict(np.load(npz_path, allow_pickle=True))
    seed = int(run["seed"])
    plot_single_scatter(run, out_dir)
    plot_single_scatter_raw(run, out_dir)
    plot_single_metrics(run, out_dir)
    return seed


def main():
    parser = argparse.ArgumentParser(description="Plot BFF results")
    parser.add_argument("--input-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--parallel", type=int, default=None,
                        help="Max parallel plot workers (default: CPU count)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("run_s*.npz"))
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return

    workers = args.parallel or os.cpu_count() or 1
    print(f"Found {len(npz_files)} run(s). Generating plots in {out_dir}/ ({workers} workers)")

    tasks = [(str(f), out_dir) for f in npz_files]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_plot_one_seed, t): t for t in tasks}
        for future in as_completed(futures):
            seed = future.result()
            print(f"  seed {seed}: done")

    if len(npz_files) > 1:
        print("Multi-run plots:")
        runs = load_runs(args.input_dir)
        plot_multi_compressibility(runs, out_dir)
        plot_multi_ops(runs, out_dir)
        plot_multi_tokens(runs, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
