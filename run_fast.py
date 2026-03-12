#!/usr/bin/env python3
"""
Run BFF experiment using the C engine (bff_engine.so).
Saves raw per-interaction data to .npz for later plotting.
"""

import argparse
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

from engine import CPopulation


def main() -> None:
    parser = argparse.ArgumentParser(description="BFF Emergent Complexity (C engine)")
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--prog-len", type=int, default=64)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**31)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_dir = Path("results") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    pop = CPopulation(
        size=args.population,
        seed=seed,
        max_steps=args.max_steps,
        prog_len=args.prog_len,
    )

    # Capture initial tape state
    init_dump = pop.dump_programs(top_n=30)
    init_path = out_dir / f"tape_initial_s{seed}.txt"
    init_path.write_text(init_dump)
    print(f"Initial tape saved to {init_path}")

    sample_epochs = []
    hoe_list = []
    tokens_list = []
    compressibility_list = []
    all_steps_chunks = []

    print(
        f"seed={seed}, pop={pop.size}, prog_len={args.prog_len}, epochs={args.epochs}, "
        f"max_steps={args.max_steps}, interactions={pop.size * args.epochs:,}"
    )
    print(f"output: {out_dir}")
    print("Epoch\tHOE\tTokens\tCompress\tMean ops\tEpochs/s")
    sys.stdout.flush()

    def sample_metrics(epoch_num: int):
        data = pop.get_values()
        compressed = zlib.compress(data, level=9)
        compress_ratio = len(compressed) / len(data)
        hoe = pop.higher_order_entropy()
        tokens = pop.unique_tokens()
        sample_epochs.append(epoch_num)
        hoe_list.append(hoe)
        tokens_list.append(tokens)
        compressibility_list.append(compress_ratio)
        return hoe, tokens, compress_ratio

    epoch = 0
    t_start = time.monotonic()
    last_log = t_start

    hoe, tokens, cr = sample_metrics(0)
    print(f"0\t{hoe:.4f}\t{tokens}\t{cr:.4f}\t0\t—")
    sys.stdout.flush()

    while epoch < args.epochs:
        chunk = min(args.batch, args.epochs - epoch)
        steps_block = pop.run_epochs(chunk)
        all_steps_chunks.append(steps_block)
        epoch += chunk

        if epoch % args.sample_every == 0 or epoch >= args.epochs:
            hoe, tokens, cr = sample_metrics(epoch)
            mean_ops = float(steps_block.mean())
            now = time.monotonic()
            rate = epoch / (now - t_start) if now > t_start else 0
            if now - last_log > 2.0 or epoch >= args.epochs:
                print(
                    f"{epoch}\t{hoe:.4f}\t{tokens}\t{cr:.4f}\t{mean_ops:.0f}\t{rate:.1f}"
                )
                sys.stdout.flush()
                last_log = now

    elapsed = time.monotonic() - t_start
    all_steps = np.concatenate(all_steps_chunks, axis=0)

    npz_path = out_dir / f"run_s{seed}.npz"
    np.savez_compressed(
        npz_path,
        seed=seed,
        population=args.population,
        prog_len=args.prog_len,
        max_steps=args.max_steps,
        steps=all_steps,
        sample_epochs=np.array(sample_epochs),
        hoe=np.array(hoe_list),
        tokens=np.array(tokens_list, dtype=np.int64),
        compressibility=np.array(compressibility_list),
    )
    print(
        f"Data saved to {npz_path} ({elapsed:.1f}s, {args.epochs / elapsed:.0f} epochs/s)"
    )

    dump = pop.dump_programs(top_n=40)
    dump_path = out_dir / f"tape_final_s{seed}.txt"
    dump_path.write_text(dump)
    print(f"\n--- Final tape (top programs) ---")
    print(dump)


if __name__ == "__main__":
    main()
