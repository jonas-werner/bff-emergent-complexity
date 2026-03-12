#!/usr/bin/env python3
"""Extract per-seed summary metrics from tape_final files into a CSV."""
import csv, os, re, sys
from pathlib import Path

rows = []
for results_dir in sorted(Path("results").iterdir()):
    if not results_dir.is_dir():
        continue
    name = results_dir.name
    # Parse experiment params from dir name
    for tape_file in sorted(results_dir.glob("tape_final_s*.txt")):
        seed = int(re.search(r"s(\d+)", tape_file.name).group(1))
        lines = tape_file.read_text().strip().split("\n")
        # Line 1: "N interactions"
        interactions = int(lines[0].split()[0].replace(",", ""))
        # Line 2: "N unique tokens, M programs, prog_len=K"
        parts = lines[1].split(",")
        unique_tokens = int(parts[0].strip().split()[0])
        n_programs = int(parts[1].strip().split()[0])
        prog_len = 64  # default
        for p in parts:
            if "prog_len" in p:
                prog_len = int(p.strip().split("=")[1])
        # Count dominant lineage size
        dom_count = 0
        n_lineages = 0
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(\d+):", line)
            if match:
                count = int(match.group(1))
                if count > dom_count:
                    dom_count = count
                n_lineages += 1
        # Determine if transitioned
        transitioned = 1 if unique_tokens < 500 else 0
        rows.append({
            "experiment": name,
            "seed": seed,
            "population": n_programs,
            "prog_len": prog_len,
            "interactions": interactions,
            "unique_tokens": unique_tokens,
            "dominant_lineage_size": dom_count,
            "n_lineages": n_lineages,
            "transitioned": transitioned,
        })

# Write CSV
with open("results_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to results_summary.csv")
# Quick stats
from collections import Counter
for exp in sorted(set(r["experiment"] for r in rows)):
    exp_rows = [r for r in rows if r["experiment"] == exp]
    trans = sum(r["transitioned"] for r in exp_rows)
    print(f"  {exp}: {trans}/{len(exp_rows)} transitioned ({100*trans//len(exp_rows)}%)")
