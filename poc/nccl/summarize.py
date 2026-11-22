#!/usr/bin/env python3
"""Summarize allreduce_bench.py results into markdown tables.

  summarize.py <results-dir>

Reads <config>-r<round>-r0.json (rank 0) files, e.g. v2312-noprof-r1-r0.json, takes the median over rounds
per message size, and prints:
  1. latency per config (pipelined and synced p50);
  2. NCCL 2.32.6 vs 2.31.2, without and with the profiler plugin;
  3. profiler-plugin overhead per NCCL version (plugin on vs off).
Configs are expected to be named v2312-noprof, v2326-noprof, v2312-prof71, v2326-prof71.
"""
import collections
import glob
import json
import os
import re
import statistics
import sys

CONFIGS = ["v2312-noprof", "v2326-noprof", "v2312-prof71", "v2326-prof71"]


def load(results_dir):
    runs = collections.defaultdict(list)  # config -> [json]
    for path in sorted(glob.glob(os.path.join(results_dir, "*-r0.json"))):
        m = re.match(r"(.+)-r(\d+)-r0\.json$", os.path.basename(path))
        if m:
            with open(path) as f:
                runs[m.group(1)].append(json.load(f))
    return runs


def medians(runs, metric):
    """config -> [(size label, size bytes, median over rounds)]"""
    out = {}
    for cfg, rs in runs.items():
        sizes = [(r["size"], r["size_bytes"]) for r in rs[0]["results"]]
        out[cfg] = [(label, b, statistics.median(run["results"][i][metric] for run in rs))
                    for i, (label, b) in enumerate(sizes)]
    return out


def spread(runs, metric, cfg, i):
    vals = [run["results"][i][metric] for run in runs[cfg]]
    return (max(vals) - min(vals)) / statistics.median(vals) * 100


def pct(a, b):
    return f"{(a / b - 1) * 100:+.1f}%"


def main():
    runs = load(sys.argv[1] if len(sys.argv) > 1 else "results")
    missing = [c for c in CONFIGS if c not in runs]
    if missing:
        sys.exit(f"missing configs: {missing}")
    for c in CONFIGS:
        print(f"<!-- {c}: {len(runs[c])} rounds, NCCL {sorted({r['nccl_version_loaded'] for r in runs[c]})} -->")
    pipe, sync = medians(runs, "pipelined_us"), medians(runs, "synced_p50_us")
    busbw = medians(runs, "busbw_GBps")

    print("\n**Latency, µs (median of rounds): pipelined / synced p50**\n")
    print("| Size | 2.31.2 | 2.32.6 | 2.31.2 + plugin | 2.32.6 + plugin |")
    print("|---|---|---|---|---|")
    for i, (label, _, _) in enumerate(pipe[CONFIGS[0]]):
        cells = [f"{pipe[c][i][2]:.1f} / {sync[c][i][2]:.1f}" for c in CONFIGS]
        print(f"| {label} | " + " | ".join(cells) + " |")

    print("\n**Bus bandwidth, GB/s (median of rounds)**\n")
    print("| Size | 2.31.2 | 2.32.6 | 2.31.2 + plugin | 2.32.6 + plugin |")
    print("|---|---|---|---|---|")
    for i, (label, b, _) in enumerate(busbw[CONFIGS[0]]):
        if b >= 1 << 20:
            print(f"| {label} | " + " | ".join(f"{busbw[c][i][2]:.2f}" for c in CONFIGS) + " |")

    print("\n**2.32.6 vs 2.31.2 (latency change; negative = 2.32.6 faster)**\n")
    print("| Size | No plugin: pipelined | No plugin: synced p50 | Plugin: pipelined | Plugin: synced p50 |")
    print("|---|---|---|---|---|")
    for i, (label, _, _) in enumerate(pipe[CONFIGS[0]]):
        print(f"| {label} | {pct(pipe['v2326-noprof'][i][2], pipe['v2312-noprof'][i][2])} "
              f"| {pct(sync['v2326-noprof'][i][2], sync['v2312-noprof'][i][2])} "
              f"| {pct(pipe['v2326-prof71'][i][2], pipe['v2312-prof71'][i][2])} "
              f"| {pct(sync['v2326-prof71'][i][2], sync['v2312-prof71'][i][2])} |")

    print("\n**Profiler-plugin overhead (plugin on vs off, same NCCL)**\n")
    print("| Size | 2.31.2: pipelined | 2.31.2: synced p50 | 2.32.6: pipelined | 2.32.6: synced p50 |")
    print("|---|---|---|---|---|")
    for i, (label, _, _) in enumerate(pipe[CONFIGS[0]]):
        print(f"| {label} | {pct(pipe['v2312-prof71'][i][2], pipe['v2312-noprof'][i][2])} "
              f"| {pct(sync['v2312-prof71'][i][2], sync['v2312-noprof'][i][2])} "
              f"| {pct(pipe['v2326-prof71'][i][2], pipe['v2326-noprof'][i][2])} "
              f"| {pct(sync['v2326-prof71'][i][2], sync['v2326-noprof'][i][2])} |")

    print("\n**Run-to-run spread, (max - min) / median of pipelined latency**\n")
    print("| Size | " + " | ".join(CONFIGS) + " |")
    print("|---|" + "---|" * len(CONFIGS))
    for i, (label, _, _) in enumerate(pipe[CONFIGS[0]]):
        print(f"| {label} | " + " | ".join(f"{spread(runs, 'pipelined_us', c, i):.1f}%" for c in CONFIGS) + " |")


if __name__ == "__main__":
    main()
