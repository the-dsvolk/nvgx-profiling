#!/usr/bin/env python3
"""Compare seeded nvgx_load.py runs (same --seed, so the same regime schedule), e.g. with and without a plugin.

  compare_runs.py --run baseline=load-a.jsonl --run inspector=load-b.jsonl [--fleet-db ~/.nvgx-fleet/fleet.db]

Per run: requests, errors, output tokens/s, and TPOT / TTFT p50 per regime type. With --fleet-db, also the fleet
profiles of the run's time window: GPU busy, NCCL share of kernel time, and NCCL share inside decode / mixed steps.
"""
import argparse
import collections
import json
import os
import sqlite3
import statistics


def load(path):
    reqs, start, stop = [], None, None
    with open(os.path.expanduser(path)) as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == "start":
                start, reqs = r["t"], []  # a log can hold several runs: keep the last one
            elif r["type"] == "stop":
                stop = r["t"]
            elif r["type"] == "req":
                reqs.append(r)
    return reqs, start, stop or (reqs[-1]["t"] if reqs else start)


def fleet(db_path, start, stop):
    db = sqlite3.connect(os.path.expanduser(db_path))
    rows = db.execute("select body from profiles where window_end_ns between ? and ?",
                      (int(start * 1e9), int(stop * 1e9))).fetchall()
    tot = collections.Counter()
    busy = []
    for (body,) in rows:
        a = json.loads(body)["attribution"]
        busy.append(a.get("gpu_busy_pct") or 0)
        tot["kernel"] += a.get("kernel_time_ms") or 0
        tot["nccl"] += (a.get("by_kernel_family_ms") or {}).get("nccl", 0)
        for k in ("decode", "mixed"):
            tot[k] += (a.get("by_phase_ms") or {}).get(k, 0)
            tot[f"nccl_{k}"] += (a.get("by_phase_nccl_ms") or {}).get(k, 0)
    share = lambda a, b: f"{100 * tot[a] / tot[b]:.1f}%" if tot[b] else "–"
    return {"profiles": len(rows), "gpu_busy": f"{statistics.mean(busy):.1f}%" if busy else "–",
            "nccl_share": share("nccl", "kernel"), "nccl_in_decode": share("nccl_decode", "decode"),
            "nccl_in_mixed": share("nccl_mixed", "mixed")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="label=path/to/load.jsonl")
    ap.add_argument("--fleet-db")
    args = ap.parse_args()
    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs[label] = load(path)

    print("| Run | Requests | Errors | Output tokens | Output tok/s |" + (" GPU busy | NCCL share | NCCL in decode steps | NCCL in mixed steps |" if args.fleet_db else ""))
    print("|---|---|---|---|---|" + ("---|---|---|---|" if args.fleet_db else ""))
    for label, (reqs, start, stop) in runs.items():
        ok = [r for r in reqs if r["status"] == "ok"]
        out = sum(r.get("output_tokens") or 0 for r in ok)
        line = f"| {label} | {len(reqs):,} | {len(reqs) - len(ok)} | {out:,} | {out / (stop - start):,.0f} |"
        if args.fleet_db:
            f = fleet(args.fleet_db, start, stop)
            line += f" {f['gpu_busy']} | {f['nccl_share']} | {f['nccl_in_decode']} | {f['nccl_in_mixed']} |"
        print(line)

    regimes = sorted({r["regime"] for reqs, _, _ in runs.values() for r in reqs})
    print("\n**TPOT p50 / TTFT p50 per regime**\n")
    print("| Regime | " + " | ".join(runs) + " |")
    print("|---|" + "---|" * len(runs))
    for g in regimes:
        cells = []
        for reqs, _, _ in runs.values():
            sel = [r for r in reqs if r["regime"] == g and r["status"] == "ok"]
            tp = [r["tpot_ms"] for r in sel if r.get("tpot_ms") is not None]
            tt = [r["ttft_s"] for r in sel if r.get("ttft_s") is not None]
            cells.append(f"{statistics.median(tp):.1f} ms / {1000 * statistics.median(tt):.0f} ms ({len(sel)})" if tp and tt else "–")
        print(f"| {g} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
