#!/usr/bin/env python3
"""Per-regime summary of a recorded run: load-generator numbers next to the dashboard's profiles.

  regime_stats.py --recording poc/fleet/static/recordings/<run>.json --load-log <nvgx_load .jsonl>

Per regime type (all instances of it in the run), from the load generator: requests, output tokens/s, TTFT
p50 / p95, TPOT p50. From the recorded profiles (the first 5 s of each regime skipped, since requests still
in flight spill over): GPU busy, decode / mixed shares, NCCL share of kernel time, launch -> start p95, the
most common verdict, and p50 / p99 from the summed latency histograms (log2 buckets, shown as bucket ranges).
"""
import argparse
import collections
import json
import statistics

HIST = {"NCCL kernel": ("nccl_kernel_us", True), "Step interval": ("step_interval_us", True),
        "Launch → start": ("launch_to_start_us", False), "Sync wait": ("sync_wait_us", False)}


def edge(i):
    us = 2 ** i
    return f"{us} µs" if us < 1000 else f"{us / 1000:.1f} ms" if us < 10_000 else f"{round(us / 1000)} ms"


def quant(counts, q):
    tot = sum(counts)
    if not tot:
        return None
    cum = 0
    for i, c in enumerate(counts):
        cum += c
        if cum >= q * tot:
            return i
    return len(counts) - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--load-log", required=True)
    args = ap.parse_args()
    rec = json.load(open(args.recording))

    # Regime timeline and requests from the load log (last run in the file).
    regimes, reqs = [], []
    for line in open(args.load_log):
        r = json.loads(line)
        if r["type"] == "start":
            regimes, reqs = [], []
        elif r["type"] == "regime":
            regimes.append((r["t"], r["regime"], r["duration_s"]))
        elif r["type"] == "req" and r["status"] == "ok":
            reqs.append(r)
    dur = collections.Counter()
    for _, name, d in regimes:
        dur[name] += d

    def regime_at(t):
        cur = None
        for t0, name, d in regimes:
            if t0 + 5 <= t <= t0 + d:
                cur = name
        return cur

    profs = collections.defaultdict(list)
    for p in rec["profiles"]:
        name = regime_at(p["window"]["end_utc_ns"] / 1e9)
        if name:
            profs[name].append(p)

    order = [n for n in ["rag", "chat", "decode", "steady-high", "burst", "trickle", "idle"] if n in dur]
    print("| Regime | Requests | Output tok/s | TTFT p50 / p95 | TPOT p50 | GPU busy | Decode / mixed | NCCL share | Launch → start p95 | Verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in order:
        rs = [r for r in reqs if r["regime"] == name]
        ps = profs.get(name, [])
        out = sum(r.get("output_tokens") or 0 for r in rs)
        tt = sorted(r["ttft_s"] for r in rs if r.get("ttft_s") is not None)
        tp = [r["tpot_ms"] for r in rs if r.get("tpot_ms") is not None]
        busy = statistics.mean(p["attribution"]["gpu_busy_pct"] for p in ps) if ps else None
        ph = collections.Counter()
        nccl = kern = 0.0
        for p in ps:
            a = p["attribution"]
            for k, v in (a.get("by_phase_ms") or {}).items():
                ph[k] += v
            nccl += (a.get("by_kernel_family_ms") or {}).get("nccl", 0)
            kern += a.get("kernel_time_ms") or 0
        tot = sum(ph.values())
        delay = [p["launch_path"]["launch_to_start_ms_p95"] for p in ps if p["launch_path"].get("launch_to_start_ms_p95") is not None]
        verdict = collections.Counter(p["verdict"] for p in ps).most_common(1)
        f = lambda v, fmt: fmt.format(v) if v is not None else "–"
        print(f"| {name} | {len(rs):,} | {out / dur[name]:,.0f} | "
              f"{f(tt[len(tt) // 2] if tt else None, '{:.3f}')} / {f(tt[int(len(tt) * 0.95)] if tt else None, '{:.3f}')} s | "
              f"{f(statistics.median(tp) if tp else None, '{:.1f}')} ms | {f(busy, '{:.0f}%')} | "
              f"{f(100 * ph['decode'] / tot if tot else None, '{:.0f}')} / {f(100 * ph['mixed'] / tot if tot else None, '{:.0f}%')} | "
              f"{f(100 * nccl / kern if kern else None, '{:.0f}%')} | {f(statistics.median(delay) if delay else None, '{:.0f} ms')} | "
              f"{verdict[0][0] if verdict else '–'} |")

    print("\n**Latency distributions per regime (p50 / p99 bucket, from the summed histograms)**\n")
    print("| Regime | " + " | ".join(HIST) + " |")
    print("|---|" + "---|" * len(HIST))
    for name in order:
        cells = []
        for key, by_phase in HIST.values():
            counts = [0] * 21
            for p in profs.get(name, []):
                h = (p.get("histograms") or {}).get(key)
                if h is None:
                    continue
                for arr in (h.values() if by_phase else [h]):
                    counts = [a + b for a, b in zip(counts, arr)]
            q50, q99 = quant(counts, 0.5), quant(counts, 0.99)
            cells.append(f"{edge(q50)}–{edge(q50 + 1)} / {edge(q99)}–{edge(q99 + 1)}" if q50 is not None else "–")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
