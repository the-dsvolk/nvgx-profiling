#!/usr/bin/env python3
"""Summarize NCCL Inspector JSON output (one record per collective) by message size.

  inspector_summary.py <inspector.log> [--start <unix s>] [--end <unix s>]

Per message size: collectives, collectives per second, execution time p50 / p95 / p99 (GPU-measured,
"kernel_gpu"), bus bandwidth p50, total execution time, and the share of collectives that took more than
5x the size's median (a sign of waiting for the other rank rather than moving data).
--start / --end filter on each record's dump timestamp (within ~1 s of when the collective ran).
"""
import argparse
import collections
import json
import statistics


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(len(vals) * q))] if vals else float("nan")


def fmt_size(b):
    return f"{b / (1 << 20):.1f} MB" if b >= 1 << 20 else f"{b / 1024:.0f} KB" if b >= 1024 else f"{b} B"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    args = ap.parse_args()

    by_size = collections.defaultdict(list)  # size -> [(exec_us, busbw)]
    sources = collections.Counter()
    t_min = t_max = None
    with open(args.log) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            c = r.get("coll_perf")
            if not c:
                continue
            t = r["metadata"]["dump_timestamp_us"] / 1e6
            if (args.start and t < args.start) or (args.end and t > args.end):
                continue
            t_min = t if t_min is None else min(t_min, t)
            t_max = t if t_max is None else max(t_max, t)
            sources[c.get("coll_timing_source")] += 1
            by_size[(c["coll"], c["coll_msg_size_bytes"], f'{c["coll_algo"]}/{c["coll_proto"]}')].append(
                (c["coll_exec_time_us"], c["coll_busbw_gbs"]))
    if not by_size:
        raise SystemExit("no collectives in range")
    span = max(1e-9, t_max - t_min)
    total_n = sum(len(v) for v in by_size.values())
    total_us = sum(e for v in by_size.values() for e, _ in v)
    print(f"{total_n:,} collectives over {span:.0f} s ({total_n / span:,.0f}/s); "
          f"execution time {total_us / 1e6:.1f} s ({100 * total_us / 1e6 / span:.1f}% of wall time); "
          f"timing sources {dict(sources)}\n")
    print("| Collective | Size | Algo/proto | Count | Per s | Exec p50 µs | p95 | p99 | Busbw p50 GB/s | Total exec s | > 5× p50 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = sorted(by_size.items(), key=lambda kv: -sum(e for e, _ in kv[1]))
    for (coll, size, ap_), v in rows[:15]:
        ex = [e for e, _ in v]
        p50 = statistics.median(ex)
        slow = sum(1 for e in ex if e > 5 * max(p50, 1)) / len(ex) * 100
        print(f"| {coll} | {fmt_size(size)} | {ap_} | {len(v):,} | {len(v) / span:,.0f} | {p50:.0f} | {pct(ex, .95):.0f} "
              f"| {pct(ex, .99):.0f} | {statistics.median(b for _, b in v):.2f} | {sum(ex) / 1e6:.2f} | {slow:.1f}% |")
    if len(rows) > 15:
        print(f"\n({len(rows) - 15} more size/algorithm combinations not shown)")

    # Roll-up by size band: decode-sized, mid, prefill-sized (NCCL switches protocol across these), large.
    bands = [("< 64 KB", 0, 64 << 10), ("64 KB – 512 KB", 64 << 10, 512 << 10),
             ("512 KB – 2 MB", 512 << 10, 2 << 20), ("≥ 2 MB", 2 << 20, 1 << 62)]
    print("\n| Size band | Count | Per s | Exec p50 µs | p99 | Busbw p50 GB/s | Total exec s | Share of exec time | Protocols |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, lo, hi in bands:
        sel = [(k, v) for k, v in by_size.items() if lo <= k[1] < hi]
        ex = [e for _, v in sel for e, _ in v]
        if not ex:
            continue
        bw = [b for _, v in sel for _, b in v]
        protos = collections.Counter()
        for k, v in sel:
            protos[k[2]] += len(v)
        top = ", ".join(f"{p} {100 * n / len(ex):.0f}%" for p, n in protos.most_common(2))
        print(f"| {name} | {len(ex):,} | {len(ex) / span:,.0f} | {statistics.median(ex):.0f} | {pct(ex, .99):.0f} "
              f"| {statistics.median(bw):.2f} | {sum(ex) / 1e6:.1f} | {100 * sum(ex) / total_us:.0f}% | {top} |")


if __name__ == "__main__":
    main()
