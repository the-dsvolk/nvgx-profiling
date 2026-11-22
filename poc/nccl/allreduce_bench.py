#!/usr/bin/env python3
"""Two-node NCCL all-reduce micro-benchmark (one GPU per node), for comparing NCCL builds.

Run the same command on every node (rank 0 on the node whose link IP is --master-addr):
  python3 allreduce_bench.py --rank 0 --world 2 --master-addr <rank0-link-ip> --label nccl-2.31.2 --out r0.json
  python3 allreduce_bench.py --rank 1 --world 2 --master-addr <rank0-link-ip> --label nccl-2.31.2 --out r1.json

Per message size (bf16 tensors) it measures:
  - pipelined latency: N back-to-back all-reduces between two CUDA events, averaged (launch overhead overlapped);
  - synced latency: one all-reduce at a time with a device sync, p50/p95 over N calls (closer to a decode step's
    sequential all-reduces);
  - bus bandwidth = size / pipelined latency * 2(n-1)/n.
Records the NCCL version actually in use (ncclGetVersion via the global symbol scope), so an LD_PRELOAD-ed build is visible;
torch.cuda.nccl.version() only reports the NCCL torch was built with.
"""
import argparse
import datetime
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

SIZES = [4 << 10, 8 << 10, 16 << 10, 32 << 10, 64 << 10, 128 << 10, 256 << 10,
         1 << 20, 4 << 20, 16 << 20, 64 << 20]


def loaded_nccl():
    """Version of the libnccl whose symbols NCCL calls resolve to, plus every libnccl mapped into the process.

    With LD_PRELOAD both the image's and the preloaded library can be mapped; the global symbol scope
    (dlopen(NULL)) resolves to the preloaded one first, as torch's own calls do.
    """
    import ctypes
    with open("/proc/self/maps") as f:
        mapped = sorted({line.split()[-1] for line in f if "libnccl.so" in line})
    code = ctypes.c_int()
    try:
        ctypes.CDLL(None).ncclGetVersion(ctypes.byref(code))
    except AttributeError:  # not in the global scope: fall back to the first mapped copy
        if not mapped:
            return "unknown", mapped
        ctypes.CDLL(mapped[0]).ncclGetVersion(ctypes.byref(code))
    c = code.value  # e.g. 23206 -> 2.32.6
    return f"{c // 10000}.{c % 10000 // 100}.{c % 100}", mapped


def fmt_size(b):
    return f"{b >> 20} MB" if b >= 1 << 20 else f"{b >> 10} KB"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--master-addr", required=True)
    ap.add_argument("--master-port", type=int, default=29511)
    ap.add_argument("--label", default="run")
    ap.add_argument("--repeats", type=int, default=3, help="rounds over all sizes; medians are reported")
    ap.add_argument("--iters-small", type=int, default=400, help="iterations for sizes <= 256 KB")
    ap.add_argument("--iters-large", type=int, default=40, help="iterations for sizes > 256 KB")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"tcp://{args.master_addr}:{args.master_port}",
                            rank=args.rank, world_size=args.world,
                            timeout=datetime.timedelta(seconds=180), device_id=torch.device("cuda:0"))
    # torch.cuda.nccl.version() reports the NCCL torch was built with, not the library actually loaded,
    # so ask the loaded libnccl itself (after a first collective has initialised it).
    dist.all_reduce(torch.ones(1, device="cuda"))
    torch.cuda.synchronize()
    nccl_version, nccl_libs = loaded_nccl()
    tv = torch.cuda.nccl.version()
    torch_nccl_build = ".".join(map(str, tv)) if isinstance(tv, tuple) else str(tv)
    factor = 2 * (args.world - 1) / args.world

    rows = {s: {"pipelined_us": [], "synced_p50_us": [], "synced_p95_us": []} for s in SIZES}
    for _ in range(args.repeats):
        for size in SIZES:
            iters = args.iters_small if size <= 256 << 10 else args.iters_large
            x = torch.ones(size // 2, dtype=torch.bfloat16, device="cuda")
            for _ in range(20):  # warm-up
                dist.all_reduce(x)
            torch.cuda.synchronize()
            dist.barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                dist.all_reduce(x)
            end.record()
            torch.cuda.synchronize()
            rows[size]["pipelined_us"].append(start.elapsed_time(end) * 1000 / iters)
            dist.barrier()
            lat = []
            for _ in range(iters):
                t0 = time.perf_counter()
                dist.all_reduce(x)
                torch.cuda.synchronize()
                lat.append((time.perf_counter() - t0) * 1e6)
            lat.sort()
            rows[size]["synced_p50_us"].append(lat[len(lat) // 2])
            rows[size]["synced_p95_us"].append(lat[int(len(lat) * 0.95) - 1])
            del x

    results = []
    for size in SIZES:
        r = {k: statistics.median(v) for k, v in rows[size].items()}
        busbw = size / (r["pipelined_us"] * 1e-6) / 1e9 * factor
        results.append({"size_bytes": size, "size": fmt_size(size), **{k: round(val, 2) for k, val in r.items()},
                        "busbw_GBps": round(busbw, 3)})
    out = {"label": args.label, "rank": args.rank, "world": args.world, "nccl_version_loaded": nccl_version,
           "nccl_libs_mapped": nccl_libs, "torch_nccl_build": torch_nccl_build, "torch": torch.__version__,
           "repeats": args.repeats,
           "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_") or k in ("LD_PRELOAD",)},
           "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    if args.rank == 0:
        print(f"{args.label}: NCCL {nccl_version}")
        for r in results:
            print(f"  {r['size']:>6}  pipelined {r['pipelined_us']:9.1f} us  synced p50 {r['synced_p50_us']:9.1f} "
                  f"p95 {r['synced_p95_us']:9.1f} us  busbw {r['busbw_GBps']:7.2f} GB/s")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
