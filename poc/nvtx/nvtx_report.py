"""NVTX report for an Nsight Perf SDK Perfetto trace of vLLM with the nvgx_nvtx hook.

Usage:
    unzip <capture>.perfetto-trace.zip -d x
    venv/bin/python poc/nvtx/nvtx_report.py x/<capture>.perfetto-trace

How the SDK's Perfetto exporter represents NVTX:
  * push/pop ranges are slices on the CPU thread track, category "nvtx: <domain>"
    ("nvtx: Default" for the unnamed domain vLLM's own scopes use);
  * the NVTX -> GPU projection is a second set of slices on the GPU track(s), category
    "nvtx_gpu_projection: <domain>", one per run of consecutive GPU operations launched inside the
    range (exported from the nvtx_gpu_projection_spans view), linked to the CPU range by a flow.
This script counts both, then attributes GPU kernel time to step classes by assigning every kernel
slice that starts inside a projection span on the same GPU track to that span's step.
"""

import re
import sys

from perfetto.trace_processor import TraceProcessor

STEP = re.compile(r"prefill_tokens=(\d+) decode_tokens=(\d+)")


def step_class(p: int, d: int) -> str:
    return "mixed" if p and d else ("prefill" if p else "decode")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tp = TraceProcessor(trace=sys.argv[1])

    def rows(sql):
        return list(tp.query(sql))

    print("=== NVTX slices by category and name")
    for r in rows(
        "select category, case when name like 'step %' then 'step ...' else name end as name, "
        "count(*) n, round(sum(dur) / 1e6, 1) ms from slice where category like 'nvtx%' "
        "group by 1, 2 order by n desc"
    ):
        print(f"  {r.category:32s} {r.name:40s} n={r.n:<7d} total_ms={r.ms}")

    print("\n=== sample step messages")
    for r in rows(
        "select name from (select name, row_number() over (partition by (name like 'step prefill_tokens=0 %') "
        "order by ts) rn from slice where category = 'nvtx: nvgx') where rn <= 3"
    ):
        print("  ", r.name)

    flows = rows(
        "select count(*) n from flow f join slice a on f.slice_out = a.id join slice b on f.slice_in = b.id "
        "where a.category = 'nvtx: nvgx' and b.category = 'nvtx_gpu_projection: nvgx'"
    )[0].n
    print(f"\nflows nvgx range -> GPU projection: {flows}")

    steps = rows("select name from slice where category = 'nvtx: nvgx'")
    n_cls = {"prefill": 0, "decode": 0, "mixed": 0}
    tok = {"prefill": 0, "decode": 0}
    for s in steps:
        m = STEP.search(s.name)
        p, d = int(m[1]), int(m[2])
        n_cls[step_class(p, d)] += 1
        tok["prefill"] += p
        tok["decode"] += d
    print(f"steps: {len(steps)} by class {n_cls}; tokens {tok}; distinct messages {len({s.name for s in steps})}")

    tot = rows("select count(*) n, sum(dur) d from slice where category = 'cuda_gpu_kernel_launch'")[0]
    per = rows(
        """
        with p as (select id, ts, dur, track_id, name from slice where category = 'nvtx_gpu_projection: nvgx'),
             k as (select ts, dur, track_id from slice where category = 'cuda_gpu_kernel_launch')
        select p.name, count(k.ts) nk, coalesce(sum(k.dur), 0) kd
        from p left join k on k.track_id = p.track_id and k.ts >= p.ts and k.ts < p.ts + p.dur
        group by p.id
        """
    )
    agg = {c: [0, 0] for c in n_cls}
    for r in per:
        m = STEP.search(r.name)
        c = step_class(int(m[1]), int(m[2]))
        agg[c][0] += r.nk
        agg[c][1] += r.kd
    covered = sum(v[1] for v in agg.values())
    print(f"\n=== GPU kernel time by step class (kernels: {tot.n}, {tot.d / 1e6:.1f} ms)")
    for c, (nk, kd) in agg.items():
        print(f"  {c:8s} kernels={nk:<7d} kernel_ms={kd / 1e6:8.1f}  {kd / tot.d:6.1%}")
    print(f"  {'none':8s} kernels={tot.n - sum(v[0] for v in agg.values()):<7d} "
          f"kernel_ms={(tot.d - covered) / 1e6:8.1f}  {(tot.d - covered) / tot.d:6.1%}  (outside any step range)")
    tp.close()


if __name__ == "__main__":
    main()
