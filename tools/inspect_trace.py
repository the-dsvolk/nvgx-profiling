"""Summarise an Nsight Perf SDK Perfetto trace with Perfetto's trace processor.

Usage:
    python -m venv venv && venv/bin/pip install perfetto
    unzip <capture>.perfetto-trace.zip -d x
    venv/bin/python tools/inspect_trace.py x/<capture>.perfetto-trace

Prints slice categories, top GPU kernels by total time, top CUDA driver calls,
API->GPU flow links, kernel-slice argument keys and GPU metric counters.
"""
import sys

from perfetto.trace_processor import TraceProcessor

QUERIES = [
    ("slices by category",
     "select coalesce(category, '(none)') as cat, count(*) as n, round(sum(dur) / 1e6, 1) as total_ms "
     "from slice group by cat order by n desc", 20),
    ("tracks with most slices",
     "select t.name as track, count(*) as n from slice s join track t on s.track_id = t.id "
     "group by t.id order by n desc", 10),
    ("top GPU kernels by total time",
     "select name, count(*) as n, round(sum(dur) / 1e6, 2) as total_ms, round(avg(dur) / 1e3, 1) as avg_us "
     "from slice where category = 'cuda_gpu_kernel_launch' group by name order by sum(dur) desc", 15),
    ("unnamed GPU slices (names containing ??)",
     "select count(*) as n from slice where name like '%??%'", 1),
    ("top CUDA driver calls",
     "select name, count(*) as n, round(sum(dur) / 1e6, 1) as total_ms "
     "from slice where category = 'cuda_driver' group by name order by n desc", 10),
    ("API -> GPU flow links",
     "select count(*) as n from flow", 1),
    ("argument keys on GPU kernel slices",
     "select distinct a.key from slice s join args a on s.arg_set_id = a.arg_set_id "
     "where s.category = 'cuda_gpu_kernel_launch'", 30),
    ("GPU metric counters",
     "select t.name, count(*) as n from counter c join counter_track t on c.track_id = t.id "
     "group by t.id order by n desc", 15),
    ("time span (s)",
     "select round((max(ts + dur) - min(ts)) / 1e9, 1) as seconds from slice", 1),
]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tp = TraceProcessor(trace=sys.argv[1])
    for title, sql, limit in QUERIES:
        print(f"\n=== {title}")
        for i, row in enumerate(tp.query(sql)):
            if i >= limit:
                break
            print("  ", {k: v for k, v in row.__dict__.items()})
    tp.close()


if __name__ == "__main__":
    main()
