# POC: live fleet dashboard from the Nsight Perf SDK

Proof of concept for [design/fleet-profiling.md](../design/fleet-profiling.md): a GPU node streams a
**workload profile every 5 s** to a fleet service, which shows it on a live dashboard
**~5 s behind** (5 s window + well under 1 s of export and transfer).

```text
DGX Spark                                                        Mac (or any always-on box)
nvidia-nsight-perf (root)                                        nvgx_fleet.py  127.0.0.1:9100
   ▲ traced workloads                                              ├ POST /v1/profiles → SQLite
nvtrace_continuous_export (forked) --snapshot-ms 5000              ├ GET  /events (server-sent events)
   └ TeeSink ─┬─ DuckDbTableSink (local, unchanged)                └ GET  /  dashboard
              └─ FleetSink ── POST every snapshot ── SSH tunnel ──►
```

| Piece | Where | What it is |
|---|---|---|
| [`agent/NvgxFleetSink.h`](agent/NvgxFleetSink.h) | GPU node | `FleetSink`: a table sink with the same 7 methods as NVIDIA's DuckDB sink. Aggregates each snapshot's rows into a workload profile (kernel time by family, tiny-kernel share, CUDA-graph share, launch → GPU start delay, CPU sync wait, memcpy, GPU counters, verdict) and POSTs it from a background thread with a bounded queue. `TeeSink` feeds DuckDB and FleetSink together. |
| [`agent/apply_fleet_patch.py`](agent/apply_fleet_patch.py) | GPU node | Applies 9 anchored edits to NVIDIA's `nvtrace_continuous_export.cpp` (SDK `beta-preview-update1`): include the header, add `--fleet-url` / `--fleet-node`, export through the TeeSink. |
| [`fleet/nvgx_fleet.py`](fleet/nvgx_fleet.py) | Mac | Fleet service, Python standard library only: receive profiles, store in SQLite (`~/.nvgx-fleet/fleet.db`), push to browsers over server-sent events, serve the dashboard. |
| [`fleet/static/`](fleet/static/) | Mac | Dashboard: stat tiles, GPU utilization, GPU time by kernel family, launch delay, lag, tables. Vanilla JS + SVG, light and dark mode. |

The POC posts straight to the fleet service. In production a message bus (Kafka, Pub/Sub, NATS) sits in
between; see the design doc.

## Run it

**On the GPU node (DGX Spark)**, after [Option A](../plan/option-a-cuda-compat.md) setup:

```shell
# 1. Build the forked exporter (~3 min, CUDA 13.0 toolkit is enough)
mkdir -p ~/nsight-perf-lab/fork && cd ~/nsight-perf-lab/fork
curl -fsSLO https://github.com/NVIDIA/nsight-perf-sdk-for-ai/releases/download/beta-preview-update1/nsight-perf-sdk-for-ai-beta-preview-update1-src.tar.gz
tar -xzf nsight-perf-sdk-for-ai-beta-preview-update1-src.tar.gz
python3 <repo>/poc/agent/apply_fleet_patch.py nsight-perf-sdk-for-ai-beta-preview
cmake -S nsight-perf-sdk-for-ai-beta-preview -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --target nvtrace_continuous_export -j

# 2. Root service (see scripts/start-root-service.sh), then the traced workload (scripts/run-vllm.sh)

# 3. Stream 5 s snapshots to the fleet (the tunnel below makes 127.0.0.1:9100 reach the Mac)
build/bin/nvtrace_continuous_export --snapshot-ms 5000 --stop-on-enter-key false \
  --output-dir ~/nsight-perf-lab/poc/live --fleet-url http://127.0.0.1:9100
```

**On the Mac:**

```shell
python3 poc/fleet/nvgx_fleet.py                        # fleet service + dashboard on 127.0.0.1:9100
ssh -N -R 9100:127.0.0.1:9100 <gpu-node>               # lets the node reach it
open http://127.0.0.1:9100                             # dashboard
```

## Measured (2026-11-22)

- Exporter: each 5 s snapshot exported to DuckDB **and** FleetSink in 100–270 ms (budget: 5 s).
- Lag from the end of a snapshot window to the profile arriving on the Mac: **0.06–0.31 s**.
- Without the root service the profiles carry CUDA API and memcpy data but no kernel timings or counters
  (the same permission limit as in the [results](../results/2026-11-21-option-a-dgx-spark.md)).

## Limits (POC)

- One GPU per node and no per-process breakdown yet; plain HTTP, no auth (localhost + SSH tunnel only).
- GPU busy % sums kernel durations (capped at 100%); overlapping kernels on several streams overcount.
- Rows whose GPU times land in the next snapshot are completed from one snapshot of history; older ones are dropped.
- Verdict thresholds are the design doc's initial heuristics, not calibrated.
