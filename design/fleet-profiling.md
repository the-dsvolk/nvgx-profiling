# Design: fleet-wide continuous GPU profiling on the Nsight Perf SDK

**Status:** proposal, not implemented (2026-11-21). Builds on the working setup in this repo
([results](../results/2026-11-21-option-a-dgx-spark.md), [showcase](../showcases/2026-11-21-vllm-qwen3-0.6b/README.md)).

## 1. Summary

Collect profiling data continuously on every GPU node with the NVIDIA Nsight Perf SDK for AI, reduce it
on the node into a compact **workload profile** every snapshot (30 s), and send it to a **fleet API**.
Upload full trace snapshots only when something looks wrong.

The point is to go **beyond device metrics** such as DCGM's. DCGM answers *"how busy is each GPU?"*.
This design answers *"which workload, which phase and which kernels use the GPU, and why isn't it faster?"*.
Device health (temperature, power, ECC, XID errors, NVLink) stays with DCGM and is joined in, not rebuilt.

The node side reuses NVIDIA's own exporter, `nvtrace_continuous_export`, and adds one component: a custom
**sink** that receives the SDK's normalized tables and turns each snapshot into a workload profile.

## 2. Problem

Device-level telemetry is coarse in three ways:

- **No attribution.** One time series per GPU; it can't say which process, model phase or kernel used it.
- **No causality.** Metrics share a clock but aren't linked; you see *that* utilization dropped, not *what* stalled.
- **Low time resolution.** Typically scraped every 15–30 s; sub-second bursts average away.

### Example from our own run

vLLM serving Qwen3-0.6B on a DGX Spark ([showcase](../showcases/2026-11-21-vllm-qwen3-0.6b/README.md)):

| Question | Device-level view | What the SDK trace shows |
|---|---|---|
| Is the GPU busy? | Kernel time covers **39.1 s of the 39.5 s** benchmark window, so a "kernel was running" utilization metric (DCGM's GPU utilization field) would read **~99%** (our estimate; DCGM wasn't run) | Yes, but SM instruction utilization averages **3.7%** and Tensor Active **8.6%** |
| Why so little real work? | No answer | Decode runs as ~11.5 ms steps of many tiny kernels (2–50 µs average); the CPU spends most of each step waiting in `cuEventSynchronize`; kernels start up to 10.5 ms after their `cuGraphLaunch` |
| Where does GPU time go? | No answer | GEMM 50.9%, FlashAttention 42.6%, everything else < 3% |
| What do the bursts look like? | Averaged into one 30 s point | Prefill bursts every 3.08 s, each a fraction of a second, Tensor Active peaking at 54.6% |

Device view: "GPU 99% busy". Trace view: "decode is bound by small kernels and synchronization; the model
is too small for this GPU". This design aims to produce the second answer automatically, for every node.

## 3. Goals and non-goals

**Goals**

1. Per-node, always-on collection with overhead in the SDK's < 1% range (measured < 0.5% in our run).
2. A compact per-snapshot **workload profile** with attribution, efficiency, stall and bottleneck signals (§6).
3. Fleet-wide **regression detection** per kernel, across nodes and software versions (§8).
4. **Anomaly-triggered full snapshots** for root cause, viewable in Perfetto (§9).
5. Join with DCGM health data on GPU UUID and time.

**Non-goals (for now)**

- Re-implementing DCGM device health metrics.
- Streaming raw events to the fleet (~4 MB/s per GPU even at our light load).
- CPU call-stack sampling or in-kernel instruction (PC) sampling: not provided by the SDK.
- Cross-node correlation (NCCL, RDMA, network counters): a later phase.

## 4. DCGM vs. this design

| | DCGM (exporter + dashboards) | This design |
|---|---|---|
| Unit of data | Per GPU (some per MIG slice) | Per process, thread, stream, kernel, API call, NVTX range |
| Time resolution | Typically seconds to tens of seconds | Every event with exact timestamps; counters at 1 kHz (up to 20 kHz), summarised per snapshot |
| Causality | None | CPU call → GPU work links; kernels attributed to their CUDA graph launch |
| Workload semantics | None | NVTX phases (prefill/decode, forward/backward), process and container identity |
| Root cause | Shows that something changed | Flight-recorder snapshots of the window around the change |
| Health (thermal, power, clocks, ECC/XID, NVLink) | Core strength | Out of scope: joined from DCGM |
| Hardware / maturity | Broad GPU support, production-grade | Blackwell and Rubin only; SDK is a beta |

## 5. Architecture

```text
GPU node                                                                     Fleet
┌──────────────────────────────────────────────────────────────────┐        ┌───────────────────────────┐
│ workloads ──► nvidia-nsight-perf (root, always on, ring buffers) │        │ ingest API                │
│                         │                                        │        │   /v1/profiles            │
│                         ▼                                        │        │   /v1/snapshots           │
│ nvgx-agent (fork of nvtrace_continuous_export)                   │  HTTPS │ store (columnar)          │
│   snapshot every 30 s ─► TableExporter (SDK, ~30 tables)         │  /OTLP │ regression + anomaly jobs │
│                         ─► TeeSink ─┬─► DuckDbTableSink (local)  │ ─────► │ DCGM join (GPU UUID, time)│
│                                     └─► FleetSink                │        │ UI: profiles, verdicts,   │
│                                          ├ WorkloadProfile ──────┼──────► │     Perfetto links        │
│                                          └ trigger ─► snapshot ──┼──────► │ object storage (snapshots)│
└──────────────────────────────────────────────────────────────────┘        └───────────────────────────┘
```

### 5.1 Why extend NVIDIA's exporter

`nvtrace_continuous_export` (nvtrace-tools, Apache-2.0, ~1,400 lines) already does the hard parts: rolling
history, periodic snapshots, and the SDK's `TableExporter`, which converts raw provider data into
normalized tables with CPU→GPU correlation, CUDA-graph attribution and demangled names. Its output goes to
a **sink**, a C++ template parameter with seven methods:

`Open` · `BeginSnapshotIngestionTransaction` · `AppendRow` · `QueueUpdate` · `CommitSnapshotIngestionTransaction` · `AbortSnapshotIngestionTransaction` · `Close`

Today the only sink is `DuckDbTableSink`. Each snapshot arrives as one transaction, which is exactly the
unit we want to summarise.

| Option | Verdict |
|---|---|
| A. Run the stock exporter, sidecar reads DuckDB | Fastest to try, but DuckDB is single-writer and Parquet appears only at shutdown: awkward for live data |
| **B. Fork the exporter, add `FleetSink` (and a `TeeSink`)** | **Chosen.** Reuses NVIDIA's data model; small, contained change; follows SDK upgrades; could be proposed upstream |
| C. Own consumer on `nvtrace-client` observers | Lowest latency, but re-implements correlation on a beta API |
| D. Batch upload of trace files | Simple, not live |

### 5.2 FleetSink behaviour

- Buffers the rows of the current snapshot transaction in memory (only the columns the profile needs).
- On `Commit…`: computes the `WorkloadProfile` (§6), evaluates triggers (§9), and **hands the result to a
  background sender queue**. It never blocks on the network: each snapshot must finish exporting within
  its interval, or data is lost.
- `QueueUpdate` rows (lifetimes and other facts merged across snapshots) update in-memory resource maps
  (processes, streams, functions); they aren't sent as-is.
- The sender batches, retries with backoff, and drops the oldest profiles if the fleet is unreachable for
  long (bounded memory).

## 6. Workload profile (the data model)

One profile per **(node, GPU UUID, process, snapshot)**, plus one per-GPU roll-up. Source tables are the
SDK's (`NvTraceTableSchema.h`, schema as of `beta-preview-update1`).

| # | Signal group | Contents | Main source tables |
|---|---|---|---|
| 1 | **Attribution** | GPU time by process / container, by NVTX phase, by kernel family, by stream | `cuda_immediate_kernel_launches`, `cuda_graph_node_launches`, `cuda_gpu_operations`, `nvtx_gpu_projection_spans`, `processes` |
| 2 | **Kernel efficiency** | Launches/s, duration p50/p95, share of tiny kernels (< 10 µs), kernels per graph launch, share of work inside CUDA graphs | kernel tables above, `cuda_functions`, `cuda_graph_launches` |
| 3 | **Launch path and stalls** | API call latency, launch→start delay p50/p95, CPU time in sync calls, GPU idle gaps between work items, each attributed to host / sync / copy | `cuda_driver_api_calls`, `cuda_runtime_api_calls`, correlation IDs, kernel and command intervals |
| 4 | **Memory traffic** | Copy bytes and time by direction, count of tiny copies | `cuda_gpu_commands`, `cuda_gpu_memcpy_details` |
| 5 | **Counters, summarised** | Avg / p95 / max per metric, burst count and duration above threshold, per-phase averages | `gpu_pm_samples`, `gpu_metrics` |
| 6 | **Bottleneck verdict** | One label per snapshot (§7) | derived |
| 7 | **Kernel fingerprints** | Per kernel key: count, total time, p50/p95 duration (§8) | kernel tables, `cuda_functions` |
| 8 | **Identity** | GPU UUID and chip, driver and CUDA versions, SDK schema version, process / container | `gpu_partitions`, `cuda_devices`, `processes`, `snapshot_metadata` |

Illustrative profile (values from our vLLM run, abbreviated):

```json
{
  "schema": "nvgx.workload_profile/v0",
  "sdk_schema_version": 0,
  "node": "<node-id>", "gpu_uuid": "GPU-…", "chip": "GB20B",
  "process": {"pid_host": 0, "container": "<id>", "exe": "python3"},
  "window": {"start_utc_ns": 0, "duration_s": 30},
  "attribution": {
    "gpu_busy_pct": 99.0,
    "by_kernel_family_pct": {"gemm": 50.9, "flash_attention": 42.6, "other": 6.5},
    "by_nvtx_phase_pct": {}
  },
  "kernels": {"launches_per_s": 20000, "tiny_kernel_share_pct": 60, "graph_share_pct": 95},
  "launch_path": {"launch_to_start_ms_p50": 4.0, "launch_to_start_ms_p95": 10.5, "cpu_sync_wait_pct": 80},
  "memory": {"h2d_bytes": 0, "d2h_bytes": 0, "tiny_copies": 0},
  "counters": {"sm_util_pct": {"avg": 3.7, "max": 13.3}, "tensor_active_pct": {"avg": 8.6, "max": 54.6}, "bursts": 10},
  "verdict": {"label": "launch_sync_bound", "confidence": 0.8},
  "fingerprints_ref": "batch-…",
  "trigger": null
}
```

Field names and several values above are placeholders to show the shape; only the measured numbers
quoted in §2 are real.

## 7. Bottleneck verdict (initial heuristics)

| Label | Rule of thumb (to be validated on real workloads) |
|---|---|
| `idle` | GPU busy < 10% |
| `launch_sync_bound` | GPU busy high, SM utilization low, many tiny kernels, large CPU sync-wait share |
| `host_bound` | Large GPU idle gaps attributed to the host (CPU not launching fast enough) |
| `memory_bound` | High copy time or memory-throughput counters high while SM utilization is low |
| `compute_bound` | High SM / Tensor utilization with few idle gaps |

Our vLLM run would be labelled `launch_sync_bound`. Thresholds need calibration across several workloads
before they're trusted; the profile always carries the raw numbers behind the label.

## 8. Fleet-wide regression fingerprints

- **Key:** hash of demangled kernel name + grid/block dimensions.
- **Per snapshot:** count, total time, p50/p95 duration per key.
- **Fleet jobs:**
  - *Across nodes, same software:* the same kernel slower on one node points at hardware (clocks, thermals; check the DCGM join).
  - *Across versions, same node class:* slower everywhere after a deploy points at a software regression.
- Kernel names are long; send the hash with a name dictionary that's refreshed when new keys appear.

## 9. Anomaly triggers and snapshots

- **Triggers:** verdict changes, a fingerprint drifts beyond N σ, utilization or throughput drops sharply,
  or an external signal (alert, job failure) calls the node's API.
- **Action:** export that snapshot (and optionally the next) as Perfetto and Parquet, upload to object
  storage, and send the fleet API a pointer with the trigger reason.
- This is the SDK's flight-recorder model (`keepLatest` ring buffers + rolling history), applied fleet-wide.

## 10. Transport and API sketch

- **Profiles:** `POST /v1/profiles` (batched JSON, gzip), or OTLP metrics for the numeric parts so any
  OpenTelemetry backend can chart them (e.g. `nvgx.gpu.time` with `kernel_family`, `phase`, `process` attributes).
- **Snapshots:** `POST /v1/snapshots` with metadata and the object-storage URL.
- **Node control (optional):** `POST /v1/nodes/{id}/capture` to request a snapshot on demand.
- **Every batch carries** the SDK schema version and the profile schema version.
- **Push model:** nodes push when a snapshot is ready; the fleet only receives. Nodes behind NAT or
  firewalls only need outbound connections.
- **Production path: a message bus.** Agents publish profiles to a durable topic (Kafka, Google Pub/Sub,
  NATS JetStream or similar) keyed by node and GPU; independent consumers subscribe: storage writer, live
  dashboard, regression jobs, alerting. That gives buffering when the fleet is slow or down, fan-out and
  replay. Triggered snapshots go to object storage with only a pointer on the bus. The POC posts directly.

## 11. Dashboard and latency

The fleet dashboard updates in near real time; its lag is set mainly by the snapshot interval.

| Snapshot interval | Dashboard lag | Notes |
|---|---|---|
| 30 s (exporter default) | ~30 s + export | Lightest load |
| **5 s (POC setting)** | **~5 s + export** | NVIDIA's exporter warns that intervals below 5 s risk losing data under heavy load, so 5 s is the practical floor for this route |
| < 1 s | needs a direct consumer of the SDK's provider buffers (option C in §5.1), e.g. a counters-only fast lane | Not needed so far |

**Measured in the POC** ([`poc/`](../poc/README.md), 2026-11-22): each 5 s snapshot is exported to DuckDB and
FleetSink in 100–270 ms, and the profile reaches the fleet service **0.06–0.31 s after the window ends**.
So the newest dashboard data is at most ~5.3 s old.

**How it stays live:** the fleet service pushes each new profile to open dashboards over server-sent
events, so charts update without polling or reloading. Grafana on top of the stored profiles (or OTLP
metrics) with a 5 s refresh is the off-the-shelf alternative.

**What the dashboard shows** (beyond device-level metrics):

- stat tiles: bottleneck verdict, GPU busy (kernel time), SM and Tensor utilization, launch → start p95,
  dashboard lag;
- GPU utilization over time: kernel-time "busy" next to SM and tensor-core utilization;
- GPU time by kernel family per snapshot (stacked);
- launch → GPU start delay p50/p95;
- end-to-end lag against the 10 s target;
- table views: latest kernel families, top kernels, recent snapshots.

## 12. Volumes

| Data | Size | Source |
|---|---|---|
| Raw trace stream | ~4 MB/s per GPU at our light load (291 MB per 70 s) | measured |
| Workload profile | a few KB per process per 30 s snapshot | estimate |
| Fingerprints | hundreds of keys × a few numbers per snapshot | estimate |
| Triggered snapshot | tens to hundreds of MB (67 MB zip for 70 s in our run) | measured / scaled |

## 13. Risks and open questions

- **Beta SDK:** the table schema and tool code may change between releases; pin versions and send schema versions.
- **Build:** the fork needs the SDK built from source (CMake). On DGX Spark the existing CUDA 13.0 toolkit
  was enough (verified in the POC).
- **Snapshot time budget:** profile computation must fit well inside the snapshot interval.
- **Permissions:** the service needs root or `CAP_PERFMON`; on DGX Spark today, the CUDA compat route as well.
- **Hardware scope:** Blackwell and Rubin only; mixed fleets need a DCGM-only fallback.
- **NVTX coverage:** phase attribution depends on workloads emitting NVTX (vLLM doesn't by default).
- **Security:** node identity, auth tokens, TLS; kernel names and process info can be sensitive.

## 14. Milestones

| | Milestone | Proves | Status |
|---|---|---|---|
| M0 | **Offline prototype:** compute the workload profile in Python from an existing capture (DuckDB from `nvtrace_continuous_export`, or the Perfetto trace via `trace_processor`) | The signals and verdict are computable and useful, no C++ yet | Skipped: went straight to M1 |
| M1 | `FleetSink` + `TeeSink` in a fork of `nvtrace_continuous_export`, on DGX Spark | Live per-snapshot profiles within the time budget | **POC working** ([`poc/`](../poc/README.md)) |
| M2 | Minimal fleet receiver and one view: GPU time by phase and kernel family over time, with the verdict | End to end | **POC working**: Python stdlib + SQLite + SSE dashboard on a Mac |
| M3 | Triggers and snapshot upload | Root cause in Perfetto from a fleet alert | – |
| M4 | DCGM join | Health + workload in one place | – |
| M5 | NVTX-instrumented workloads (vLLM options, PyTorch training) and a larger model | Phase attribution, verdicts on heavier loads | – |
