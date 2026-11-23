# nvgx-profiling

**NVidia GPU eXamination: profiling experiments.** The first experiment is the
[NVIDIA Nsight Perf SDK for AI](https://github.com/NVIDIA/nsight-perf-sdk-for-ai) (Beta Preview),
NVIDIA's always-on, low-overhead GPU/CUDA tracing SDK, running on **NVIDIA DGX Spark (GB10)**.

> Personal experiments. Not affiliated with, endorsed by, or supported by NVIDIA.
> NVIDIA, Nsight, DGX and CUDA are trademarks of NVIDIA Corporation.

## Goal

- Get the Nsight Perf SDK for AI service and tools running on a DGX Spark.
- Trace real AI workloads (vLLM serving, PyTorch) and inspect the traces in [Perfetto](https://ui.perfetto.dev).
- Measure overhead and write down what works, what doesn't, and why.

## Status

**2026-11-21: the SDK works on a stock DGX Spark** through the CUDA 13.5 compat route (Option A),
with no driver change, no installed packages and no reboot. The smoke test passes, vLLM traces show
named GPU kernels with timings linked to their CUDA API calls, and overhead was within noise.
Details and learnings: [`results/2026-11-21-option-a-dgx-spark.md`](results/2026-11-21-option-a-dgx-spark.md).

## Showcases

Annotated Perfetto views of real traces (GPU utilization, kernels, CUDA graphs, SQL): [`showcases/`](showcases/README.md).

## Design

Proposal for fleet-wide continuous GPU profiling built on the SDK, going beyond DCGM-style device metrics:
[`design/fleet-profiling.md`](design/fleet-profiling.md).

## Setup

**Hardware:** DGX Spark (GB10, aarch64, 128 GB unified memory) running stock DGX OS. A DGX Spark that
serves other workloads stays untouched; experiments run on a DGX Spark set aside for them.

**Where things run:**

| Piece | Runs on | Notes |
|---|---|---|
| CUDA 13.5 compat driver (`libcuda.so.620.70.06`) | DGX Spark, unpacked in `~/nsight-perf-lab/compat-root` | Not installed; only traced programs load it (`LD_LIBRARY_PATH`) |
| `nvidia-nsight-perf` service | DGX Spark, **as root**, global scope (`/run/nvidia/nsight-perf`) | One-off run with a timeout (`scripts/start-root-service.sh`); loads the system R580 `libcuda` |
| Traced workload | DGX Spark, Docker (`scripts/run-vllm.sh`) | Compat folder + service socket mounted into the container |
| Consumer + exporter | DGX Spark (`nvtrace_streaming_consumer`, `nvtrace_perfetto_trace_exporter`) | Traces stay on the Spark: `.nvtsf` database → `.perfetto-trace.zip` |
| `trace_processor` (Perfetto) | DGX Spark, `127.0.0.1:9001` (`scripts/install-trace-processor.sh`) | Parses and queries big traces natively |
| Perfetto UI | Laptop browser (Chrome), ui.perfetto.dev | Talks to the Spark's `trace_processor` through an SSH tunnel |

```text
DGX Spark                                                                     Laptop
┌───────────────────────────────────────────────────────────────────┐        ┌──────────────────┐
│ vLLM container ──(compat libcuda 620)──► nvidia-nsight-perf (root) │        │                  │
│                                              │                    │        │  Chrome:         │
│                     nvtrace_streaming_consumer ──► .nvtsf database │        │  ui.perfetto.dev │
│                     nvtrace_perfetto_trace_exporter ──► .zip       │        │        ▲         │
│                     trace_processor @ 127.0.0.1:9001 ◄─────────────┼─ SSH ──┼─ 127.0.0.1:9001  │
└───────────────────────────────────────────────────────────────────┘ tunnel └──────────────────┘
```

**Folder layout on the Spark** (`~/nsight-perf-lab`): `compat-root/` (CUDA 13.5 compat driver),
`sdk/` (SDK release; binaries in `sdk/bin`), `tools/` (`trace_processor`), `logs/`, `traces/`,
`bench/`, `venv/` (Perfetto Python package for `tools/inspect_trace.py`).

## Viewing traces

| Trace size | How | Measured on the 70 s vLLM trace (67 MB zip, 301 MB unzipped) |
|---|---|---|
| Small (up to a few hundred MB) | Copy the `.perfetto-trace.zip` to the laptop, open ui.perfetto.dev → **Open trace file** | Loads in the browser's built-in engine (WASM) in ~30 s |
| Big, or to avoid copying | `scripts/view-trace.sh <ssh-host> <trace-path-on-spark>` from the laptop | `trace_processor` on the Spark loads it in 5.6 s (~1.2 GB RAM); the UI only draws |

`view-trace.sh` starts `trace_processor` on the Spark with the trace loaded, tunnels port 9001, and
opens the UI build that matches the `trace_processor` build. Choose **"YES, use loaded trace"**.

- Use the **matched UI build** (`https://ui.perfetto.dev/<version>/`, e.g. `v59.1-1e9c61e98`). The default
  ui.perfetto.dev may be a different build and shows a "Version mismatch" dialog.
- Accelerator mode: one browser tab at a time; Download and sharing are disabled. If the dialog doesn't
  appear (the badge in the bottom-left says WSM instead of RPC), close other Perfetto tabs and reload.
- While the trace loads, SSH prints a few harmless `channel 3: open failed: connect failed` lines.
- For smaller files, export a time window: `nvtrace_perfetto_trace_exporter --start-time-utc-ms ... --end-time-utc-ms ...`.
- Traces embed the host name: keep them out of this repo.

## Requirements vs. DGX Spark (checked 2026-11-21)

| SDK requirement | DGX Spark today | Status |
|---|---|---|
| GPU: Blackwell or Rubin | GB10, compute capability 12.1 (the SDK ships a metric config for it) | ✅ |
| Linux, Ubuntu 24.04 / DGX OS on aarch64 | DGX OS 7.5 / 7.6 (Ubuntu 24.04), kernel 6.17 / 7.0 `-nvidia` | ✅ |
| CMake 3.24+, C++17 compiler (source build) | CMake 3.28, GCC 13.3 | ✅ |
| CUDA Toolkit 13.5+ | 13.0 installed; `cuda-toolkit-13-5` is in NVIDIA's apt repo | ⚠️ install side by side |
| NVIDIA driver R620+ | R580 (580.17x) | ❌ no R620 for DGX Spark yet |
| Fallback: kernel-mode driver ≥ 580.95.05 + CUDA 13.5 compat package | R580 ✅; `cuda-compat-13-5` (user-mode 620.70.06) is in apt | ✅ the route we use, verified 2026-11-21 |
| GPU performance-counter access | Admin only (`RmProfilingAdminOnly=1`) | ⚠️ run the service as root (verified: as a normal user, GPU kernels and counters are silently missing) |

## Findings so far

**No native R620 driver for DGX Spark.** The R620 driver (620.70.06) lists GB10 as a supported GPU,
but NVIDIA validates DGX Spark as a whole stack (DGX OS + NVIDIA's kernel + one driver branch), and
that branch is still R580. NVIDIA says a newer DGX OS is "in the pipeline" with no date
([forum thread](https://forums.developer.nvidia.com/t/r620-driver-and-cuda-13-5/388164)). In practice:

- Ubuntu publishes prebuilt, signed driver modules for the DGX Spark kernels only up to R610/R615;
  nothing exists for R620. R620 would mean building the modules locally with DKMS.
- **Secure Boot is on**, so locally built modules won't load unless you enroll your own signing key
  (MOK) at the console during a reboot, or turn Secure Boot off.
- Newer driver branches have had GB10-specific regressions (reports include CUDA-graph hangs and
  64k-page problems), so our boxes keep a local **apt pin that holds the driver on 580**. The pin
  also blocks a native R620 install through apt.

**So we use the CUDA forward-compatibility route.** `cuda-compat-13-5` only adds user-space
libraries in a separate folder (`/usr/local/cuda-13.5/compat`). It doesn't touch the kernel module,
Secure Boot, the pin or OS updates, and only programs pointed at it use it. An apt dry run installs
it cleanly (one package, nothing removed). Caveats:

- NVIDIA's [forward-compatibility doc](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html)
  lists Data Center GPUs, select RTX server SKUs and Jetson. DGX Spark isn't named, but it works
  there (verified 2026-11-21).
- **Two drivers on one box:** traced workloads load the 13.5 compat driver (`LD_LIBRARY_PATH` pointing
  at the compat folder), while the `nvidia-nsight-perf` service loads the system `libcuda.so` that
  matches the kernel driver. NVIDIA warns GR-context tracing can fail in this split setup; in our run
  it initialised and GPU events correlated fully.
- The R620 user-mode driver brings two known NVTX issues: mixing core and payload push/pop calls,
  and NVTX used on both sides of `fork()` (for PyTorch `DataLoader`, use `spawn` or `forkserver`).
- No GPU power metrics: those need kernel driver R615+, and NVIDIA documents total power only for Rubin.
- Only one tool can own a GPU's performance counters, so Nsight Systems / Nsight Compute can't
  collect counters while the service is running.

**Traces stay on the box.** Consumers attach through a local socket and shared memory. The data is
written to local disk as a Trace Stream Database (`.nvtsf` blocks), as snapshots, or into
DuckDB/Parquet (`nvtrace_continuous_export`). Getting data elsewhere means copying files or
modifying the continuous-export hook.

## Plan

Full plan, gates and rollback steps: [`plan/`](plan/README.md).

- **[Option A: CUDA 13.5 compat route](plan/option-a-cuda-compat.md)** ✅ **passed 2026-11-21**:
  unpack `cuda-compat-13-5` (no install), run the service as root on the host-matched driver,
  smoke test, trace vLLM, measure overhead. User-space only, no reboot.
- **[Option B: native R620 driver](plan/option-b-native-r620.md)** (not needed; kept for reference):
  lift the 580 pin, install R620 with DKMS, enroll a Secure Boot key, run GB10 regression checks.
- **[NCCL across two DGX Sparks](plan/nccl-two-spark.md)** (stage 1 done 2026-11-22): test NCCL 2.32.6
  for its profiler-overhead fix, loaded into the same vLLM container on both Sparks (no host install).
  With NCCL's example profiler plugin, 2.32.6 costs about a third less than 2.31.2; without it they're equal.
  Stage 2 ([results](results/2026-11-22-nccl-tp2/README.md)): vLLM TP=2 across both Sparks; NCCL all-reduce is 25%
  of the traced GPU's kernel time, visible on the dashboard from the SDK alone; NCCL Inspector adds per-size detail at no measurable cost.

Results go in [`results/`](results/). Reusable scripts: [`scripts/`](scripts/); trace queries:
[`tools/inspect_trace.py`](tools/inspect_trace.py).

## Quick start (DGX Spark)

```shell
./scripts/setup-option-a.sh                                         # no root: compat driver + SDK
sudo setsid -f ./scripts/start-root-service.sh "$HOME/nsight-perf-lab"   # root service, 6 h
journalctl -k --since "-5 min" | grep INSUFFICIENT                  # must print nothing
./scripts/run-vllm.sh                                               # traced vLLM test container
~/nsight-perf-lab/sdk/bin/nvtrace_streaming_consumer --output-dir ~/nsight-perf-lab/traces --duration-ms 60000
~/nsight-perf-lab/sdk/bin/nvtrace_perfetto_trace_exporter --tsf-database-dir ~/nsight-perf-lab/traces/<hostname>.<timestamp>
./scripts/install-trace-processor.sh                                # once: Perfetto trace_processor
```

Then, on the laptop:

```shell
./scripts/view-trace.sh <ssh-host> '~/nsight-perf-lab/traces/<hostname>.<timestamp>.perfetto-trace.zip'
```

## Links

- SDK: <https://github.com/NVIDIA/nsight-perf-sdk-for-ai>
  (components: `nvtrace-client`, `nvtrace-tools`, `nvtrace-samples-cuda`, `nsight-perf-daemon`)
- Service docs and Linux install guide: <https://github.com/NVIDIA/nsight-perf-daemon>
- CUDA forward compatibility: <https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html>

## License

[Apache-2.0](LICENSE). The SDK itself and its proprietary binaries are under their own licenses;
nothing from the SDK is redistributed here.
