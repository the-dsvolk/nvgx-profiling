# 2026-11-21: Option A (CUDA 13.5 compat route) on DGX Spark: PASS

**Result:** the Nsight Perf SDK for AI runs on a stock DGX Spark (R580 kernel driver) through the
CUDA 13.5 forward-compatibility libraries. All [gates](../plan/README.md#gates) pass: the smoke test
passes, there's no GR-context error, a vLLM trace has CUDA API calls and **named GPU kernels with
timings, linked to the API calls that launched them**, and tracing overhead is within noise.

## Setup

| | |
|---|---|
| Box | DGX Spark (GB10, aarch64), DGX OS 7.5.0, kernel `6.17.0-1029-nvidia`, Secure Boot on |
| Kernel-mode driver | 580.173.02 (stock, `nvidia-driver-580-open`), unchanged |
| CUDA user-mode driver for traced programs | 620.70.06 from `cuda-compat-13-5`, **unpacked into a home folder, not installed** |
| Service | `nvidia-nsight-perf` from the `beta-preview-update1` aarch64 release, run **as root** (global scope), loading the system `libcuda.so.580.173.02` |
| Workload | `vllm/vllm-openai:v0.33.0` (torch 2.14.0+cu130), `Qwen/Qwen3-0.6B` (rev `c1899de`), `--gpu-memory-utilization 0.25`, CUDA graphs on |
| System changes | None: no packages installed, no driver or pin change, no reboot |

Scripts used: [`scripts/`](../scripts/). Trace queries: [`tools/inspect_trace.py`](../tools/inspect_trace.py).

## Results by step

| Step | Result |
|---|---|
| Compat driver on GB10 | ✅ `cuDriverGetVersion` = 13050 with the compat path (13000 without); a test kernel runs |
| SDK release download | ✅ SHA-256 matches GitHub's published asset digest |
| Service, non-root (user scope) | ⚠️ Starts and says "All required GPU providers are measuring", but has **no counter/HW-trace permission** (see learnings) |
| Smoke test, non-root service | ❌ `gpu_kernels=0` (CUDA API 1164, NVTX 373, memcpy 335 captured) |
| Smoke test, root service | ✅ PASSED: `gpu_kernels=475`, `gpu_pm=13682`, `cuda_api=1164`, `nvtx=376`, `gpu_dtoh_memcpy=335`, 0 unnamed GPU slices |
| vLLM on the compat driver, untraced | ✅ Both vLLM processes load `libcuda.so.620.70.06`; CUDA graph capture (51 piecewise + 35 full) completes; ready in ~145 s |
| vLLM traced (service socket mounted) | ✅ Ready in ~140 s; traced from start |
| 70 s capture during load | ✅ 1.40 M named GPU kernel slices, 186 k CUDA driver calls, 1.41 M API→GPU flow links, ~1 kHz GPU metrics |

## Overhead (vLLM, 200 random prompts, 512 in / 256 out, concurrency 16)

| Run | Output tok/s | Mean TTFT | Mean TPOT |
|---|---|---|---|
| Untraced, run 1 | 1297.6 | 111.9 ms | 11.57 ms |
| Untraced, run 2 | 1292.6 | 106.3 ms | 11.64 ms |
| Traced + streaming capture running | 1295.3 | 110.3 ms | 11.59 ms |
| Traced, no capture attached | 1295.6 | 111.0 ms | 11.59 ms |

The difference is within run-to-run noise (< 0.5%). This is a small model that isn't saturating the GPU;
heavier workloads may show more.

## What's in the vLLM trace

- **GPU kernels** (`cuda_gpu_kernel_launch`, on the "GPU 0" track): FlashAttention
  `flash_fwd_splitkv_kernel`, CUTLASS bf16 GEMMs, `nvjet_sm121_*` GEMMs, `vllm::reshape_and_cache_flash_kernel`,
  FlashInfer top-k sampling, PyTorch elementwise/softmax. Every slice has a demangled name.
- **Per-kernel args:** process and thread ID, `apiCorrelationId` / `hwCorrelationId`, stream, mangled name,
  and CUDA-graph fields (`isGraphLaunch`, `graphExecId`, `graphLaunchApiCorrelationId`), so kernels launched
  inside vLLM's CUDA graphs are attributed too.
- **CUDA driver API** (`cuda_driver`): `cuLaunchKernel` (66 k), `cuLaunchKernelEx` (30 k), event and async
  memcpy calls.
- **GPU metrics** (default `ReducedOverview` set): Tensor Active, SM Instructions (utilization and total),
  GPU Activity, ~70 k samples each over 70 s.
- **NVTX:** none, because vLLM emits no NVTX ranges by default. NVTX tracing itself works (smoke test: 376 ranges).
- The process appears as "Process 144", its PID inside the container.

## Sizes and costs

- Capture: 70 s → 291 MB of `.nvtsf` blocks (~4 MB/s at this load).
- Perfetto export: 6 s wall time, ~1.8 GB RAM, 67 MB `.perfetto-trace.zip` (301 MB unzipped).

## Learnings

1. **The service must run as root (or with `CAP_PERFMON`) on DGX Spark.** Counter profiling is admin-only
   (`RmProfilingAdminOnly=1`). As a normal user the service starts, the readiness check says "All required
   GPU providers are measuring", and the service log is clean, but the trace has no GPU kernels and no
   counter data. **The only sign is in the kernel log:**
   `NV_ERR_INSUFFICIENT_PERMISSIONS ... NVB0CC_CTRL_CMD_INTERNAL_ALLOC_PMA_STREAM`. Check
   `journalctl -k | grep INSUFFICIENT` after starting the service.
2. **The compat driver is required, not optional.** The SDK's CUDA programs are built with CUDA 13.5; on
   the stock R580 user-mode driver they fail with "the provided PTX was compiled with an unsupported
   toolchain". Traced workloads must load the 13.5 compat `libcuda`.
3. **Nothing needs installing.** `apt-get download cuda-compat-13-5` + `dpkg-deb -x` gives the compat
   libraries in a home folder, and the prebuilt aarch64 SDK release needs no CUDA 13.5 toolkit. Root is
   needed only to run the service.
4. **The smoke test starts its own service if none is running** (`Timeout: Failed to connect ... Creating
   NVIDIA Nsight Perf service`), as the current user. So a smoke test can run "without" your root service
   and fail for permission reasons. Start the root service first and check the test log says "Connected".
5. **Starting a root service over SSH:** `sudo -b` died with the SSH terminal before the service started.
   `sudo setsid -f <script>` (plus `trap "" HUP` in the script) works.
6. **Containers:** mount the service's socket folder (`/run/nvidia/nsight-perf`) and the compat folder, and
   put the compat folder first in `LD_LIBRARY_PATH` (keep the image's own entries after it). The vLLM
   image (CUDA 13.0 build) runs unchanged on the 620 user-mode driver, CUDA graphs included. The container
   must start **after** the service.
7. **The split-driver caveat didn't bite.** With the service on the host-matched `libcuda.so.580.173.02`
   and workloads on the 620 compat driver, GR-context tracing initialised and GPU events correlated fully.
8. **Traces embed the host name** (database folder name, Perfetto track names, manifest). Scrub before
   sharing.
9. At root-service start the kernel log showed one
   `NVRM: refcntRequestReference_IMPL: Failed to enter state 1 (status 0x56)` line with no permission error;
   it had no visible effect.
10. Option B (native R620) isn't needed for the SDK on DGX Spark.

## Not done yet

- NVTX ranges from a real workload (vLLM's NVTX options, or a PyTorch script with `torch.cuda.nvtx`).
- A large NVFP4 model that fills the GPU, and PM-counter sampling-rate sweeps.
- The installed systemd service (`install_nvidia_nsight_perf.sh`) instead of a one-off root run.
- `nvtrace_continuous_export` (DuckDB/Parquet) and the snapshot consumer.
