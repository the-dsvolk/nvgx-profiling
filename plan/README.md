# Plan: Nsight Perf SDK for AI on DGX Spark (GB10)

The SDK asks for NVIDIA driver **R620+** and CUDA **13.5+**. DGX Spark ships with **R580** and there
is no R620 for DGX Spark yet (see the [main README](../README.md#findings-so-far)). There are two ways
to get the SDK running:

| | [Option A: CUDA 13.5 compat route](option-a-cuda-compat.md) | [Option B: native R620 driver](option-b-native-r620.md) |
|---|---|---|
| What changes | Adds user-space libraries only (`cuda-compat-13-5`, `cuda-toolkit-13-5`) | Replaces the kernel driver (R580 → R620.70.06, built with DKMS) |
| Supported by the SDK | Yes, documented fallback (kernel driver ≥ 580.95.05) | Yes, the primary requirement |
| Supported on DGX Spark | Not named in NVIDIA's forward-compat doc; likely works | No: DGX OS is validated with R580 only |
| Reboot / console | No | Yes: reboot, plus console access for Secure Boot key (MOK) enrollment |
| Main risk | GPU GR-context tracing needs the service on the host-matched `libcuda`; named-kernel GPU timings may be missing | GB10 regressions in a non-validated driver (CUDA graphs, unified memory); OS/kernel updates fight it |
| Rollback | `apt remove`, minutes, no reboot | Reinstall the R580 packages + reboot, ~1 h, may need the console |

## Status

**Option A passed on 2026-11-21** ([results](../results/2026-11-21-option-a-dgx-spark.md)).
Option B isn't needed and is kept for reference only.

## Decision

1. **Start with Option A**, on a DGX Spark set aside for experiments.
2. Move to **Option B only if Option A fails its gate** (below) and the cause is the compat route
   itself, not configuration. Never on a DGX Spark that serves other workloads.
3. **A DGX Spark that serves other workloads stays on stock R580** until DGX OS officially ships R620.
   Option A can be repeated there later: it's additive and reversible.

## Gates

**Option A passes when, on the DGX Spark:**

- the SDK smoke test (`TestNvTraceCudaContinuousProfiling`) passes;
- the service log has no `GPU GR Context Trace provider failed to initialize the RM event buffer`;
- a Perfetto export of a vLLM run shows CUDA API calls, NVTX ranges **and GPU timings for named kernels**;
- tracing overhead on vLLM throughput is small (target < 3% tokens/s).

**Option B is worth trying only if** Option A fails on GR-context tracing or GPU-event correlation
even with the service on the host-matched `libcuda`, or the compat driver doesn't work on GB10.

## Next: NCCL across two DGX Sparks

[`nccl-two-spark.md`](nccl-two-spark.md): profile NCCL once a model runs tensor-parallel over the two Sparks.
Stage 1 (done 2026-11-22) benchmarked NCCL 2.31.2 (the vLLM image's, and NVIDIA's DGX Spark guide version)
against 2.32.6 (profiler-overhead fix), both inside the same container on both nodes, with and without
NCCL's example profiler plugin. Decision: 2.32.6 for the profiling stages, 2.31.2 as the fallback.

## Results log

Record each run in `results/` (date, option, driver/CUDA versions, what passed, what failed, overhead).
