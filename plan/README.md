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

## Decision

1. **Start with Option A on DGX Spark #2.**
2. Move to **Option B on DGX Spark #2 only if Option A fails its gate** (below) and the cause is the
   compat route itself, not configuration.
3. **DGX Spark #1 stays on stock R580** until DGX OS officially ships R620. If Option A passes on #2
   and is useful, it can be repeated on #1 later (it's additive and reversible).

## Gates

**Option A passes when, on DGX Spark #2:**

- the SDK smoke test (`TestNvTraceCudaContinuousProfiling`) passes;
- the service log has no `GPU GR Context Trace provider failed to initialize the RM event buffer`;
- a Perfetto export of a vLLM run shows CUDA API calls, NVTX ranges **and GPU timings for named kernels**;
- tracing overhead on vLLM throughput is small (target < 3% tokens/s).

**Option B is worth trying only if** Option A fails on GR-context tracing or GPU-event correlation
even with the service on the host-matched `libcuda`, or the compat driver doesn't work on GB10.

## Results log

Record each run in `results/` (date, option, driver/CUDA versions, what passed, what failed, overhead).
