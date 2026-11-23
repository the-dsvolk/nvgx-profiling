# NCCL all-reduce micro-benchmark (two DGX Sparks)

Tools for stage 1 of [`plan/nccl-two-spark.md`](../../plan/nccl-two-spark.md): compare NCCL builds, with and
without a profiler plugin, across two nodes with one GPU each, inside the vLLM container.

| File | What it does |
|---|---|
| [`allreduce_bench.py`](allreduce_bench.py) | One rank of a bf16 all-reduce sweep (4 KB–64 MB): pipelined latency, synced p50/p95, bus bandwidth. Records the NCCL actually in use (`ncclGetVersion`), not torch's build-time version. |
| [`run_bench.sh`](run_bench.sh) | Runs one rank in `vllm/vllm-openai:v0.33.0` over the direct RoCE link. Optional 4th argument: a `libnccl.so.2` to `LD_PRELOAD`; `PROFILER_PLUGIN=` / `PROFILE_EVENT_MASK=` load an NCCL profiler plugin. |
| [`summarize.py`](summarize.py) | Turns the rank-0 JSONs of the four configs into the markdown tables in the plan (medians over rounds, deltas, plugin overhead, spread). |
| [`run_vllm_tp2.sh`](run_vllm_tp2.sh) | vLLM with tensor parallelism 2 across the two DGX Sparks (`mp` backend, `--nnodes 2`), NCCL 2.32.6 on both ranks via `LD_PRELOAD` + `VLLM_NCCL_SO_PATH`. `head` = API server, traced by the Nsight Perf SDK, NVTX step ranges; `worker` = headless. `INSPECTOR=1` loads NCCL Inspector. |
| [`inspector_summary.py`](inspector_summary.py) | NCCL Inspector JSON → per-message-size table (count, exec time p50/p95/p99, bus bandwidth, slow-call share) and a size-band roll-up. |
| [`compare_runs.py`](compare_runs.py) | Compares seeded `nvgx_load.py` runs (tokens/s, TPOT/TTFT per regime) and, with `--fleet-db`, the fleet profiles' NCCL share over each run's window. |

Stage 2 (vLLM TP=2, NCCL in the dashboard, NCCL Inspector) is written up in
[`results/2026-11-22-nccl-tp2/`](../../results/2026-11-22-nccl-tp2/README.md).

Setup and the exact commands are in the plan's [Reproduce](../../plan/nccl-two-spark.md#reproduce) section.
The interface and HCA names in `run_bench.sh` are the DGX Spark's (`enp1s0f0np0` / `enP2p1s0f0np0`,
`rocep1s0f0` / `roceP2p1s0f0`); override them with `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA`.
