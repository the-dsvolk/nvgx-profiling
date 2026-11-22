# NCCL all-reduce micro-benchmark (two DGX Sparks)

Tools for stage 1 of [`plan/nccl-two-spark.md`](../../plan/nccl-two-spark.md): compare NCCL builds, with and
without a profiler plugin, across two nodes with one GPU each, inside the vLLM container.

| File | What it does |
|---|---|
| [`allreduce_bench.py`](allreduce_bench.py) | One rank of a bf16 all-reduce sweep (4 KB–64 MB): pipelined latency, synced p50/p95, bus bandwidth. Records the NCCL actually in use (`ncclGetVersion`), not torch's build-time version. |
| [`run_bench.sh`](run_bench.sh) | Runs one rank in `vllm/vllm-openai:v0.33.0` over the direct RoCE link. Optional 4th argument: a `libnccl.so.2` to `LD_PRELOAD`; `PROFILER_PLUGIN=` / `PROFILE_EVENT_MASK=` load an NCCL profiler plugin. |
| [`summarize.py`](summarize.py) | Turns the rank-0 JSONs of the four configs into the markdown tables in the plan (medians over rounds, deltas, plugin overhead, spread). |

Setup and the exact commands are in the plan's [Reproduce](../../plan/nccl-two-spark.md#reproduce) section.
The interface and HCA names in `run_bench.sh` are the DGX Spark's (`enp1s0f0np0` / `enP2p1s0f0np0`,
`rocep1s0f0` / `roceP2p1s0f0`); override them with `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA`.
