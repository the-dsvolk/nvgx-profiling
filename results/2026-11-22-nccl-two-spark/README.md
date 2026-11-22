# 2026-11-22: NCCL 2.31.2 vs 2.32.6 across two DGX Sparks

Raw output of stage 1 of [`plan/nccl-two-spark.md`](../../plan/nccl-two-spark.md), where the results and
conclusions are written up.

`raw/<config>-r<round>-r<rank>.json`: one file per rank and round, from
[`poc/nccl/allreduce_bench.py`](../../poc/nccl/allreduce_bench.py). Configs: `v2312` / `v2326` = NCCL 2.31.2 / 2.32.6,
`noprof` = no profiler plugin, `prof71` = NCCL example profiler plugin with `NCCL_PROFILE_EVENT_MASK=71`.
6 rounds per config, interleaved. Re-create the tables with `python3 poc/nccl/summarize.py results/2026-11-22-nccl-two-spark/raw`.
