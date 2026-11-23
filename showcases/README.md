# Showcases

Annotated Perfetto views of traces captured with the NVIDIA Nsight Perf SDK for AI on DGX Spark.
Each showcase has its own folder with screenshots, an explanation of every view, and how to reproduce it.

| Date | Showcase | Workload | Highlights |
|---|---|---|---|
| 2026-11-21 | [vLLM serving Qwen3-0.6B](2026-11-21-vllm-qwen3-0.6b/README.md) | vLLM 0.33.0, Qwen3-0.6B bf16, 200 prompts at concurrency 16 | Prefill bursts every 3.08 s; ~11.5 ms decode steps launched as CUDA graphs; GEMM 51% + FlashAttention 43% of GPU time; GPU always busy but SM utilization ~4% |
| 2026-11-22 | [Live fleet dashboard under a variable workload](2026-11-22-live-fleet-dashboard/README.md) (animated time-lapse) | vLLM 0.33.0 + NVTX, Qwen3-0.6B, seeded 30-min `mixed` run: 18 regimes, 5,078 requests | Verdicts and phases follow each regime; saturation shows as 57 ms/token and ~150 ms launch delay; host-bound under light load; unattributed GPU time grows with load (open question) |
| 2026-11-22 | [Tensor parallel across two DGX Sparks, with latency distributions](2026-11-22-tp2-two-sparks/README.md) (animated time-lapse) | vLLM 0.33.0 TP=2 over RoCE, NCCL 2.32.6, Qwen3-0.6B, same seeded 30-min `mixed` run: 6,142 requests | NCCL all-reduce = 23.5% of GPU time; step interval tracks TPOT; under saturation the all-reduce median jumps from 16–32 µs to 64–128 µs and launch delay to 131–262 ms p99; TP=2 beat one Spark in every regime (TPOT −27–33%) |

## Adding a showcase

1. Create `showcases/<YYYY-MM-DD>-<workload>/` with an `img/` folder for screenshots.
2. Write its `README.md`: setup, one section per screenshot (what's on screen, how to read it,
   takeaway), and how to reproduce the views.
3. Add a row to the table above.
