# Showcases

Annotated Perfetto views of traces captured with the NVIDIA Nsight Perf SDK for AI on DGX Spark.
Each showcase has its own folder with screenshots, an explanation of every view, and how to reproduce it.

| Date | Showcase | Workload | Highlights |
|---|---|---|---|
| 2026-11-21 | [vLLM serving Qwen3-0.6B](2026-11-21-vllm-qwen3-0.6b/README.md) | vLLM 0.33.0, Qwen3-0.6B bf16, 200 prompts at concurrency 16 | Prefill bursts every 3.08 s; ~11.5 ms decode steps launched as CUDA graphs; GEMM 51% + FlashAttention 43% of GPU time; GPU always busy but SM utilization ~4% |

## Adding a showcase

1. Create `showcases/<YYYY-MM-DD>-<workload>/` with an `img/` folder for screenshots.
2. Write its `README.md`: setup, one section per screenshot (what's on screen, how to read it,
   takeaway), and how to reproduce the views.
3. Add a row to the table above.
