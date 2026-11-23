# Plan: NCCL profiling across two DGX Sparks (NCCL 2.32.6)

> **Status (2026-11-22):** stage 1 (micro-benchmark) **done**, results below. Stage 2 steps 1–2 **done**: vLLM TP=2
> across both Sparks on NCCL 2.32.6, NCCL time on the dashboard from the SDK alone, and NCCL Inspector on the
> traced side ([results](../results/2026-11-22-nccl-tp2/README.md)). Stage 3 is next.

The Nsight Perf SDK for AI shows what one GPU does. Once a model is split across two DGX Sparks with
tensor parallelism, every decode step also waits on all-reduces over the network, and we want those in
the same picture. NCCL has a **profiler plugin API** for that: NCCL calls a plugin on group, collective,
proxy and kernel events, and the plugin decides what to record.

The question for this plan is **which NCCL to profile with**:

| | NCCL 2.31.2 | NCCL 2.32.6 |
|---|---|---|
| Where it comes from | Bundled in our vLLM image (`vllm/vllm-openai:v0.33.0`, pip `nvidia-nccl`, `2.31.2+cuda13.3`) | Newest release (2026-11-09); `libnccl2 2.32.6-1+cuda13.5` in NVIDIA's CUDA apt repo |
| NVIDIA's DGX Spark guide | **The recommended version** ([build.nvidia.com/spark/nccl](https://build.nvidia.com/spark/nccl), updated 2026-10-14) | Not yet |
| Profiler plugin API | v7 | v8 |
| Profiler overhead | Regressed in 2.31.2: [NVIDIA/nccl#2355](https://github.com/NVIDIA/nccl/issues/2355) measured 4.3–13.3% all-reduce overhead at 128 KB–4 MB with the example plugin (8× A100), vs 0.1–1.1% on 2.29.7 | Fixed in 2.32.3: *"Fixes profiler overhead when no profiler plugin is loaded"* ([release notes](https://github.com/NVIDIA/nccl/releases/tag/v2.32.3-1), issue #2355) |

So 2.32.6 should cost less to profile, and maybe a little less even without a plugin, but it isn't
what NVIDIA's Spark guide uses. **Stage 1 measures that difference on our two Sparks before we pick one.**

## Approach: containers only, nothing installed on the hosts

- **vLLM uses the NCCL inside its container**, not the host's, so the host `libnccl2` doesn't matter.
  Neither Spark gets an NCCL upgrade.
- 2.32.6 is tested **inside the same image** on both nodes: `apt-get download libnccl2=2.32.6-1+cuda13.5`,
  `dpkg-deb -x` into a folder, mount it into the container, and force it with `LD_PRELOAD`. The image stays
  `vllm/vllm-openai:v0.33.0`.
- **Both ranks always run the same NCCL build.** Different NCCL versions across ranks are a classic cause of hangs.
- Every run records the NCCL that was actually used (`ncclGetVersion` through the process's global symbol
  scope) and checks the `NCCL version` line in NCCL's log. `torch.cuda.nccl.version()` is no good for this:
  it reports the NCCL torch was **built** with (2.30.7), whatever is loaded.
- If 2.32.6 becomes the standard, the next step is a small derived image with it built in, still on both nodes.

## Stages

| Stage | What | Gate |
|---|---|---|
| **1. Micro-benchmark** (done) | Two-node all-reduce, 2.31.2 vs 2.32.6, each without and with NCCL's example profiler plugin | Both builds run over RoCE on both Sparks; overhead numbers per message size |
| **2. vLLM, tensor parallel 2** (steps 1–2 done) | Same image on both Sparks, `--tensor-parallel-size 2` with the NCCL picked in stage 1; the load generator from [`poc/workload/`](../poc/workload/README.md). Step 1: NCCL kernels as the SDK sees them, on the dashboard. Step 2: NCCL Inspector on the traced side only | No hangs (3 × 10 min, 0 errors ✅); NCCL share per step on the dashboard ✅; Inspector overhead below noise ✅. Not yet: a 30-min run, tokens/s vs a single Spark |
| **3. NCCL in the fleet profile** | A profiler plugin (or NCCL Inspector) that appends collective events lock-free and hands them to the exporter next to the Nsight Perf SDK data | Per-step all-reduce time on the dashboard; overhead on vLLM tokens/s < 3% |

## Stage 1: micro-benchmark

### Setup

| | |
|---|---|
| Nodes | 2× DGX Spark (GB10), one GPU each, one rank per node |
| Link | Direct QSFP cable between the two Sparks, RoCE. NCCL uses both rails (`NET/IB/0`, `NET/IB/1`); no GPUDirect RDMA on GB10 (`GDR 0`) |
| Container | `vllm/vllm-openai:v0.33.0` (torch 2.14.0+cu130), host network, `/dev/infiniband` devices, `IPC_LOCK` |
| User-mode driver | CUDA 13.5 compat (620.70.06), same as [Option A](option-a-cuda-compat.md), for **all** configs so the driver isn't a variable (not tested whether the `+cuda13.5` NCCL build also runs on the stock R580 user-mode driver) |
| NCCL A | **2.31.2+cuda13.3**, the image's own |
| NCCL B | **2.32.6+cuda13.5**, unpacked `libnccl2` .deb, `LD_PRELOAD` |
| Profiler plugin | NCCL's **example profiler plugin** built from the v2.32.6-1 source inside the same image. It exports API v5–v8, so one build works with both NCCL versions (2.31.2 loads it as v7, 2.32.6 as v8). `NCCL_PROFILE_EVENT_MASK=71` (Group, Coll, P2p, KernelCh), the setting from issue #2355. No trace dump file, so events only go to its in-memory pools. "No plugin" runs set `NCCL_PROFILER_PLUGIN=none` |
| Benchmark | [`poc/nccl/allreduce_bench.py`](../poc/nccl/allreduce_bench.py) via [`run_bench.sh`](../poc/nccl/run_bench.sh): bf16 all-reduce, 4 KB–64 MB |
| Schedule | 4 configs × **6 rounds, interleaved** (A, B, A+plugin, B+plugin, repeat) so drift hits all configs alike. Each round runs 3 passes over all sizes; 400 calls per size up to 256 KB, 40 above |

Two latencies per size:

- **pipelined:** back-to-back all-reduces between two CUDA events, averaged. Launch overhead is hidden, as in
  a CUDA graph.
- **synced p50:** one all-reduce, then wait for the device, repeated. This is closer to a decode step, where
  each layer's all-reduce has to finish before the next layer runs.

Each table cell is the median over the 6 rounds. Message sizes matter differently: decode all-reduces for the
models we run are small (hidden size × batch × 2 bytes, so a few KB to ~64 KB); prefill and large batches
reach hundreds of KB to MB.

### Results

**Latency, µs: pipelined / synced p50**

| Size | 2.31.2 | 2.32.6 | 2.31.2 + plugin | 2.32.6 + plugin |
|---|---|---|---|---|
| 4 KB | 15.7 / 25.6 | 15.2 / 25.5 | 15.9 / 26.7 | 15.4 / 26.3 |
| 8 KB | 17.5 / 26.9 | 17.3 / 26.8 | 18.0 / 28.3 | 17.5 / 27.7 |
| 16 KB | 19.0 / 28.7 | 18.9 / 28.5 | 19.8 / 30.8 | 19.2 / 29.9 |
| 32 KB | 21.9 / 31.9 | 21.8 / 32.0 | 22.9 / 34.3 | 22.1 / 33.3 |
| 64 KB | 30.2 / 40.9 | 30.8 / 40.9 | 31.0 / 43.8 | 30.9 / 43.4 |
| 128 KB | 39.0 / 46.7 | 38.7 / 46.4 | 39.8 / 47.5 | 38.7 / 46.6 |
| 256 KB | 59.5 / 62.5 | 58.4 / 61.9 | 59.0 / 67.8 | 58.1 / 66.7 |
| 1 MB | 458.0 / 444.4 | 458.3 / 455.1 | 458.5 / 473.2 | 452.3 / 463.0 |
| 4 MB | 326.1 / 333.5 | 330.6 / 339.8 | 332.5 / 358.4 | 333.3 / 352.8 |
| 16 MB | 771.8 / 832.7 | 771.0 / 837.2 | 773.8 / 842.1 | 770.3 / 841.7 |
| 64 MB | 3354.8 / 3464.9 | 3338.0 / 3434.9 | 3331.3 / 3427.1 | 3362.1 / 3448.6 |

**Bus bandwidth, GB/s**

| Size | 2.31.2 | 2.32.6 | 2.31.2 + plugin | 2.32.6 + plugin |
|---|---|---|---|---|
| 1 MB | 2.29 | 2.29 | 2.29 | 2.32 |
| 4 MB | 12.86 | 12.68 | 12.62 | 12.58 |
| 16 MB | 21.74 | 21.76 | 21.68 | 21.78 |
| 64 MB | 20.00 | 20.10 | 20.14 | 19.96 |

**Profiler-plugin overhead: plugin on vs off, same NCCL**

| Size | 2.31.2: pipelined | 2.31.2: synced p50 | 2.32.6: pipelined | 2.32.6: synced p50 |
|---|---|---|---|---|
| 4 KB | +1.7% | +4.4% | +1.5% | +3.1% |
| 8 KB | +3.3% | +5.5% | +1.2% | +3.3% |
| 16 KB | +4.1% | +7.2% | +1.3% | +4.8% |
| 32 KB | +4.8% | +7.5% | +1.7% | +4.2% |
| 64 KB | +2.4% | +7.0% | +0.4% | +5.9% |
| 128 KB | +1.9% | +1.7% | +0.1% | +0.5% |
| 256 KB | -0.9% | +8.5% | -0.6% | +7.6% |
| 1 MB | +0.1% | +6.5% | -1.3% | +1.7% |
| 4 MB | +2.0% | +7.4% | +0.8% | +3.8% |
| 16 MB | +0.3% | +1.1% | -0.1% | +0.5% |
| 64 MB | -0.7% | -1.1% | +0.7% | +0.4% |

**2.32.6 vs 2.31.2: latency change (negative = 2.32.6 faster)**

| Size | No plugin: pipelined | No plugin: synced p50 | Plugin: pipelined | Plugin: synced p50 |
|---|---|---|---|---|
| 4 KB | -3.0% | -0.4% | -3.2% | -1.6% |
| 8 KB | -1.2% | -0.4% | -3.1% | -2.4% |
| 16 KB | -0.7% | -0.6% | -3.4% | -2.9% |
| 32 KB | -0.4% | +0.4% | -3.4% | -2.8% |
| 64 KB | +1.7% | +0.0% | -0.2% | -0.9% |
| 128 KB | -1.0% | -0.7% | -2.7% | -1.9% |
| 256 KB | -1.9% | -0.8% | -1.6% | -1.6% |
| 1 MB | +0.1% | +2.4% | -1.3% | -2.2% |
| 4 MB | +1.4% | +1.9% | +0.3% | -1.6% |
| 16 MB | -0.1% | +0.5% | -0.4% | -0.0% |
| 64 MB | -0.5% | -0.9% | +0.9% | +0.6% |

### What it shows

1. **Both builds work over RoCE between the two Sparks.** 2.32.6 loaded into the unchanged vLLM image on
   both nodes, picked the same transport (both rails, `GDR 0`) and finished all 24 of its runs; so did 2.31.2.
2. **Without a plugin, there's no measurable difference.** Synced latency is within ±1% at every size up to
   256 KB, and bandwidth is the same (~21.7 GB/s at 16 MB). Whatever no-plugin overhead 2.32.3 removed is
   below our noise floor with 2 ranks.
3. **With the plugin loaded, 2.32.6 costs about a third less.** Averaged over decode-sized messages
   (4–64 KB), plugin overhead drops from **6.3% to 4.3%** of synced latency and from **3.3% to 1.2%** of
   pipelined latency. Over issue #2355's range (128 KB–4 MB) it drops from 6.0% to 3.4% synced. With the plugin,
   2.32.6 is faster than 2.31.2 at every size from 4 KB to 1 MB, in both measures.
4. **The overhead is small in absolute terms:** ~1–3 µs per all-reduce at decode sizes (e.g. 16 KB: +2.1 µs
   on 2.31.2, +1.4 µs on 2.32.6). Large messages (16–64 MB) show none on either version.
5. **We don't see #2355's 4–13% regression at full size on 2.31.2.** That report used 8 GPUs with many
   more channels, so more `KernelCh` events per collective; with 2 ranks over the network the
   per-collective event count is lower.

**Decision:** use **NCCL 2.32.6** (loaded into the same image on both nodes) for stages 2–3, where the
profiler plugin is always on. It costs nothing without the plugin and roughly a third less with it. Keep
2.31.2, NVIDIA's Spark-guide version, as the fallback if 2.32.6 hangs under vLLM in stage 2.

### Caveats

- **Noise.** Rank 0's Spark also serves other models, so a request there can slow a run. Single rounds spread
  by up to ~60% at 32–256 KB (see `summarize.py`'s spread table); interleaving and the 6-round median absorb
  this, but differences under ~2% in one cell are not meaningful. The 16–64 MB rows spread by < 3%.
- **Eager, not CUDA graphs.** vLLM captures its decode all-reduces in CUDA graphs. The profiler plugin
  works there, with caveats: inter-node collective events come from NCCL's proxy thread when the graph runs,
  not from the API call ([NVIDIA/nccl#1805](https://github.com/NVIDIA/nccl/issues/1805)). Stage 2 has to
  measure the overhead inside vLLM itself.
- **The example plugin is not our plugin.** It is a reference implementation with mutex-protected pools;
  a plugin built for always-on use (lock-free appends, nothing slow on the proxy thread) should cost less.
- **Two ranks, one link.** No intra-node NVLink, no tree algorithms; results don't carry over to bigger systems.

## Reproduce

On both Sparks (nothing is installed; everything lives in `~/nccl-lab`):

```shell
mkdir -p ~/nccl-lab/{dl,compat,results} && cd ~/nccl-lab/dl
apt-get download libnccl2=2.32.6-1+cuda13.5 cuda-compat-13-5
dpkg-deb -x libnccl2_*.deb ../nccl-2.32.6
dpkg-deb -x cuda-compat-13-5_*.deb compat-root && cp -a compat-root/usr/local/cuda-13.5/compat/* ../compat/
cp <repo>/poc/nccl/{allreduce_bench.py,run_bench.sh} ~/nccl-lab/
```

Build the example profiler plugin once, inside the image, and copy it to the other Spark:

```shell
git clone --depth 1 -b v2.32.6-1 https://github.com/NVIDIA/nccl.git ~/nccl-lab/nccl-src
docker run --rm -v ~/nccl-lab:/lab -w /lab/nccl-src/plugins/profiler/example --entrypoint bash \
  vllm/vllm-openai:v0.33.0 -c "g++ -O2 -Inccl -I/usr/local/cuda/include -fPIC -shared \
  -o /lab/libnccl-profiler-example.so *.cc -L/usr/local/cuda/lib64 -lcudart -lpthread"
```

Run one config: rank 1 first, then rank 0 (`<rank0-link-ip>` is rank 0's address on the direct link):

```shell
LIB=nccl-2.32.6/usr/lib/aarch64-linux-gnu/libnccl.so.2
# Spark #2:
PROFILER_PLUGIN=libnccl-profiler-example.so ~/nccl-lab/run_bench.sh 1 <rank0-link-ip> v2326-prof71-r1 $LIB
# Spark #1:
PROFILER_PLUGIN=libnccl-profiler-example.so ~/nccl-lab/run_bench.sh 0 <rank0-link-ip> v2326-prof71-r1 $LIB
```

Leave out `$LIB` for the image's NCCL and `PROFILER_PLUGIN` for no plugin. Repeat the four configs
interleaved, then `python3 poc/nccl/summarize.py <results-dir>`. The raw results of this run are in
[`results/2026-11-22-nccl-two-spark/raw/`](../results/2026-11-22-nccl-two-spark/raw/).
