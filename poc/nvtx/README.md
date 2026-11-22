# NVTX phases: prefill vs. decode GPU time for vLLM

Goal: let the Nsight Perf SDK for AI attribute vLLM's GPU work to **prefill** and **decode**, without
rebuilding the vLLM image and without turning CUDA graphs off.

The SDK traces NVTX push/pop ranges and projects every GPU operation onto the ranges that were open on
the CPU thread when it was launched (including kernels inside CUDA graph replays). So one NVTX range
per engine step, whose message says how many prefill and decode tokens the step carries, is enough to
split GPU time by phase.

| Piece | What it is |
|---|---|
| [`nvgx_nvtx/__init__.py`](nvgx_nvtx/__init__.py) | **Option 3**: vLLM general plugin. One NVTX range per engine step in domain `nvgx`, message `step prefill_tokens=… decode_tokens=… reqs=… prefill_reqs=… decode_reqs=…` |
| [`nvgx_nvtx-0.1.dist-info/`](nvgx_nvtx-0.1.dist-info/) | Entry point `vllm.general_plugins` → `nvgx_nvtx:register`, so vLLM loads the plugin from `PYTHONPATH` with no `pip install` |
| [`selftest.py`](selftest.py) | Checks the counting rule on real vLLM `SchedulerOutput` objects and times the hook |
| [`nvtx_report.py`](nvtx_report.py) | Reads an exported Perfetto trace: NVTX range counts, sample messages, GPU kernel time per step class |
| [`overhead.sh`](overhead.sh) | NVTX on/off A/B benchmark on two side-by-side traced containers |
| [`nesting_test.py`](nesting_test.py) | Checks that the SDK keeps default-domain ranges nested inside an `nvgx` range (it does: 1,200 / 1,200) |
| [`exporter-supervisor.sh`](exporter-supervisor.sh) | Keeps the fleet exporter running (restart + gdb backtrace on crash), see [../README.md](../README.md#prefill-vs-decode-nvtx-phases) |
| [`../../scripts/run-vllm.sh`](../../scripts/run-vllm.sh) | `NVTX=1` mode: Option 1 + Option 3 + injection proxy (+ `RUNNER=v1` to force the V1 model runner) |

## Option 1: vLLM's own NVTX scopes (`VLLM_NVTX_SCOPES_FOR_PROFILING=1`)

`vllm/v1/utils.py: record_function_or_nullcontext` switches to `nvtx.annotate` (default NVTX domain).
What that covers in vLLM 0.33 depends on the GPU model runner:

- **Model Runner V2** (what vLLM 0.33 picks for Qwen3-0.6B: log line `Using V2 Model Runner`) has **no**
  NVTX scopes. You only get the scheduler's: `schedule: allocate_slots` (once per scheduled request),
  `schedule: make_cached_request_data`, `schedule: update_after_schedule`. They are CPU-only, so they
  have no GPU projection.
- **V1 model runner** (`VLLM_USE_V2_MODEL_RUNNER=0`, `RUNNER=v1` in `run-vllm.sh`) adds the stage ranges in
  `gpu_model_runner.py`: `gpu_model_runner: preprocess / forward / postprocess / sample / bookkeep / …`.

  Verified with `RUNNER=v1` in a side container (exporter DuckDB, 1 min of load): 529 `nvgx` steps, and
  `gpu_model_runner: preprocess / forward / postprocess / sample / bookkeep / eplb / ModelRunnerOutput /
  AsyncGPUModelRunnerOutput / set_async_sampled_token_ids` 529–531 times each, nested inside the step ranges.

None of these carry token counts, so Option 1 alone can't tell prefill from decode.
`--enable-layerwise-nvtx-tracing` needs CUDA graphs off and is not used.

## Option 3: the `nvgx_nvtx` plugin (one range per engine step)

**Loading.** `run-vllm.sh` mounts this folder at `/opt/nvgx-nvtx` and sets `PYTHONPATH=/opt/nvgx-nvtx`.
vLLM calls `load_general_plugins()` in every process it starts; that finds the entry point in
`nvgx_nvtx-0.1.dist-info` and calls `register()`. `register()` imports nothing heavy: it installs an
import hook that patches `vllm.v1.worker.gpu_worker` right after vLLM imports it. Only the engine-core
process imports that module (logs: `nvgx_nvtx: patched … (pid <EngineCore>)`), so the API server is never
patched. (A `sitecustomize.py` would also work, but the image already ships Ubuntu's.)

**What is wrapped.** `Worker.execute_model` and `Worker.sample_tokens`, which both runners (V1 and V2)
share. The range is pushed when `execute_model` starts and popped when that step's `sample_tokens`
returns (or right away if `execute_model` returns the output itself). So the step's forward pass,
CUDA graph replays and sampler kernels all fall inside one range. Both calls run on the engine
busy-loop thread (also with async scheduling and the batch queue), so push/pop nest correctly. Steps with
no scheduled tokens get no range.

**Counting rule** (per request in `SchedulerOutput.num_scheduled_tokens`, `n` = its scheduled tokens):

| Request | Counts as |
|---|---|
| new this step (`scheduled_new_reqs`) | `n` prefill tokens |
| resumed after preemption (`scheduled_cached_reqs.resumed_req_ids`): its tokens are recomputed | `n` prefill tokens |
| cached with `num_output_tokens == 0`: a chunked-prefill continuation | `n` prefill tokens |
| cached with outputs: generating | `1 + len(scheduled_spec_decode_tokens[req])` decode tokens (next token + speculative drafts); anything beyond that (e.g. prompt tokens appended to a streaming session) is prefill |

A request with any prefill tokens counts in `prefill_reqs`, otherwise in `decode_reqs`;
`prefill_tokens + decode_tokens = total_num_scheduled_tokens`. `spec_tokens=` is appended only when drafts
are scheduled. This is vLLM's own `compute_iteration_details` context/generation split (new or
`num_output_tokens == 0` → context) plus the resumed and spec-decode cases. With async scheduling,
`num_output_tokens` includes the in-flight placeholder token, so the step after a request's last prefill
chunk correctly counts as decode.

Step class (used by `nvtx_report.py` and the fleet sink): **prefill** (decode_tokens = 0),
**decode** (prefill_tokens = 0), **mixed** (both; chunked prefill piggybacks on decode batches, so most
prefill work runs in mixed steps).

**Safety.**

- *fork*: NVTX is called only when a step runs, i.e. only in the engine-core process. The NVTX domain
  (whose creation initializes NVTX and loads the injection) is created on the first step. Option 1's
  scopes also run only in the engine core. On top of that `run-vllm.sh` sets
  `VLLM_WORKER_MULTIPROC_METHOD=spawn` (vLLM's default here is fork), so the engine core starts from a
  fresh interpreter and no NVTX state can cross a `fork()` (the R620 driver issue in the SDK README).
- *CUDA graphs*: the hook only runs CPU-side Python around `execute_model`; nothing is captured.
  Graphs stay `FULL_AND_PIECEWISE`.
- *push/pop*: core `nvtxDomainRangePushEx` / `nvtxDomainRangePop` only, no payload extension; the SDK
  doesn't support start/end ranges.
- *registered strings*: the `nvtx` package always registers messages. The hook caches event attributes
  per distinct message, bounded at 65,536 entries (a 25 s capture under load had ~100 distinct messages).
- `NVGX_NVTX=0` disables the hook; `NVGX_NVTX_DOMAIN` renames the domain.

**Cost** (`selftest.py` in the container): building the message for a 256-request batch: ~24 µs;
push + pop with the SDK attached: 0.65 µs. Engine steps here take 3–100 ms.

## NVTX version and the injection proxy

The image's `nvtx` 0.2.15 Python package embeds NVTX older than 3.7: its extension modules contain
`NVTX_INJECTION64_PATH` and `InitializeInjectionNvtx2` but not `cuNvtxGetInjectionInitializer`, the CUDA
driver discovery that NVTX 3.7 added. Tested in the container with the SDK attached:

- without the proxy: `nvtx.Domain("nvgx")` is disabled (dummy domain), every range is a no-op;
- with `NVTX_INJECTION64_PATH=<sdk>/bin/libnvtx_cuda_driver_injection_proxy.so`: domain enabled, ranges traced.

The proxy activates only when the SDK's NVTX configuration exists (`/run/nvidia/nvtx/nvtx-cuda.yaml`,
written by the service), so `run-vllm.sh` also mounts `/run/nvidia/nvtx` read-only.

## Run it

```shell
# Root service first (scripts/start-root-service.sh), then:
NVTX=1 scripts/run-vllm.sh                  # vLLM with Option 1 + Option 3 + proxy (V2 runner)
NVTX=1 RUNNER=v1 scripts/run-vllm.sh        # same, V1 model runner (adds gpu_model_runner stage ranges)
docker exec nvgx-vllm-test python3 /opt/nvgx-nvtx/selftest.py

# Capture 25 s under load, export, report
sdk/bin/nvtrace_streaming_consumer --output-dir cap --duration-ms 25000
sdk/bin/nvtrace_perfetto_trace_exporter cap/<db>
unzip cap/<db>.perfetto-trace.zip -d x && venv/bin/python poc/nvtx/nvtx_report.py x/*.perfetto-trace
```

The container command `NVTX=1` produces:

```shell
docker run -d --name nvgx-vllm-test --restart no --gpus all --ipc=host -p 127.0.0.1:8010:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $COMPAT:/opt/cuda-compat-13.5:ro \
  -v /run/nvidia/nsight-perf:/run/nvidia/nsight-perf \
  -v $REPO/poc/nvtx:/opt/nvgx-nvtx:ro -e PYTHONPATH=/opt/nvgx-nvtx \
  -v $LAB/sdk/bin/libnvtx_cuda_driver_injection_proxy.so:/opt/nvtx-proxy/libnvtx_cuda_driver_injection_proxy.so:ro \
  -e NVTX_INJECTION64_PATH=/opt/nvtx-proxy/libnvtx_cuda_driver_injection_proxy.so \
  -v /run/nvidia/nvtx:/run/nvidia/nvtx:ro \
  -e VLLM_NVTX_SCOPES_FOR_PROFILING=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e LD_LIBRARY_PATH=/opt/cuda-compat-13.5:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib \
  -e HF_HUB_OFFLINE=1 vllm/vllm-openai:v0.33.0 Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --served-model-name qwen3-0.6b --gpu-memory-utilization 0.25 --max-model-len 4096
```

## How the SDK represents NVTX → GPU

- TSF / DuckDB tables: `nvtx_push_pop_ranges` (one row per range: thread, start/end, domain, message,
  nesting) and `nvtx_gpu_projection_segments` (one row per GPU operation × NVTX range on its projected
  stack; keys identify the operation: `kernel_launch_correlation_id` for immediate kernels,
  `graph_launch_api_correlation_id` + `graph_node_correlation_id` for kernels of a CUDA graph replay).
  The view `nvtx_gpu_projection_spans` coalesces consecutive operations of one range into spans.
- Perfetto export: ranges are slices on the CPU thread track (category `nvtx: nvgx`, `nvtx: Default` for
  vLLM's scopes); the projection is a second set of slices on the GPU tracks (category
  `nvtx_gpu_projection: nvgx`), one per span, linked to the CPU range by a flow arrow.

## Verification (DGX Spark, 2026-11-22)

vLLM 0.33 (V2 model runner, CUDA graphs `FULL_AND_PIECEWISE`), Qwen3-0.6B, root Nsight Perf service,
`NVTX=1`. 25 s captures with `nvtrace_streaming_consumer`, exported to Perfetto, read with `nvtx_report.py`.

**Capture 1** (`vllm bench serve` 1024 in / 128 out and 512 / 256, concurrency 16, plus other traffic):

| Category | Ranges | Note |
|---|---|---|
| `nvtx: nvgx` (our step ranges) | 902 | 97 distinct messages |
| `nvtx_gpu_projection: nvgx` | 1,846 | GPU-side spans of those ranges on two GPU tracks; 1,846 flows link them to the CPU ranges |
| `nvtx: Default` `schedule: allocate_slots` | 33,546 | Option 1, once per scheduled request |
| `nvtx: Default` `schedule: make_cached_request_data` / `update_after_schedule` | 902 / 902 | Option 1, once per step |

Sample messages:

```text
step prefill_tokens=0 decode_tokens=39 reqs=39 prefill_reqs=0 decode_reqs=39
step prefill_tokens=322 decode_tokens=30 reqs=31 prefill_reqs=1 decode_reqs=30
step prefill_tokens=2007 decode_tokens=41 reqs=44 prefill_reqs=3 decode_reqs=41
```

GPU kernel time by step class: decode 20,499 ms (82.5%), mixed 4,324 ms (17.4%), prefill 0, outside any step
range 19.5 ms (0.1%) of 24,842 ms. Steps: 823 decode, 79 mixed, 0 prefill-only (under load, chunked prefill
always rides along with decodes); 68,801 prefill and 33,523 decode tokens.

**Capture 2** (another agent's mixed load generator): 377 step ranges, 726 projection spans; kernel time decode
61.5%, mixed 38.2%, outside step ranges 0.4%. The fleet profiles for the same 25 s (below) said decode 62.5%,
mixed 37.5%, unattributed 0%.

Typical decode step at ~40 requests: the CPU range lasts 3–5 ms, its GPU projection ~21 ms. The GPU runs
behind the CPU (async scheduling), and the projection still lands on the right step.

**Overhead.** Two traced containers on the same GPU, identical except `NVTX=1` / `NVTX=0` (`overhead.sh`,
64 prompts, concurrency 16), each benchmark run on both at the same time:

| Workload | NVTX on: output tok/s (TPOT ms) | NVTX off: output tok/s (TPOT ms) | Δ |
|---|---|---|---|
| 1024 in / 128 out | 274.2 (54.9) | 280.3 (54.1) | −2.2% |
| 1024 in / 128 out | 260.9 (57.3) | 260.3 (57.6) | +0.2% |
| 128 in / 512 out | 459.7 (34.7) | 457.2 (34.8) | +0.6% |

No measurable difference (mean −0.5%, inside run-to-run noise). Running the containers one after the other
was useless here: other traffic on the GPU moved single runs by up to 2x. Per step the hook costs ~25 µs of
Python plus 0.65 µs for push/pop, against 3–100 ms steps (<0.5%).

## Fleet profiles

FleetSink turns the step ranges into `attribution.by_phase_ms`, `phase_tokens` and `phase_steps` in every
5 s workload profile; see [../README.md](../README.md#prefill-vs-decode-nvtx-phases).

## Limits

- Perfetto view: two containers whose engine processes have the same PID and TID inside their namespaces
  (148 / 1 here) end up on one Perfetto thread track, and the trace processor drops slices that interleave
  badly there. Count ranges from the exporter's DuckDB (`nvtx_push_pop_ranges`, keyed by
  `process_id_unique`), or trace one vLLM container at a time.
- The step range covers the engine thread only. GPU work launched from other threads (none seen here)
  would be outside it.
- Pipeline parallelism (`execute_model` returning intermediate tensors) and pooling models are handled
  (the range closes when `execute_model` returns output) but untested.
- Distinct messages are registered strings in the tool; long runs with very varied batches register more
  strings (one per distinct count combination).
