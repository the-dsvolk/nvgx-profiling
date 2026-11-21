# Option A: CUDA 13.5 forward-compatibility route

> **Status: PASSED on 2026-11-21** on a stock DGX Spark (R580 kernel driver).
> Run report and learnings: [`results/2026-11-21-option-a-dgx-spark.md`](../results/2026-11-21-option-a-dgx-spark.md).
> The steps below are the ones that worked.

Keep the R580 kernel driver and give traced programs the CUDA 13.5 user-mode driver from
`cuda-compat-13-5` (version 620.70.06). The SDK documents this as a supported fallback for kernel
drivers ≥ 580.95.05.

## Why it's low risk

- **Nothing is installed.** The compat libraries are unpacked into a home folder, and the prebuilt aarch64
  SDK release needs no CUDA 13.5 toolkit.
- Doesn't touch the kernel module, Secure Boot, the local 580 apt pin or OS updates. No reboot.
- Only processes started with the compat folder in `LD_LIBRARY_PATH` use the new driver.
- Root is needed for one thing only: running the service (counter profiling is admin-only).

## Steps

### 0. Preflight

- Record the baseline: `nvidia-smi`, `uname -r`, `mokutil --sb-state`, `dpkg -l | grep -E 'nvidia|cuda'`,
  `ls -l /usr/local/cuda`, the pin files in `/etc/apt/preferences.d/`.
- Free the GPU, or leave room for the test workload: stop other containers with `docker stop`
  (`docker start` brings them back unchanged).

### 1. Compat driver + SDK (no root)

```shell
./scripts/setup-option-a.sh        # LAB=~/nsight-perf-lab by default
```

It runs `apt-get download cuda-compat-13-5` + `dpkg-deb -x` into `$LAB/compat-root`, downloads the SDK
release, checks its SHA-256 against GitHub's published digest, and unpacks it into `$LAB/sdk`.

### 2. Check the compat driver works on GB10

```shell
nvcc -arch=sm_121 -o driver_version tools/driver_version.cu -lcuda
./driver_version                                                              # driver=13000 (R580)
LD_LIBRARY_PATH=$LAB/compat-root/usr/local/cuda-13.5/compat ./driver_version  # driver=13050
```

### 3. Start the service as root

```shell
sudo setsid -f ./scripts/start-root-service.sh "$HOME/nsight-perf-lab"   # global scope, stops after 6 h
$LAB/sdk/bin/nvtrace_poll_until_service_ready
journalctl -k --since "-5 min" | grep INSUFFICIENT                         # must print nothing
```

- Use `setsid -f`: `sudo -b` died with the SSH terminal before the service started.
- **Don't trust the readiness message alone.** A non-root service also says "All required GPU providers
  are measuring", but the kernel log shows `NV_ERR_INSUFFICIENT_PERMISSIONS ... ALLOC_PMA_STREAM` and the
  traces have no GPU kernels or counters.
- Don't put the compat folder in the service's `LD_LIBRARY_PATH`: it should load the system `libcuda.so.580.*`.
- For a permanent setup, the SDK ships `nsight-perf-daemon/scripts/install_nvidia_nsight_perf.sh`
  (systemd unit, dedicated account, `CAP_PERFMON`). Not tried yet.

### 4. Smoke test

```shell
cd $LAB/traces
LD_LIBRARY_PATH=$LAB/compat-root/usr/local/cuda-13.5/compat $LAB/sdk/bin/TestNvTraceCudaContinuousProfiling
```

- The log must say **"Connected to nvidia-nsight-perf service"**. If no service is running, the test
  starts its own as the current user, which lacks permissions, and fails on missing GPU kernel events.
- Pass looks like: `gpu_kernels=475, gpu_pm=13682, cuda_api=1164, nvtx=376`, `[  PASSED  ] 1 test`.
- Without the compat path the test kernels fail with "PTX was compiled with an unsupported toolchain":
  traced programs must use the 13.5 driver.

### 5. Trace vLLM

```shell
TRACED=0 ./scripts/run-vllm.sh     # untraced baseline (same compat driver, no service socket)
TRACED=1 ./scripts/run-vllm.sh     # traced: mounts /run/nvidia/nsight-perf into the container
```

- The container gets the compat folder mounted and first in `LD_LIBRARY_PATH`, followed by the image's
  own entries. `vllm/vllm-openai:v0.33.0` (CUDA 13.0 build) runs unchanged on it, CUDA graphs included.
- Start the container **after** the service.
- NVTX: apps built with NVTX 3.7+ are found automatically; older NVTX needs
  `nvtx_cuda_driver_injection_proxy` through `NVTX_INJECTION64_PATH`. vLLM emits no NVTX by default.
- R620 user-mode NVTX issues: don't mix NVTX core and payload push/pop on one range; avoid NVTX on both
  sides of `fork()` (PyTorch `DataLoader`: use `spawn` or `forkserver`).

Capture and export:

```shell
$LAB/sdk/bin/nvtrace_streaming_consumer --output-dir $LAB/traces --duration-ms 70000 &
# ...run load...
$LAB/sdk/bin/nvtrace_perfetto_trace_exporter --tsf-database-dir $LAB/traces/<hostname>.<timestamp>
```

Inspect with [`tools/inspect_trace.py`](../tools/inspect_trace.py) or <https://ui.perfetto.dev>.
Traces embed the host name: scrub before sharing.

### 6. Measure overhead

Same model, prompts and concurrency, untraced vs. traced (with and without a capture running):

```shell
docker exec nvgx-vllm-test vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
  --model qwen3-0.6b --tokenizer Qwen/Qwen3-0.6B --dataset-name random \
  --random-input-len 512 --random-output-len 256 --num-prompts 200 --max-concurrency 16 --seed 1
```

2026-11-21: ~1295 output tok/s in all four runs (difference < 0.5%).

## Gate

Pass/fail criteria are in [the plan overview](README.md#gates). Write each run to `results/`.

## Limits that stay with Option A

- No GPU power metrics (needs kernel driver R615+; NVIDIA documents total power only for Rubin).
- Only one tool can own the GPU's performance counters: Nsight Systems / Nsight Compute can't
  collect counters while the service runs.
- Traces stay on the box (local socket + shared memory). Copy files to move them.

## Rollback (seconds, no reboot)

```shell
sudo pkill -f 'nvidia-nsight-perf --global'    # or wait for the timeout in start-root-service.sh
docker rm -f nvgx-vllm-test
rm -rf ~/nsight-perf-lab                        # compat libraries, SDK, traces
```
