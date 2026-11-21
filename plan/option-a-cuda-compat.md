# Option A: CUDA 13.5 forward-compatibility route (recommended first)

Keep the R580 kernel driver and add the CUDA 13.5 user-mode driver from `cuda-compat-13-5`
(version 620.70.06). The SDK documents this as a supported fallback for kernel drivers
≥ 580.95.05. Box: **DGX Spark #2**.

## Why it's low risk

- Installs user-space files only: `/usr/local/cuda-13.5/compat` and the 13.5 toolkit next to 13.0.
- Doesn't touch the kernel module, Secure Boot, the local 580 apt pin or OS updates.
- Only processes started with `LD_LIBRARY_PATH=/usr/local/cuda-13.5/compat` use the new driver.
- An apt dry run installs `cuda-compat-13-5` cleanly (one package, nothing removed). The toolkit adds
  ~64 packages, removes nothing, and pulls no driver packages (`cuda-driver-dev-13-5` is headers/stubs).

## Steps

### 0. Preflight

- Record the baseline: `nvidia-smi`, `uname -r`, `mokutil --sb-state`, `dpkg -l | grep -E 'nvidia|cuda'`,
  `ls -l /usr/local/cuda`, and the pin files in `/etc/apt/preferences.d/`.
- Decide what runs on the GPU during tests. Either free it, or keep a vLLM container as the test workload.

### 1. Install CUDA 13.5 side by side

```shell
sudo apt install cuda-compat-13-5 cuda-toolkit-13-5
ls -l /usr/local/cuda    # may now point at 13.5; set it back to 13.0 if anything depends on it
```

### 2. Check the compat driver works on GB10

Build a tiny program that prints `cudaDriverGetVersion()`, then run it with and without the compat path:

```shell
./driver_version                                                  # expect 13000 (R580)
LD_LIBRARY_PATH=/usr/local/cuda-13.5/compat ./driver_version      # expect 13050
```

If the second run fails, the compat route doesn't work on DGX Spark: stop and go to the gate.

### 3. Get the SDK

- Use the aarch64 release tarball (`beta-preview-update1` or newer) and check `SHA256SUMS`,
  **or** build from source: `git clone --recursive`, `cmake -S . -B build`, `cmake --build build -j`.
- Record the SDK version.

### 4. Start the service on the host-matched driver

- Counter profiling is admin-only (`RmProfilingAdminOnly=1`), so run the service as root
  (systemd unit from the daemon's Linux install guide, or `sudo ./nvidia-nsight-perf --global`).
- **Don't** set the compat `LD_LIBRARY_PATH` for the service. Confirm it loaded the system driver:
  `sudo grep libcuda /proc/$(pgrep -f nvidia-nsight-perf)/maps` → `libcuda.so.580.*`.
- `./nvtrace_poll_until_service_ready`

### 5. Smoke test

```shell
LD_LIBRARY_PATH=/usr/local/cuda-13.5/compat ./TestNvTraceCudaContinuousProfiling
./nvtrace_streaming_consumer --output-dir results/ --duration-ms 10000
./nvtrace_perfetto_trace_exporter --tsf-database-dir results/<hostname>.<timestamp>
```

- Check the service log for `GPU GR Context Trace provider failed to initialize the RM event buffer`.
- Open the `.perfetto-trace.zip` in <https://ui.perfetto.dev>: CUDA API calls, GPU kernels with names
  and timings.

### 6. Trace a real workload (vLLM)

- The service runs on the host (host PID namespace). The container must see the same socket directory:
  bind-mount `/run/nvidia/nsight-perf`.
- Give the container the compat driver: bind-mount `/usr/local/cuda-13.5/compat` and set
  `LD_LIBRARY_PATH` to it. A CUDA 13.0 build runs on the newer user-mode driver.
- NVTX: apps built with NVTX 3.7+ are found automatically. Older NVTX needs the
  `nvtx_cuda_driver_injection_proxy` through `NVTX_INJECTION64_PATH`.
- Known R620 user-mode issues: don't mix NVTX core and payload push/pop on one range, and avoid NVTX on
  both sides of `fork()` (PyTorch `DataLoader`: use `spawn` or `forkserver`).
- First run vLLM under the compat driver **without** tracing (CUDA graphs on) to check it's stable.
  Then capture with tracing on.

### 7. Measure overhead

- Same model, prompt set and concurrency: tokens/s and latency with the service off vs. on.
- Try a couple of PM-counter sampling rates (the max recommended is 20 kHz).

## Gate

Pass/fail criteria are in [the plan overview](README.md#gates). Write the result to `results/`.

## Limits that stay with Option A

- No GPU power metrics (needs kernel driver R615+; NVIDIA documents total power only for Rubin).
- Only one tool can own the GPU's performance counters: Nsight Systems / Nsight Compute can't
  collect counters while the service runs.
- Traces stay on the box (local socket + shared memory). Copy files to move them.

## Rollback (minutes, no reboot)

```shell
sudo systemctl stop nvidia-nsight-perf    # or kill the service
sudo apt remove cuda-compat-13-5 cuda-toolkit-13-5 && sudo apt autoremove
ls -l /usr/local/cuda                     # make sure it points at 13.0 again
```
