# Option B: native R620 driver (fallback, experiments DGX Spark only)

> **Not needed:** Option A passed on 2026-11-21
> ([results](../results/2026-11-21-option-a-dgx-spark.md)). Kept for reference.

Replace the R580 kernel driver with **R620.70.06** so the kernel driver and the CUDA 13.5 user-mode
driver match. That meets the SDK's main requirement and removes Option A's split-driver issue with
GPU GR-context tracing.

**Only if Option A fails its gate** (see [the plan overview](README.md#gates)). Never on a DGX Spark
that serves other workloads until DGX OS officially ships R620.

## Why it's risky

- **Not validated on DGX Spark.** The R620 driver lists GB10, but DGX OS is validated with R580 only.
  NVIDIA says a newer DGX OS is "in the pipeline" with no date
  ([forum thread](https://forums.developer.nvidia.com/t/r620-driver-and-cuda-13-5/388164)).
- **No prebuilt modules.** Ubuntu publishes signed driver modules for the DGX Spark kernels only up to
  R610/R615. R620 modules have to be built locally with DKMS, and rebuilt on every kernel update.
- **Secure Boot is on.** Locally built modules are unsigned, so they need your own signing key (MOK)
  enrolled at the console during a reboot (monitor + keyboard on the box), or Secure Boot turned off.
- **GB10 regressions.** Newer branches have had GB10-specific problems (reports include CUDA-graph
  hangs and 64k-page issues). Nobody has shown R620 is clean on the stock DGX OS kernel.
- **Updates fight it.** The local apt pin holds the driver on 580, and the DGX Spark repo has higher apt
  priority. OS updates (OTA) expect R580.

## Steps

### 0. Preflight

- Console access ready (monitor + keyboard) for the MOK enrollment screen.
- Save the baseline so rollback is exact:
  - `dpkg -l | grep -E 'nvidia|cuda|linux-modules-nvidia' > baseline-packages.txt`
  - copies of `/etc/apt/preferences.d/*`
  - `nvidia-smi -q > baseline-nvidia-smi.txt`, `uname -r`, `mokutil --sb-state`
- Stop and disable GPU containers (`restart=no`) so nothing grabs the GPU mid-change.
- Dry run first: `apt-get install -s nvidia-open=620.70.06-1ubuntu1` and review every removal.

### 1. Lift the 580 pin (on this box only)

- Move the pin file out of `/etc/apt/preferences.d/` (keep the copy).
- Optionally install `nvidia-driver-pinning-620.70.06` so apt stays on exactly that version.

### 2. Install R620 and build modules

```shell
sudo apt install nvidia-open=620.70.06-1ubuntu1    # DKMS builds the open kernel modules
```

- Check DKMS built for the running kernel: `dkms status`.

### 3. Secure Boot key enrollment

- DKMS creates a MOK key. Import it: `sudo mokutil --import /var/lib/dkms/mok.pub` (set a one-time password).
- Reboot, and on the console choose **Enroll MOK** → enter the password.

### 4. Verify the driver

- `nvidia-smi`: driver 620.70.06, CUDA 13.5; `mokutil --list-enrolled` shows the key;
  `dmesg | grep -i nvidia` has no signature errors.
- **GB10 regression checks before anything else:**
  - a CUDA-graph-heavy vLLM run (CUDA graphs on) under sustained load, no hangs;
  - a large unified-memory allocation test;
  - `nvidia-container-toolkit`: a GPU container starts and sees the GPU.

### 5. Run the SDK

Same as [Option A](option-a-cuda-compat.md) steps 3–7, without the compat `LD_LIBRARY_PATH`:
kernel and user-mode drivers now match, so GR-context tracing should initialise directly.

## Gate

Same pass criteria as Option A, plus the GB10 regression checks in step 4. Write results to `results/`.

## Rollback (~1 hour, reboot, may need the console)

1. `sudo apt purge nvidia-open 'nvidia-*620*' 'libnvidia-*620*'` and remove `nvidia-driver-pinning-620*`.
2. Put the 580 pin file back.
3. Reinstall the R580 packages from `baseline-packages.txt`, including
   `linux-modules-nvidia-580-open-$(uname -r)` (prebuilt, signed).
4. Reboot, then check `nvidia-smi` shows R580 and the GPU containers start.
5. Optional: remove the enrolled MOK key (`mokutil --delete`).
