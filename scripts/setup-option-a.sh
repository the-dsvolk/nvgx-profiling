#!/usr/bin/env bash
# Option A setup on DGX Spark, no root needed:
#   - CUDA 13.5 forward-compat user-mode driver (cuda-compat-13-5), unpacked, not installed
#   - Nsight Perf SDK for AI prebuilt aarch64 release, checksum-verified
# Everything goes under $LAB (default ~/nsight-perf-lab).
set -euo pipefail

LAB=${LAB:-$HOME/nsight-perf-lab}
SDK_TAG=${SDK_TAG:-beta-preview-update1}
SDK_ASSET="nsight-perf-sdk-for-ai-${SDK_TAG}-linux-aarch64.tar.gz"
# GitHub's published digest for the beta-preview-update1 aarch64 asset. Update with SDK_TAG.
SDK_SHA256=${SDK_SHA256:-bb18aa0a0c04de1e047d92a010696506cbd6939f6ff4a40b9cb78d59b7d31397}

mkdir -p "$LAB"/{dl,compat-root,sdk,logs,traces,bench}
cd "$LAB/dl"

# 1. CUDA 13.5 compat libraries (needs NVIDIA's CUDA apt repo, which DGX OS has).
apt-get download cuda-compat-13-5
dpkg-deb -x cuda-compat-13-5_*_arm64.deb "$LAB/compat-root"

# 2. SDK release.
curl -fsSL -o "$SDK_ASSET" \
  "https://github.com/NVIDIA/nsight-perf-sdk-for-ai/releases/download/${SDK_TAG}/${SDK_ASSET}"
echo "${SDK_SHA256}  ${SDK_ASSET}" | sha256sum -c -
tar -xzf "$SDK_ASSET" -C "$LAB/sdk" --strip-components=1

echo "compat driver: $LAB/compat-root/usr/local/cuda-13.5/compat"
echo "SDK binaries:  $LAB/sdk/bin"
