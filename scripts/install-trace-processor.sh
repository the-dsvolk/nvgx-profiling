#!/usr/bin/env bash
# Install Perfetto's trace_processor on the DGX Spark (no root), for viewing big traces.
# Uses Perfetto's official launcher (get.perfetto.dev). On first run the launcher downloads the
# native binary for this platform (linux-arm64), checks its SHA-256 against the manifest built
# into the launcher, and caches it in ~/.local/share/perfetto/prebuilts/.
# The launcher tracks Perfetto's latest release (v59.1 on 2026-11-21).
set -euo pipefail

LAB=${LAB:-$HOME/nsight-perf-lab}
mkdir -p "$LAB/tools"
curl -fsSL -o "$LAB/tools/trace_processor" https://get.perfetto.dev/trace_processor
chmod +x "$LAB/tools/trace_processor"
"$LAB/tools/trace_processor" --version
