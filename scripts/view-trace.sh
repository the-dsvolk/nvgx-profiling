#!/usr/bin/env bash
# View a trace that lives on the DGX Spark in ui.perfetto.dev, with trace_processor running on the
# Spark as Perfetto's "native accelerator". Run this on your laptop; the trace is never copied.
#
#   scripts/view-trace.sh <ssh-host> <trace-path-on-spark>
#   e.g. scripts/view-trace.sh my-spark '~/nsight-perf-lab/traces/vllm/<hostname>.<timestamp>.perfetto-trace.zip'
#
# What it does:
#   1. Starts trace_processor on the Spark, bound to the Spark's 127.0.0.1:9001, with the trace loaded.
#   2. Tunnels the laptop's 127.0.0.1:9001 to it over SSH (the Perfetto UI only probes 127.0.0.1:9001).
#   3. Opens the UI build that matches the trace_processor build (avoids the "Version mismatch" dialog).
#      In the UI, choose "YES, use loaded trace".
# Ctrl+C stops the tunnel and the remote trace_processor. It also stops after 1 h idle.
# While the trace loads, ssh prints a few harmless "channel 3: open failed" lines.
# If the UI shows no dialog (badge says WSM, not RPC): close other Perfetto tabs and reload.
# Needs scripts/install-trace-processor.sh run on the Spark first. Use Chrome (or another Chromium).
set -euo pipefail

usage="usage: view-trace.sh <ssh-host> <trace-path-on-spark>"
HOST=${1:?$usage}
TRACE=${2:?$usage}
TP=${TP:-'~/nsight-perf-lab/tools/trace_processor'}   # expanded on the Spark
PORT=9001

if curl -sf -o /dev/null "http://127.0.0.1:${PORT}/status"; then
  echo "Port ${PORT} is already in use on this machine (another trace_processor or tunnel?)." >&2
  exit 1
fi

VERSION=$(ssh "$HOST" "$TP --version" | sed -n 's/^Perfetto \(v[0-9.]*-[0-9a-f]*\).*/\1/p')
URL="https://ui.perfetto.dev/${VERSION:+${VERSION}/}"

open_url() {
  if [ -d "/Applications/Google Chrome.app" ]; then open -a "Google Chrome" "$1"
  elif command -v open >/dev/null; then open "$1"
  else xdg-open "$1"; fi
}

# Open the UI once trace_processor answers through the tunnel.
(
  for _ in $(seq 1 90); do
    if curl -sf -o /dev/null "http://127.0.0.1:${PORT}/status"; then
      echo "Trace loaded. Opening $URL -> choose \"YES, use loaded trace\"."
      open_url "$URL"
      exit 0
    fi
    sleep 2
  done
  echo "trace_processor did not come up within 3 minutes." >&2
) &

exec ssh -tt -o ExitOnForwardFailure=yes -L "${PORT}:127.0.0.1:${PORT}" "$HOST" \
  "exec $TP server http --ip-address 127.0.0.1 --port ${PORT} --idle-timeout 1h $TRACE"
