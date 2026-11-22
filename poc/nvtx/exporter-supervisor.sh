#!/usr/bin/env bash
# Keep the forked nvtrace_continuous_export running: restart it if it exits, and run it under gdb so a crash
# leaves a backtrace in the log. Start detached:
#   setsid -f bash exporter-supervisor.sh > ~/nsight-perf-lab/logs/exporter-fleet.log 2>&1 < /dev/null
# Stop: kill "$(cat ~/nsight-perf-lab/logs/exporter-supervisor.pid)"; pkill -x nvtrace_continu
LAB=${LAB:-$HOME/nsight-perf-lab}
OUT=${OUT:-$LAB/poc/live2}
FLEET=${FLEET:-http://127.0.0.1:9100}
echo $$ > "$LAB/logs/exporter-supervisor.pid"
mkdir -p "$OUT"
while true; do
  echo "[supervisor] $(date -u +%FT%TZ) starting exporter"
  gdb -q -batch -ex "set pagination off" -ex "handle SIGPIPE nostop noprint pass" -ex run -ex "bt 30" \
    --args "$LAB/fork/build/bin/nvtrace_continuous_export" --snapshot-ms 5000 --stop-on-enter-key false \
    --duration-ms 21600000 --data-rate 32 --output-dir "$OUT" --fleet-url "$FLEET" 2>&1 |
    grep --line-buffered -vE '^\[(New Thread|Thread .* exited)'
  echo "[supervisor] $(date -u +%FT%TZ) exporter exited; restarting in 3 s"
  sleep 3
done
