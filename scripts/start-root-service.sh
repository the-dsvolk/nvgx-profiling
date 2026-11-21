#!/bin/sh
# One-off root run of the Nsight Perf service (global scope). Nothing is installed.
# Root is needed on DGX Spark: counter profiling is admin-only (RmProfilingAdminOnly=1).
#
# Start it detached from the SSH terminal (sudo -b dies with the terminal):
#   sudo setsid -f ./scripts/start-root-service.sh "$HOME/nsight-perf-lab" [duration]
# Pass the lab folder explicitly: under sudo, $HOME is root's.
# The service stops by itself after [duration] (default 6h).
# Afterwards check:  journalctl -k --since "-5 min" | grep INSUFFICIENT   (should print nothing)
trap "" HUP
LAB=${1:?usage: start-root-service.sh <lab-dir> [duration]}
DURATION=${2:-6h}
mkdir -p "$LAB/logs"
cd "$LAB/sdk/bin" || exit 1
exec timeout "$DURATION" ./nvidia-nsight-perf --global > "$LAB/logs/service-root.log" 2>&1 < /dev/null
