#!/usr/bin/env bash
# Run a vLLM test container on the CUDA 13.5 compat driver.
#   TRACED=1 (default) mounts the Nsight Perf service socket so the container is traced.
#   TRACED=0 runs the same container untraced (baseline).
# Start the root service first (scripts/start-root-service.sh); the container must start after it.
set -euo pipefail

LAB=${LAB:-$HOME/nsight-perf-lab}
NAME=${NAME:-nvgx-vllm-test}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.33.0}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
REVISION=${REVISION:-c1899de289a04d12100db370d81485cdf75e47ca}
SERVED=${SERVED:-qwen3-0.6b}
PORT=${PORT:-8010}
GPU_MEM=${GPU_MEM:-0.25}
TRACED=${TRACED:-1}

COMPAT="$LAB/compat-root/usr/local/cuda-13.5/compat"
[ -e "$COMPAT/libcuda.so.1" ] || { echo "missing $COMPAT (run scripts/setup-option-a.sh)"; exit 1; }

trace_args=()
if [ "$TRACED" = 1 ]; then
  [ -d /run/nvidia/nsight-perf ] || { echo "no root service socket in /run/nvidia/nsight-perf"; exit 1; }
  trace_args=(-v /run/nvidia/nsight-perf:/run/nvidia/nsight-perf)
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
# Compat folder first in LD_LIBRARY_PATH, then the image's own entries.
docker run -d --name "$NAME" --restart no --gpus all --ipc=host -p "127.0.0.1:${PORT}:8000" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$COMPAT:/opt/cuda-compat-13.5:ro" \
  "${trace_args[@]}" \
  -e LD_LIBRARY_PATH=/opt/cuda-compat-13.5:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib \
  -e HF_HUB_OFFLINE=1 \
  "$IMAGE" "$MODEL" --revision "$REVISION" --served-model-name "$SERVED" \
  --gpu-memory-utilization "$GPU_MEM" --max-model-len 4096

for _ in $(seq 1 90); do
  curl -sf -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null && { echo "$NAME ready on :$PORT (TRACED=$TRACED)"; exit 0; }
  sleep 5
done
echo "$NAME not ready after 450 s; see: docker logs $NAME"; exit 1
