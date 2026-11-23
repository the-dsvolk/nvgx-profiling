#!/usr/bin/env bash
# vLLM with tensor parallelism 2 across two DGX Sparks (one GPU each), NCCL 2.32.6 on both ranks.
#
#   run_vllm_tp2.sh head   <head-link-ip>   # node rank 0: API server on 127.0.0.1:$PORT, traced by the Nsight Perf SDK
#   run_vllm_tp2.sh worker <head-link-ip>   # node rank 1: headless, not traced
#
# Start the worker and the head in either order; each waits for the other.
#   TRACED=1  (head, default) mount the Nsight Perf service socket (start the root service first)
#   NVTX=1    (head, default) step ranges via poc/nvtx (same as scripts/run-vllm.sh NVTX=1)
#   INSPECTOR=1 load the NCCL Inspector profiler plugin ($LAB/libnccl-profiler-inspector.so), writing JSON
#               to $LAB/inspector/ every INSPECTOR_INTERVAL_US (default 1 s)
#
# Nothing is installed on the hosts. $LAB (default ~/nccl-lab) holds: compat/ (CUDA 13.5 compat libs),
# nccl-2.32.6/ (unpacked libnccl2 .deb), hf/ (Hugging Face cache, if not using ~/.cache/huggingface).
set -euo pipefail
ROLE=${1:?head|worker}; HEAD_IP=${2:?head node link ip}
LAB=${LAB:-$HOME/nccl-lab}
SDK_LAB=${SDK_LAB:-$HOME/nsight-perf-lab}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.33.0}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
REVISION=${REVISION:-c1899de289a04d12100db370d81485cdf75e47ca}
SERVED=${SERVED:-qwen3-0.6b-tp2}
PORT=${PORT:-8010}
GPU_MEM=${GPU_MEM:-0.10}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
IFACES=${NCCL_SOCKET_IFNAME:-enp1s0f0np0,enP2p1s0f0np0}
HCAS=${NCCL_IB_HCA:-rocep1s0f0,roceP2p1s0f0}
NCCL_SO=/lab/nccl-2.32.6/usr/lib/aarch64-linux-gnu/libnccl.so.2
NODE_IP=${NODE_IP:-$(ip -4 -o addr show "${IFACES%%,*}" | awk '{split($4, a, "/"); print a[1]}')}

case "$ROLE" in
  head)   NAME=${NAME:-nvgx-vllm-tp2-head};   TRACED=${TRACED:-1}; NVTX=${NVTX:-1}
          role_args=(--node-rank 0 --host 127.0.0.1 --port "$PORT") ;;
  worker) NAME=${NAME:-nvgx-vllm-tp2-worker}; TRACED=${TRACED:-0}; NVTX=${NVTX:-0}
          role_args=(--node-rank 1 --headless) ;;
  *) echo "role must be head or worker"; exit 1 ;;
esac
[ -e "$LAB/compat/libcuda.so.1" ] || { echo "missing $LAB/compat"; exit 1; }
[ -e "$LAB/${NCCL_SO#/lab/}" ] || { echo "missing $LAB/${NCCL_SO#/lab/}"; exit 1; }

devs=()
for d in /dev/infiniband/rdma_cm /dev/infiniband/uverbs*; do devs+=(--device="$d"); done
extra=()
if [ "$TRACED" = 1 ]; then
  [ -d /run/nvidia/nsight-perf ] || { echo "no root service socket in /run/nvidia/nsight-perf"; exit 1; }
  extra+=(-v /run/nvidia/nsight-perf:/run/nvidia/nsight-perf)
fi
if [ "$NVTX" = 1 ]; then
  HOOK_DIR=${HOOK_DIR:-$SDK_LAB/poc/nvtx/hook}
  NVTX_PROXY=${NVTX_PROXY:-$SDK_LAB/sdk/bin/libnvtx_cuda_driver_injection_proxy.so}
  [ -e "$HOOK_DIR/nvgx_nvtx/__init__.py" ] || { echo "missing $HOOK_DIR/nvgx_nvtx (set HOOK_DIR)"; exit 1; }
  extra+=(-v "$HOOK_DIR:/opt/nvgx-nvtx:ro" -e PYTHONPATH=/opt/nvgx-nvtx
          -v "$NVTX_PROXY:/opt/nvtx-proxy/libnvtx_cuda_driver_injection_proxy.so:ro"
          -e NVTX_INJECTION64_PATH=/opt/nvtx-proxy/libnvtx_cuda_driver_injection_proxy.so
          -v /run/nvidia/nvtx:/run/nvidia/nvtx:ro -e VLLM_NVTX_SCOPES_FOR_PROFILING=1)
fi
subsys=INIT,NET,ENV
if [ "${INSPECTOR:-0}" = 1 ]; then
  [ -e "$LAB/libnccl-profiler-inspector.so" ] || { echo "missing $LAB/libnccl-profiler-inspector.so"; exit 1; }
  mkdir -p "$LAB/inspector"
  extra+=(-e NCCL_PROFILER_PLUGIN=/lab/libnccl-profiler-inspector.so -e NCCL_INSPECTOR_ENABLE=1
          -e NCCL_INSPECTOR_DUMP_DIR=/lab/inspector
          -e NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS="${INSPECTOR_INTERVAL_US:-1000000}"
          -e NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES=0          # decode all-reduces are a few KB (default 8 KB)
          -e NCCL_INSPECTOR_DUMP_COLL_RING_SIZE=65536)     # ~60 all-reduces per decode step
  subsys=$subsys,PROFILE
else
  extra+=(-e NCCL_PROFILER_PLUGIN=none)
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart no --gpus all --network host --ipc=host \
  --cap-add IPC_LOCK --ulimit memlock=-1 "${devs[@]}" \
  -v "$LAB:/lab" -v "$HF_CACHE:/root/.cache/huggingface" \
  -e LD_LIBRARY_PATH=/lab/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib \
  -e LD_PRELOAD="$NCCL_SO" -e VLLM_NCCL_SO_PATH="$NCCL_SO" \
  -e NCCL_IB_HCA="$HCAS" -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME="$IFACES" -e GLOO_SOCKET_IFNAME="${IFACES%%,*}" -e VLLM_HOST_IP="$NODE_IP" \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS="$subsys" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e HF_HUB_OFFLINE=1 \
  ${extra[@]+"${extra[@]}"} \
  "$IMAGE" "$MODEL" --revision "$REVISION" --served-model-name "$SERVED" \
  --tensor-parallel-size 2 --distributed-executor-backend mp \
  --nnodes 2 --master-addr "$HEAD_IP" --master-port 29501 "${role_args[@]}" \
  --gpu-memory-utilization "$GPU_MEM" --max-model-len 4096
echo "$NAME started ($ROLE, node ip $NODE_IP, TRACED=$TRACED NVTX=$NVTX INSPECTOR=${INSPECTOR:-0}); logs: docker logs -f $NAME"
