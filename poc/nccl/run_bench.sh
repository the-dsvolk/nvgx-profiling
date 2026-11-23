#!/usr/bin/env bash
# Run one rank of allreduce_bench.py in the vLLM image over the direct RoCE link between two DGX Sparks.
#
#   run_bench.sh <rank> <rank0-link-ip> <label> [nccl-lib]
#     nccl-lib: path under $LAB of a libnccl.so.2 to force with LD_PRELOAD (e.g. an unpacked libnccl2 .deb);
#               omit to use the NCCL bundled in the image.
#   PROFILER_PLUGIN=<path under $LAB>  load an NCCL profiler plugin (e.g. the example plugin built from NCCL source),
#   PROFILE_EVENT_MASK=71              with this event mask (71 = Group | Coll | P2p | KernelCh, as in NCCL issue #2355).
# Start rank 1 first (it waits for rank 0), then rank 0. Results: $LAB/results/<label>-r<rank>.{json,log}
#
# Nothing is installed on the host: $LAB holds the CUDA 13.5 compat libs (compat/), NCCL builds and the script.
set -euo pipefail
RANK=${1:?rank}; MASTER=${2:?rank0 link ip}; LABEL=${3:?label}; NCCL_LIB=${4:-}
LAB=${LAB:-$HOME/nccl-lab}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.33.0}
IFACES=${NCCL_SOCKET_IFNAME:-enp1s0f0np0,enP2p1s0f0np0}   # both rails of the direct QSFP link
HCAS=${NCCL_IB_HCA:-rocep1s0f0,roceP2p1s0f0}

devs=()
for d in /dev/infiniband/rdma_cm /dev/infiniband/uverbs*; do devs+=(--device="$d"); done
preload=()
[ -n "$NCCL_LIB" ] && preload=(-e "LD_PRELOAD=/lab/$NCCL_LIB")
profiler=(-e NCCL_PROFILER_PLUGIN=none)
[ -n "${PROFILER_PLUGIN:-}" ] && profiler=(-e "NCCL_PROFILER_PLUGIN=/lab/$PROFILER_PLUGIN" \
  -e "NCCL_PROFILE_EVENT_MASK=${PROFILE_EVENT_MASK:-71}")
mkdir -p "$LAB/results"

exec docker run --rm --name "nccl-bench-r$RANK" --network host --gpus all --ipc=host \
  --cap-add IPC_LOCK --ulimit memlock=-1 "${devs[@]}" \
  -v "$LAB:/lab" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e LD_LIBRARY_PATH=/lab/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  -e NCCL_IB_HCA="$HCAS" -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME="$IFACES" -e GLOO_SOCKET_IFNAME="${IFACES%%,*}" \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET,ENV \
  "${preload[@]}" "${profiler[@]}" \
  --entrypoint python3 "$IMAGE" /lab/allreduce_bench.py \
  --rank "$RANK" --world 2 --master-addr "$MASTER" --label "$LABEL" \
  --out "/lab/results/$LABEL-r$RANK.json" > "$LAB/results/$LABEL-r$RANK.log" 2>&1
