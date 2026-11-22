#!/usr/bin/env bash
# NVTX overhead A/B: two traced vLLM containers, identical except NVTX=1 / NVTX=0, benchmarked side by side.
#   NAME=nvgx-nvtx-on  PORT=8011 GPU_MEM=0.12 NVTX=1 scripts/run-vllm.sh
#   NAME=nvgx-nvtx-off PORT=8012 GPU_MEM=0.12 NVTX=0 scripts/run-vllm.sh
#   poc/nvtx/overhead.sh [rounds]
# Each round runs the same benchmark on both containers at the same time, so they share the GPU and any
# background load equally (run one after the other, other traffic on the GPU swamped the difference).
set -euo pipefail
ROUNDS=${1:-4}
ON=${ON:-nvgx-nvtx-on}
OFF=${OFF:-nvgx-nvtx-off}
bench() {
  docker exec "$1" vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --model qwen3-0.6b \
    --tokenizer Qwen/Qwen3-0.6B --dataset-name random --random-input-len "$2" --random-output-len "$3" \
    --num-prompts 64 --max-concurrency 16 --seed "$4" 2>&1 |
    awk -v c="$1" -v w="$2/$3" '/Output token throughput/ {t=$NF} /Mean TPOT/ {p=$NF} END {printf "%s %s tok/s=%s tpot_ms=%s\n", c, w, t, p}'
}
for c in "$ON" "$OFF"; do bench "$c" 256 32 0 >/dev/null; done # warm-up
for r in $(seq 1 "$ROUNDS"); do
  for w in "1024 128" "128 512"; do
    bench "$ON" $w "$r" > /tmp/nvgx-ovh-on.$$ & bench "$OFF" $w "$r" > /tmp/nvgx-ovh-off.$$ & wait
    cat /tmp/nvgx-ovh-on.$$ /tmp/nvgx-ovh-off.$$
  done
done
rm -f /tmp/nvgx-ovh-on.$$ /tmp/nvgx-ovh-off.$$
