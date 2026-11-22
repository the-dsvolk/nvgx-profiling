# Variable workload for the live dashboard: `nvgx_load.py`

A load generator for a vLLM (OpenAI-compatible) server that keeps changing what it asks the GPU to do,
so the always-on profiler and the [fleet dashboard](../README.md) see realistic, shifting behaviour:
prefill-heavy vs decode-heavy, light vs saturated, bursty vs steady, and idle gaps. It replaces the
fixed three-phase `vllm bench serve` loop used for the first live run.

One file, Python 3 standard library only. Default target: `http://127.0.0.1:8010`, model `qwen3-0.6b`
(Qwen3-0.6B, `--max-model-len 4096`).

## How the variability works

**Sessions.** A session is a conversation of 1..K turns. Each turn resends the whole conversation (a
shared system prompt, every earlier user message and answer) plus a new user message, so later turns
share a growing prefix (vLLM's prefix cache). Sessions think between turns, then continue or end.
The number of concurrent sessions follows a **target** that is re-drawn every few seconds (a random
walk inside the regime's range, with occasional jumps); when the target drops, sessions above it leave
at their next turn boundary. Total concurrency is capped at 64.

**Prompt length.** Random per request, from tens to ~3,000 tokens: lognormal (chat), uniform
1.2k–3k (RAG), or a bimodal short/long mix (burst, steady-high). Prompts are random English filler; the
tokens-per-word ratio is measured once with vLLM's `/tokenize`, and the server-reported
`usage.prompt_tokens` is logged next to the estimate (estimates land within a few percent).

**Output length.** Random `max_tokens` from its own distribution (16..512 typical, plus a 5–15% tail of
768..2,048-token generations). Most requests set vLLM's `ignore_eos` so the output really has that
length; the rest stop naturally. Prompt + output always stay below the 4,096-token context (the history
size is exact from the previous turn's `usage`; a conversation that is full ends).

**Arrivals.** `steady` regimes start sessions at a limited rate (1–8 per second) up to the target and
replace sessions as they end; `burst` alternates 3–12 s waves in which 24–64 sessions start at the same
instant with 8–30 s near-idle gaps; the `idle` regime sends nothing for 10–60 s.

**Regimes** (durations in seconds; `--speed 2` halves them):

| Regime | Sessions | Prompt tokens | `max_tokens` | Turns / think | Meant to exercise on the GPU |
|---|---|---|---|---|---|
| `idle` | 0 | | | | Idle baseline: no kernels, idle verdict. 10–60 s |
| `trickle` | 1–2, steady | LogN med 150 (20–1,200) | 64–384, 10%: 768–1,536 | 2–6 / 0.5–4 s | Tiny decode batches: launch- and latency-bound, low utilisation |
| `chat` | 2–40, steady | LogN med 200 (20–2,000), follow-ups med 60 | LogN med 160 (16–512), 5%: 768–1,536 | 2–8 / LogN med 3 s | Mixed prefill and decode, growing shared prefixes, varying batch size |
| `rag` | 4–32, steady | 1,200–3,000, follow-ups med 250 | 16–128 | 1–3 / 0.5–5 s | Prefill-heavy: FlashAttention and GEMM on long sequences, high TTFT |
| `decode` | 8–48, steady | LogN med 80 (20–400) | 256–512, 15%: 1,024–2,048 | 1–3 / 0–2 s | Decode-heavy: large steady decode batches, memory-bound GEMMs |
| `burst` | waves of 24–64, gaps of 0–2 | 80% LogN med 150, 20% 1,500–3,000 | LogN med 96 (16–512) | 1–2 / 0–1 s | Sudden prefill spikes, queueing and TTFT spikes, on/off utilisation |
| `steady-high` | 48–64, steady | 70% LogN med 200, 30% 1,200–3,000 | LogN med 200 (16–512), 5%: 768–1,536 | 1–6 / 0–0.5 s | Saturated server: maximum batch, mixed prefill/decode, KV-cache pressure |

**Scenarios** (`--scenario`): `mixed` (default) picks regimes at random, weighted chat 3, rag 2,
decode 2, burst 2, steady-high 2, trickle 1, never the same twice in a row, with a 40% chance of an
idle gap between two regimes. `chat`, `rag`, `decode`, `burst`, `steady-high` and `trickle` run that
single regime (re-drawing its duration and levels), with an occasional idle gap (none for
`steady-high`). `python3 nvgx_load.py --list` prints all parameters. Regime order, durations, levels
and every session's lengths derive from `--seed`, so a run's schedule is reproducible (server timing
still varies).

## Measurements

Every request is streamed. One JSONL line per request (`type: "req"`) in `--log`
(default `~/nsight-perf-lab/logs/nvgx-load.jsonl`):

```json
{"type":"req","t":1795316748.333,"regime":"rag","regime_seq":1,"session":17,"turn":2,"turns":2,"attempt":1,
 "prompt_tokens_approx":2399,"prompt_tokens":2402,"cached_tokens":null,"max_tokens":19,"ignore_eos":false,
 "output_tokens":19,"ttft_s":0.1774,"latency_s":1.1137,"tpot_ms":52.018,"finish":"length","status":"ok","error":null}
```

`ttft_s` is the time to the first content chunk, `tpot_ms` is (latency − TTFT) / (output tokens − 1),
`status` is `ok`, `http_<code>`, `conn_error`, `timeout` or `bad_stream`. `prompt_tokens` and
`output_tokens` come from the server's `usage`; `cached_tokens` stays `null` unless vLLM runs with
`--enable-prompt-tokens-details`. The log also holds
`regime`, `target`, `server_down` / `server_up`, `summary`, `start` and `stop` events, all with Unix
timestamps to line up with the dashboard. Every `--summary-interval` seconds (10) it prints one line,
and it prints each regime and target change:

```text
03:05:43 nvgx_load pid 3211323 scenario=mixed seed=2026 duration=forever url=http://127.0.0.1:8010 cap=64 speed=1 log=~/nsight-perf-lab/logs/nvgx-load.jsonl
03:05:43 calibrated filler: 1.074 tokens/word (via /tokenize)
03:05:43 === regime #1 rag for 164 s: long retrieved context (1.2k-3k tokens), short answers; sessions 4-32, steady ===
03:05:43     prompt U(1200..3000), follow-up LogN(med 250, s 0.6, 30..800), max_tokens U(16..128), turns 1-3
03:05:43     target sessions -> 22
03:05:53 [rag#1] sess 22/22 infl 16 |  3.20 req/s  out    210 tok/s  in   7224 tok/s | TTFT p50 0.127 p95 0.308 s | TPOT p50  23.0 ms | err 0 (total 0)
...
03:00:03 === regime #5 idle for 15 s: no traffic: GPU idle gap ===
03:00:03     target sessions -> 0
03:00:18 === regime #6 burst for 30 s: waves of 24-64 sessions starting at once, separated by near-idle gaps; sessions 24-64, burst ===
03:00:18     target sessions -> 57 (burst on)
03:00:20     target sessions -> 1 (burst off)
03:00:21 [burst#6] sess 53/ 1 infl 52 |  0.50 req/s  out     13 tok/s  in    788 tok/s | TTFT p50 1.371 p95 2.006 s | TPOT p50  73.9 ms | err 0 (total 0)
...
02:42:08 !!! server unreachable (conn_error: ConnectionResetError(104, 'Connection reset by peer')); requests paused, polling /v1/models
02:42:12 [rag#2] sess 24/24 infl  0 |  2.80 req/s  out    160 tok/s  in   5957 tok/s | TTFT p50 0.114 p95 0.183 s | TPOT p50  25.1 ms | err 1 (total 1)  SERVER DOWN
02:44:22 !!! server back after 100 s; resuming
```

## Robustness

- Connection refused/reset, 502/503 or a stream cut mid-response mark the server **down**: every session
  pauses before its next request while one thread polls `/v1/models` every 2 s; when the model answers
  again, sessions resume with jittered exponential backoff (1 s doubling to 30 s). Each failed attempt
  is logged and counted; a request is retried up to `--max-retries` (8) times. No new sessions start
  while the server is down, and when it returns, sessions above the current target leave instead of
  all resuming at once. A 400 (for example a
  context overflow) ends that session without retry.
- SIGTERM/SIGINT stop new requests, give in-flight requests 3 s (15 s at the end of `--duration`;
  requests still running are abandoned and vLLM aborts them), print a final summary and remove the
  PID file (`--pid-file`, default `~/nsight-perf-lab/poc/nvgx-load.pid`).

## Run and stop

```shell
python3 poc/workload/nvgx_load.py --list                                   # scenarios, regimes
python3 poc/workload/nvgx_load.py --scenario rag --duration 5m --seed 7    # short run, foreground

# long-running background load on the GPU node
setsid -f python3 ~/nsight-perf-lab/poc/workload/nvgx_load.py --scenario mixed --duration 6h \
  > ~/nsight-perf-lab/logs/nvgx-load.log 2>&1 < /dev/null
tail -f ~/nsight-perf-lab/logs/nvgx-load.log
kill $(cat ~/nsight-perf-lab/poc/nvgx-load.pid)                            # stop
```

Other options: `--url`, `--model`, `--max-model-len`, `--max-concurrency` (≤ 64), `--speed`
(compress regime durations for tests), `--summary-interval`, `--read-timeout`, `--max-retries`.

## Test runs

Short runs of every scenario on the DGX Spark (vLLM 0.33.0, Qwen3-0.6B, `--gpu-memory-utilization 0.25`,
profiler attached), 150 s each at `--speed 2` (mixed: 420 s at `--speed 3`), seed 42. Rates are averages
over the whole run, including the near-idle parts.

| Scenario | Requests | req/s | Output tok/s (peak 10 s) | Prompt tok/s | TTFT p50 / p95 | TPOT p50 | Peak sessions | Errors |
|---|---|---|---|---|---|---|---|---|
| `trickle` | 53 | 0.36 | 102 (206) | 249 | 0.029 / 0.044 s | 8.4 ms | 2 | 0 |
| `chat` | 514 | 3.4 | 759 (984) | 2,907 | 0.059 / 0.106 s | 17.6 ms | 34 | 0 |
| `rag` | 577 | 5.0 | 351 (516) | 11,794 | 0.115 / 0.220 s | 28.7 ms | 31 | 1 (container restart) |
| `decode` | 391 | 2.4 | 1,265 (1,905) | 1,118 | 0.071 / 0.152 s | 21.0 ms | 44 | 0 |
| `burst` | 990 | 6.4 | 835 (1,611) | 4,438 | 0.166 / 0.900 s | 27.7 ms | 47 | 0 |
| `steady-high` | 519 | 3.2 | 728 (998) | 4,575 | 0.251 / 0.441 s | 74.3 ms | 64 | 0 |
| `mixed` | 1,275 | 2.9 | 487 (987) | 3,358 | 0.211 / 1.233 s | 43.4 ms | 63 | 0 |

- The regimes separate clearly: `rag` pushes ~12k prompt tok/s with tiny outputs, `decode` the reverse;
  `trickle` gives single-digit-ms TPOT, `steady-high` saturates the server (TPOT ~70 ms, TTFT rising
  with 60+ requests in flight); `burst` waves push TTFT p95 to 1–2.5 s.
- Prompt-length estimates match the server's `usage.prompt_tokens` to within 0.2–5% (median); the longest
  prompts reached 4,017 tokens with no context overflow.
- **Container restart during the `rag` run:** the one in-flight request failed (`conn_error`), all
  sessions paused, `/v1/models` came back after ~100 s and the next run resumed without further errors.
