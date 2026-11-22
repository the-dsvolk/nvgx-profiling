#!/usr/bin/env python3
"""nvgx_load.py: variable, session-based load generator for a vLLM OpenAI-compatible server.

Standard library only. Simulates multi-turn chat sessions whose number, prompt lengths, output
lengths, arrival pattern and idle gaps change over time ("regimes"), so a continuous GPU profiler
sees prefill-heavy, decode-heavy, light, saturated, bursty and idle periods.

Every request is streamed; per-request measurements go to a JSONL log, and a one-line summary is
printed every --summary-interval seconds. Survives server restarts (health-gated retries with
backoff) and shuts down cleanly on SIGTERM/SIGINT.

  python3 nvgx_load.py --scenario mixed                 # run until stopped
  python3 nvgx_load.py --scenario rag --duration 5m --seed 7
  python3 nvgx_load.py --list                           # scenarios and regimes
"""

import argparse
import http.client
import json
import math
import os
import random
import signal
import socket
import sys
import threading
import time
import urllib.parse

MAX_CONCURRENCY = 64  # hard cap on concurrent sessions (keeps the GPU usable)

# ---------------------------------------------------------------------------------------------
# Distributions: tuples, sampled with sample_int / sample_float
#   ("fixed", v) | ("uniform", lo, hi) | ("lognormal", median, sigma, lo, hi) | ("mix", [(w, dist), ...])
# ---------------------------------------------------------------------------------------------


def _sample(dist, rng):
    kind = dist[0]
    if kind == "fixed":
        return dist[1]
    if kind == "uniform":
        return rng.uniform(dist[1], dist[2])
    if kind == "lognormal":
        _, median, sigma, lo, hi = dist
        return min(hi, max(lo, median * math.exp(rng.gauss(0.0, sigma))))
    if kind == "mix":
        parts = dist[1]
        x = rng.random() * sum(w for w, _ in parts)
        for w, d in parts:
            x -= w
            if x <= 0:
                return _sample(d, rng)
        return _sample(parts[-1][1], rng)
    raise ValueError(f"unknown distribution {dist!r}")


def sample_int(dist, rng):
    return int(round(_sample(dist, rng)))


def sample_float(dist, rng):
    return float(_sample(dist, rng))


def dist_str(dist):
    kind = dist[0]
    if kind == "fixed":
        return f"{dist[1]}"
    if kind == "uniform":
        return f"U({dist[1]:g}..{dist[2]:g})"
    if kind == "lognormal":
        return f"LogN(med {dist[1]:g}, s {dist[2]:g}, {dist[3]:g}..{dist[4]:g})"
    return " | ".join(f"{w:g}:{dist_str(d)}" for w, d in dist[1])


# ---------------------------------------------------------------------------------------------
# Regimes: one traffic shape each. Durations in seconds (scaled by --speed).
#   level         target concurrent sessions (lo, hi); re-drawn every level_period seconds
#   arrival       "steady": sessions start at up to `ramp` per second to reach the target;
#                 "burst":  target alternates between an on-level (all sessions start at once)
#                           and an off-level (off_level), for on_s / off_s seconds
#   turns         turns per session (lo, hi); later turns resend the whole conversation
#   think         think time between turns (s)
#   first         tokens in the first user message; followup: in later user messages
#   output        max_tokens; ignore_eos: probability the request ignores EOS (exact length)
# ---------------------------------------------------------------------------------------------

REGIMES = {
    "idle": dict(
        desc="no traffic: GPU idle gap",
        exercises="idle baseline; dashboard should show no kernels / idle verdict",
        level=(0, 0), dur=(10, 60),
    ),
    "trickle": dict(
        desc="1-2 sequential chat sessions",
        exercises="tiny decode batches: launch-/latency-bound, low GPU utilisation",
        level=(1, 2), level_period=(20, 60), arrival="steady", ramp=1.0,
        turns=(2, 6), think=("uniform", 0.5, 4.0),
        first=("lognormal", 150, 0.7, 20, 1200), followup=("lognormal", 60, 0.6, 10, 400),
        output=("mix", [(0.9, ("uniform", 64, 384)), (0.1, ("uniform", 768, 1536))]),
        ignore_eos=0.85, dur=(60, 150),
    ),
    "chat": dict(
        desc="multi-turn chat, wandering 2-40 sessions",
        exercises="mixed prefill+decode with growing shared prefixes (prefix cache), varying batch",
        level=(2, 40), level_period=(10, 30), arrival="steady", ramp=4.0,
        turns=(2, 8), think=("lognormal", 3.0, 0.7, 0.3, 15.0),
        first=("lognormal", 200, 0.9, 20, 2000), followup=("lognormal", 60, 0.7, 10, 600),
        output=("mix", [(0.95, ("lognormal", 160, 0.7, 16, 512)), (0.05, ("uniform", 768, 1536))]),
        ignore_eos=0.7, dur=(90, 240),
    ),
    "rag": dict(
        desc="long retrieved context (1.2k-3k tokens), short answers",
        exercises="prefill-heavy: FlashAttention/GEMM on long sequences, high TTFT",
        level=(4, 32), level_period=(10, 30), arrival="steady", ramp=6.0,
        turns=(1, 3), think=("uniform", 0.5, 5.0),
        first=("uniform", 1200, 3000), followup=("lognormal", 250, 0.6, 30, 800),
        output=("uniform", 16, 128),
        ignore_eos=0.9, dur=(60, 180),
    ),
    "decode": dict(
        desc="short prompts, long generations (256-2048 tokens)",
        exercises="decode-heavy: steady large decode batches, memory-bound GEMV-like GEMMs",
        level=(8, 48), level_period=(15, 40), arrival="steady", ramp=6.0,
        turns=(1, 3), think=("uniform", 0.0, 2.0),
        first=("lognormal", 80, 0.6, 20, 400), followup=("lognormal", 40, 0.5, 10, 200),
        output=("mix", [(0.85, ("uniform", 256, 512)), (0.15, ("uniform", 1024, 2048))]),
        ignore_eos=0.95, dur=(60, 180),
    ),
    "burst": dict(
        desc="waves of 24-64 sessions starting at once, separated by near-idle gaps",
        exercises="sudden prefill spikes then decode tails; queueing, TTFT spikes, on/off utilisation",
        level=(24, 64), off_level=(0, 2), on_s=(3, 12), off_s=(8, 30), arrival="burst",
        turns=(1, 2), think=("uniform", 0.0, 1.0),
        first=("mix", [(0.8, ("lognormal", 150, 0.8, 20, 1000)), (0.2, ("uniform", 1500, 3000))]),
        followup=("lognormal", 60, 0.6, 10, 300),
        output=("lognormal", 96, 0.8, 16, 512),
        ignore_eos=0.85, dur=(60, 180),
    ),
    "steady-high": dict(
        desc="48-64 sessions, no think time, short/long prompt mix",
        exercises="saturated server: max batch, mixed prefill/decode, KV-cache pressure",
        level=(48, 64), level_period=(15, 40), arrival="steady", ramp=8.0,
        turns=(1, 6), think=("uniform", 0.0, 0.5),
        first=("mix", [(0.7, ("lognormal", 200, 0.8, 20, 1200)), (0.3, ("uniform", 1200, 3000))]),
        followup=("lognormal", 80, 0.6, 10, 400),
        output=("mix", [(0.95, ("lognormal", 200, 0.6, 16, 512)), (0.05, ("uniform", 768, 1536))]),
        ignore_eos=0.85, dur=(90, 240),
    ),
}

# Scenarios: which regimes to cycle through (weights) and the chance of an idle gap between them.
SCENARIOS = {
    "mixed": dict(desc="default: random walk through all regimes with idle gaps",
                  regimes={"chat": 3, "rag": 2, "decode": 2, "burst": 2, "steady-high": 2, "trickle": 1},
                  idle_p=0.4),
    "chat": dict(desc="chat sessions only (occasional idle gap)", regimes={"chat": 1}, idle_p=0.15),
    "rag": dict(desc="long prompts, short answers (occasional idle gap)", regimes={"rag": 1}, idle_p=0.15),
    "decode": dict(desc="short prompts, long generations", regimes={"decode": 1}, idle_p=0.15),
    "burst": dict(desc="bursts of simultaneous sessions", regimes={"burst": 1}, idle_p=0.1),
    "steady-high": dict(desc="saturating steady load, no idle", regimes={"steady-high": 1}, idle_p=0.0),
    "trickle": dict(desc="one or two sequential sessions", regimes={"trickle": 1}, idle_p=0.15),
}

SYSTEM_PROMPT = ("You are a helpful assistant. Answer the user's questions about the notes they "
                 "provide. Be thorough and keep writing until you have covered every point.")

WORDS = """time year people way day man thing woman life child world school state family student
group country problem hand part place case week company system program question work government
number night point home water room mother area money story fact month lot right study book eye job
word business issue side kind head house service friend father power hour game line end member law
car city community name president team minute idea kid body information back parent face others
level office door health person art war history party result change morning reason research girl
guy moment air teacher force education foot boy age policy process music market sense nation plan
college interest death experience effect use class control care field development role effort rate
heart drug show leader light voice wife police mind price report decision son view relationship town
road arm difference value building action model season society tax director position player record
paper space ground form event official matter center couple site project activity star table need
court oil situation cost industry figure street image phone data picture practice piece land product
doctor wall patient worker news test movie north love support technology step baby computer type
attention film tree source organization hair window evidence population site truth network memory
kernel memory cache tensor batch token latency throughput profile graph launch stream sync budget
measure signal metric window sample vector matrix layer engine queue buffer thread schedule driver
quickly slowly carefully often never always rarely clearly simply really almost early later today
make take give find tell ask seem feel try leave call keep hold bring begin show hear play run move
live believe happen write provide sit stand lose pay meet include continue set learn change lead
understand watch follow stop create speak read allow add spend grow open walk win offer remember
consider appear buy wait serve die send expect build stay fall cut reach kill remain suggest raise
new good high old great big small large long little own other right early young important few
public bad same able last late hard major better economic strong possible whole free military true
federal international full special easy clear recent certain personal open red difficult available
the of and to a in for is on that by this with you it not or be are from at as your all have
""".split()


# ---------------------------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------------------------


def now_hms():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"{now_hms()} {msg}", flush=True)


def pct(values, p):
    if not values:
        return float("nan")
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1))))]


class Health:
    """Gate requests on server health. Sessions block in wait_up() while the server is down;
    a background thread polls /v1/models until it answers again."""

    def __init__(self, gen):
        self.g = gen
        self.up = threading.Event()
        self.down_since = None
        self.lock = threading.Lock()

    def mark_down(self, why):
        with self.lock:
            if self.up.is_set():
                self.up.clear()
                self.down_since = time.time()
                self.g.stats.outages += 1
                log(f"!!! server unreachable ({why}); requests paused, polling /v1/models")
                self.g.event("server_down", reason=why)

    def wait_up(self):
        while not self.up.wait(1.0):
            if self.g.stop.is_set():
                return False
        return True

    def check(self):
        try:
            status, body = self.g.http("GET", "/v1/models", None, timeout=5)
            return status == 200 and self.g.args.model in body
        except Exception:
            return False

    def run(self):
        while not self.g.stop.is_set():
            if self.up.is_set():
                self.g.stop.wait(1.0)
                continue
            if self.check():
                with self.lock:
                    down = time.time() - self.down_since if self.down_since else 0.0
                    self.up.set()
                if self.down_since:
                    log(f"!!! server back after {down:.0f} s; resuming")
                    self.g.event("server_up", down_s=round(down, 1))
                self.down_since = None
                continue
            self.g.stop.wait(2.0)


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.window = []  # completed request records since last summary
        self.total_req = 0
        self.total_ok = 0
        self.total_err = 0
        self.total_out = 0
        self.outages = 0
        self.inflight = 0

    def add(self, rec):
        with self.lock:
            self.window.append(rec)
            self.total_req += 1
            if rec["status"] == "ok":
                self.total_ok += 1
                self.total_out += rec.get("output_tokens") or 0
            else:
                self.total_err += 1

    def take(self):
        with self.lock:
            w, self.window = self.window, []
            return w


class Gen:
    def __init__(self, args):
        self.args = args
        u = urllib.parse.urlparse(args.url)
        self.host, self.port = u.hostname, u.port or 80
        self.stop = threading.Event()
        self.stats = Stats()
        self.health = Health(self)
        self.lock = threading.Lock()  # guards active / target / epoch / regime
        self.active = 0
        self.target = 0
        self.epoch = 0
        self.regime_name = "start"
        self.regime_seq = 0
        self.session_seq = 0
        self.tokens_per_word = 1.25
        self.log_lock = threading.Lock()
        self.logf = open(os.path.expanduser(args.log), "a", buffering=1)
        self.t0 = time.time()

    # -- output ---------------------------------------------------------------------------------
    def write(self, rec):
        line = json.dumps(rec, separators=(",", ":"))
        with self.log_lock:
            self.logf.write(line + "\n")

    def event(self, kind, **kw):
        self.write(dict(type=kind, t=round(time.time(), 3), regime=self.regime_name,
                        regime_seq=self.regime_seq, **kw))

    # -- HTTP -----------------------------------------------------------------------------------
    def http(self, method, path, body, timeout=10):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            data = json.dumps(body).encode() if body is not None else None
            conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            return r.status, r.read(1 << 20).decode("utf-8", "replace")
        finally:
            conn.close()

    def calibrate(self):
        """Measure tokens per filler word with /tokenize (falls back to 1.25)."""
        self.tokens_per_word = 1.0
        text = self.filler(2000, random.Random(12345))  # same generator as the prompts: 2000 words
        self.tokens_per_word = 1.25
        try:
            status, body = self.http("POST", "/tokenize", {"model": self.args.model, "prompt": text})
            if status == 200:
                self.tokens_per_word = json.loads(body)["count"] / len(text.split())
                log(f"calibrated filler: {self.tokens_per_word:.3f} tokens/word (via /tokenize)")
                return
        except Exception as e:  # noqa: BLE001
            log(f"/tokenize failed ({e}); using {self.tokens_per_word} tokens/word")

    def filler(self, n_tokens, rng):
        n_words = max(1, int(round(n_tokens / self.tokens_per_word)))
        words = [rng.choice(WORDS) for _ in range(n_words)]
        # sentence-ish structure: capitalise and punctuate every 8-20 words
        out, i = [], 0
        while i < n_words:
            k = rng.randint(8, 20)
            chunk = words[i:i + k]
            chunk[0] = chunk[0].capitalize()
            out.append(" ".join(chunk) + ".")
            i += k
        return " ".join(out)

    def est_tokens(self, text):
        return int(len(text.split()) * self.tokens_per_word)

    # -- session bookkeeping ------------------------------------------------------------------
    def should_leave(self, epoch):
        """Called at turn boundaries. True (and releases the slot) if the session must end:
        shutdown, a new regime started, or more sessions are active than the target."""
        with self.lock:
            if self.stop.is_set() or epoch != self.epoch or self.active > self.target:
                self.active -= 1
                return True
        return False

    def spawn(self, regime):
        with self.lock:
            self.active += 1
            self.session_seq += 1
            sid = self.session_seq
            epoch = self.epoch
        rng = random.Random(f"{self.args.seed}-s{sid}")
        threading.Thread(target=Session(self, sid, regime, epoch, rng).run, daemon=True,
                         name=f"session-{sid}").start()

    # -- streaming request ----------------------------------------------------------------------
    def stream_chat(self, messages, max_tokens, ignore_eos, rng):
        """One streamed chat completion. Returns a dict of measurements (+ 'text')."""
        body = {
            "model": self.args.model, "messages": messages, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True},
            "temperature": round(rng.uniform(0.6, 1.0), 2), "ignore_eos": ignore_eos,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        res = dict(status="ok", error=None, ttft_s=None, latency_s=None, output_tokens=0,
                   prompt_tokens=None, cached_tokens=None, finish_reason=None, text="", http=None)
        pieces, chunks = [], 0
        t_start = time.time()
        res["t"] = round(t_start, 3)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.args.read_timeout)
        try:
            conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                         headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            r = conn.getresponse()
            res["http"] = r.status
            if r.status != 200:
                res["status"] = f"http_{r.status}"
                res["error"] = r.read(2000).decode("utf-8", "replace")[:300]
                return res
            for raw in r:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                obj = json.loads(payload)
                if obj.get("usage"):
                    u = obj["usage"]
                    res["prompt_tokens"] = u.get("prompt_tokens")
                    res["output_tokens"] = u.get("completion_tokens") or 0
                    det = u.get("prompt_tokens_details") or {}
                    res["cached_tokens"] = det.get("cached_tokens")
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    txt = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                    if txt:
                        if res["ttft_s"] is None:
                            res["ttft_s"] = time.time() - t_start
                        pieces.append(txt)
                        chunks += 1
                    if ch.get("finish_reason"):
                        res["finish_reason"] = ch["finish_reason"]
        except (socket.timeout, TimeoutError) as e:
            res["status"], res["error"] = "timeout", repr(e)[:200]
        except (ConnectionError, http.client.HTTPException, OSError) as e:
            res["status"], res["error"] = "conn_error", repr(e)[:200]
        except ValueError as e:  # bad JSON in the stream
            res["status"], res["error"] = "bad_stream", repr(e)[:200]
        finally:
            conn.close()
            res["latency_s"] = time.time() - t_start
        if not res["output_tokens"]:
            res["output_tokens"] = chunks  # ~1 token per chunk when usage is missing
        res["text"] = "".join(pieces)
        return res


class Session:
    CTX_MARGIN = 48  # tokens kept free below --max-model-len
    TEMPLATE_OVERHEAD = 8  # chat-template tokens per message (approx)

    def __init__(self, g, sid, regime, epoch, rng):
        self.g, self.sid, self.regime, self.epoch, self.rng = g, sid, regime, epoch, rng
        self.rname = g.regime_name
        self.rseq = g.regime_seq

    def run(self):
        g, R, rng = self.g, self.regime, self.rng
        released = False
        try:
            turns = rng.randint(*R["turns"])
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            hist = g.est_tokens(SYSTEM_PROMPT) + self.TEMPLATE_OVERHEAD  # exact after turn 1
            budget = g.args.max_model_len - self.CTX_MARGIN
            for turn in range(1, turns + 1):
                if turn > 1:
                    if g.stop.wait(sample_float(R["think"], rng)):
                        break
                elif R.get("arrival") == "steady":
                    if g.stop.wait(rng.uniform(0, 1.0)):  # de-synchronise steady starts
                        break
                if not g.health.wait_up():  # server restarting: wait, then re-check the target
                    break
                if g.should_leave(self.epoch):
                    released = True
                    return
                n_new = sample_int(R["first"] if turn == 1 else R["followup"], rng)
                max_tokens = sample_int(R["output"], rng)
                # respect the context (hist is exact after turn 1; 8% slack on new filler text):
                # shrink the new message to fit, then the output
                room = budget - hist - self.TEMPLATE_OVERHEAD
                n_new = min(n_new, int((room - 16) / 1.08))
                if n_new < 10:
                    break  # conversation is full
                max_tokens = max(16, min(max_tokens, room - int(n_new * 1.08)))
                if turn == 1:
                    text = f"Notes #{self.sid}: " + g.filler(n_new, rng)
                else:
                    text = g.filler(n_new, rng) + " What do you think about this?"
                messages.append({"role": "user", "content": text})
                approx = hist + self.TEMPLATE_OVERHEAD + n_new
                ignore_eos = rng.random() < R.get("ignore_eos", 0.8)
                res = self.request(messages, max_tokens, ignore_eos, approx, turn, turns)
                if res is None or res["status"] != "ok":
                    break
                messages.append({"role": "assistant", "content": res["text"]})
                if res["prompt_tokens"]:
                    hist = res["prompt_tokens"] + res["output_tokens"] + self.TEMPLATE_OVERHEAD
                else:
                    hist = approx + res["output_tokens"] + self.TEMPLATE_OVERHEAD
        finally:
            if not released:
                with g.lock:
                    g.active -= 1

    def request(self, messages, max_tokens, ignore_eos, approx, turn, turns):
        """Send one turn; on server errors wait for health + exponential backoff and retry."""
        g = self.g
        for attempt in range(1, g.args.max_retries + 2):
            if not g.health.wait_up():
                return None
            with g.stats.lock:
                g.stats.inflight += 1
            try:
                res = g.stream_chat(messages, max_tokens, ignore_eos, self.rng)
            finally:
                with g.stats.lock:
                    g.stats.inflight -= 1
            out = res["output_tokens"]
            tpot = None
            if res["status"] == "ok" and res["ttft_s"] is not None and out > 1:
                tpot = (res["latency_s"] - res["ttft_s"]) / (out - 1) * 1000.0
            rec = dict(type="req", t=res["t"], regime=self.rname, regime_seq=self.rseq,
                       session=self.sid, turn=turn, turns=turns, attempt=attempt,
                       prompt_tokens_approx=approx, prompt_tokens=res["prompt_tokens"],
                       cached_tokens=res["cached_tokens"], max_tokens=max_tokens,
                       ignore_eos=ignore_eos, output_tokens=out,
                       ttft_s=None if res["ttft_s"] is None else round(res["ttft_s"], 4),
                       latency_s=round(res["latency_s"], 4),
                       tpot_ms=None if tpot is None else round(tpot, 3),
                       finish=res["finish_reason"], status=res["status"], error=res["error"])
            g.write(rec)
            g.stats.add(rec)
            st = res["status"]
            if st == "ok":
                return res
            if g.stop.is_set():
                return res
            retryable = st in ("conn_error", "timeout", "bad_stream") or (
                st.startswith("http_") and (res["http"] or 0) >= 500 or res["http"] == 429)
            if not retryable:
                return res  # e.g. 400 context too long: give up on this session
            if st == "conn_error" or res["http"] in (502, 503):
                g.health.mark_down(f"{st}: {(res['error'] or '')[:80]}")
            backoff = min(30.0, 1.0 * 2 ** (attempt - 1)) * self.rng.uniform(0.5, 1.5)
            if g.stop.wait(backoff):
                return res
        return res


# ---------------------------------------------------------------------------------------------
# Controller: picks regimes, drives the target number of sessions, spawns sessions
# ---------------------------------------------------------------------------------------------


class Controller:
    def __init__(self, g):
        self.g = g
        self.args = g.args
        self.rng = random.Random(f"{self.args.seed}-controller")
        self.scenario = SCENARIOS[self.args.scenario]
        self.cap = min(MAX_CONCURRENCY, self.args.max_concurrency)

    def scaled(self, lohi):
        return self.rng.uniform(*lohi) / self.args.speed

    def pick_next(self, prev):
        if prev not in (None, "idle") and self.rng.random() < self.scenario["idle_p"]:
            return "idle"
        names = list(self.scenario["regimes"])
        weights = [self.scenario["regimes"][n] for n in names]
        if len(names) > 1 and prev in names:  # avoid repeating the same regime back to back
            i = names.index(prev)
            names.pop(i)
            weights.pop(i)
        return self.rng.choices(names, weights)[0]

    def set_target(self, n, why=""):
        n = max(0, min(self.cap, n))
        with self.g.lock:
            changed = n != self.g.target
            self.g.target = n
        if changed:
            log(f"    target sessions -> {n}{why}")
            self.g.event("target", target=n)

    def run(self, t_end):
        prev = None
        while not self.g.stop.is_set() and time.time() < t_end:
            name = self.pick_next(prev)
            R = REGIMES[name]
            dur = min(self.scaled(R["dur"]), max(0.0, t_end - time.time()))
            g = self.g
            with g.lock:
                if name != prev:
                    g.epoch += 1  # sessions of the previous regime leave at their next turn
                g.regime_seq += 1
                g.regime_name = name
            lvl = "" if name == "idle" else f"; sessions {R['level'][0]}-{R['level'][1]}, {R.get('arrival')}"
            log(f"=== regime #{g.regime_seq} {name} for {dur:.0f} s: {R['desc']}{lvl} ===")
            if name != "idle":
                log(f"    prompt {dist_str(R['first'])}, follow-up {dist_str(R['followup'])}, "
                    f"max_tokens {dist_str(R['output'])}, turns {R['turns'][0]}-{R['turns'][1]}")
            g.event("regime", duration_s=round(dur, 1), desc=R["desc"], exercises=R["exercises"])
            self.run_regime(R, time.time() + dur)
            prev = name

    def run_regime(self, R, end):
        g, rng = self.g, self.rng
        if R["level"] == (0, 0):
            self.set_target(0)
            g.stop.wait(max(0.0, end - time.time()))
            return
        burst = R.get("arrival") == "burst"
        lo, hi = R["level"]
        next_change = 0.0
        burst_on = False
        next_spawn = 0.0
        while not g.stop.is_set():
            t = time.time()
            if t >= end:
                return
            if t >= next_change:
                if burst:
                    burst_on = not burst_on
                    if burst_on:
                        self.set_target(rng.randint(lo, hi), " (burst on)")
                        next_change = t + self.scaled(R["on_s"])
                    else:
                        self.set_target(rng.randint(*R["off_level"]), " (burst off)")
                        next_change = t + self.scaled(R["off_s"])
                else:
                    with g.lock:
                        cur = g.target
                    if cur < lo or cur > hi or rng.random() < 0.3:
                        new = rng.randint(lo, hi)  # jump to a random level
                    else:
                        step = max(1, (hi - lo) // 4)
                        new = cur + rng.randint(-step, step)  # random walk, reflected at the bounds
                        new = lo + abs(new - lo) if new < lo else hi - abs(new - hi) if new > hi else new
                    self.set_target(new)
                    next_change = t + self.scaled(R["level_period"])
            with g.lock:
                deficit = min(g.target, self.cap) - g.active
            if deficit > 0 and g.health.up.is_set():  # no new sessions while the server is down
                if burst:
                    for _ in range(deficit):  # everybody at once
                        g.spawn(R)
                elif t >= next_spawn:
                    g.spawn(R)
                    next_spawn = t + 1.0 / R.get("ramp", 4.0)
            g.stop.wait(0.05)


# ---------------------------------------------------------------------------------------------
# Summary thread
# ---------------------------------------------------------------------------------------------


def summary_loop(g):
    interval = g.args.summary_interval
    last = time.time()
    while not g.stop.wait(interval):
        summarize(g, last)
        last = time.time()


def summarize(g, last, final=False):
    w = g.stats.take()
    dt = max(1e-3, time.time() - last)
    ok = [r for r in w if r["status"] == "ok"]
    errs = len(w) - len(ok)
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    tpot = [r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]
    out_tok = sum(r["output_tokens"] for r in ok)
    in_tok = sum((r["prompt_tokens"] or r["prompt_tokens_approx"]) for r in ok)
    with g.lock:
        active, target, regime, seq = g.active, g.target, g.regime_name, g.regime_seq
    s = g.stats
    down = "" if g.health.up.is_set() else "  SERVER DOWN"
    log(f"[{regime}#{seq}] sess {active:2d}/{target:2d} infl {s.inflight:2d} | "
        f"{len(ok) / dt:5.2f} req/s  out {out_tok / dt:6.0f} tok/s  in {in_tok / dt:6.0f} tok/s | "
        f"TTFT p50 {pct(ttft, 50):.3f} p95 {pct(ttft, 95):.3f} s | TPOT p50 {pct(tpot, 50):5.1f} ms | "
        f"err {errs} (total {s.total_err}){down}")
    g.write(dict(type="summary", t=round(time.time(), 3), regime=regime, regime_seq=seq,
                 window_s=round(dt, 2), active=active, target=target, inflight=s.inflight,
                 req_ok=len(ok), errors=errs, req_per_s=round(len(ok) / dt, 3),
                 out_tok_per_s=round(out_tok / dt, 1), in_tok_per_s=round(in_tok / dt, 1),
                 ttft_p50=None if not ttft else round(pct(ttft, 50), 4),
                 ttft_p95=None if not ttft else round(pct(ttft, 95), 4),
                 tpot_p50_ms=None if not tpot else round(pct(tpot, 50), 3), final=final))


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------


def parse_duration(s):
    s = str(s).strip().lower()
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s and s[-1] in mult:
        return float(s[:-1]) * mult[s[-1]]
    return float(s)


def list_all():
    print("Scenarios:")
    for n, sc in SCENARIOS.items():
        rs = ", ".join(f"{k}:{v}" for k, v in sc["regimes"].items())
        print(f"  {n:12s} {sc['desc']}  [regimes {rs}; idle gap p={sc['idle_p']}]")
    print("\nRegimes:")
    for n, R in REGIMES.items():
        print(f"  {n:12s} {R['desc']}  ({R['dur'][0]}-{R['dur'][1]} s)")
        print(f"  {'':12s} exercises: {R['exercises']}")
        if n != "idle":
            print(f"  {'':12s} sessions {R['level'][0]}-{R['level'][1]} ({R['arrival']}), "
                  f"turns {R['turns'][0]}-{R['turns'][1]}, think {dist_str(R['think'])} s")
            print(f"  {'':12s} prompt {dist_str(R['first'])}, follow-up {dist_str(R['followup'])}")
            print(f"  {'':12s} max_tokens {dist_str(R['output'])}, ignore_eos p={R['ignore_eos']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="mixed", choices=sorted(SCENARIOS))
    ap.add_argument("--duration", default="0", help="run time, e.g. 300, 10m, 6h; 0 = until stopped")
    ap.add_argument("--seed", type=int, default=None, help="random seed (default: time-based, printed)")
    ap.add_argument("--url", default="http://127.0.0.1:8010", help="OpenAI-compatible server base URL")
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--max-model-len", type=int, default=4096, help="server context; prompt+output stay below")
    ap.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY,
                    help=f"cap on concurrent sessions (hard max {MAX_CONCURRENCY})")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="time compression for regime/level durations (2 = twice as fast; for tests)")
    ap.add_argument("--log", default="~/nsight-perf-lab/logs/nvgx-load.jsonl", help="JSONL output (appended)")
    ap.add_argument("--pid-file", default="~/nsight-perf-lab/poc/nvgx-load.pid")
    ap.add_argument("--summary-interval", type=float, default=10.0)
    ap.add_argument("--read-timeout", type=float, default=180.0, help="socket timeout per request read (s)")
    ap.add_argument("--max-retries", type=int, default=8, help="retries per request on server errors")
    ap.add_argument("--list", action="store_true", help="list scenarios and regimes, then exit")
    args = ap.parse_args()
    if args.list:
        list_all()
        return 0
    if args.seed is None:
        args.seed = int(time.time()) % 1000000
    duration = parse_duration(args.duration)

    pid_file = os.path.expanduser(args.pid_file)
    os.makedirs(os.path.dirname(pid_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.expanduser(args.log)) or ".", exist_ok=True)
    with open(pid_file, "w") as f:
        f.write(f"{os.getpid()}\n")

    g = Gen(args)
    signals = {}

    def on_signal(signum, _frame):
        signals["sig"] = signum
        g.stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    log(f"nvgx_load pid {os.getpid()} scenario={args.scenario} seed={args.seed} "
        f"duration={'forever' if duration <= 0 else f'{duration:.0f} s'} url={args.url} "
        f"cap={min(MAX_CONCURRENCY, args.max_concurrency)} speed={args.speed:g} log={args.log}")
    g.event("start", scenario=args.scenario, seed=args.seed, duration_s=duration, pid=os.getpid(),
            url=args.url, model=args.model)
    try:
        if g.health.check():
            g.health.up.set()
        else:
            g.health.down_since = time.time()
            log("server not ready yet; waiting for /v1/models")
        threading.Thread(target=g.health.run, daemon=True, name="health").start()
        if not g.health.wait_up():
            return 0
        g.calibrate()
        threading.Thread(target=summary_loop, args=(g,), daemon=True, name="summary").start()
        t_end = time.time() + duration if duration > 0 else float("inf")
        Controller(g).run(t_end)
    finally:
        g.stop.set()
        with g.lock:
            g.target = 0
        grace = 3.0 if signals else 15.0
        t_wait = time.time() + grace
        while g.stats.inflight > 0 and time.time() < t_wait:
            time.sleep(0.1)
        why = f"signal {signals['sig']}" if signals else "duration reached"
        summarize(g, time.time() - g.args.summary_interval, final=True)
        s = g.stats
        elapsed = time.time() - g.t0
        log(f"stopped ({why}) after {elapsed:.0f} s: {s.total_req} requests, {s.total_ok} ok, "
            f"{s.total_err} errors, {s.total_out} output tokens, {s.outages} outage(s), "
            f"{s.inflight} request(s) abandoned in flight")
        g.event("stop", reason=why, elapsed_s=round(elapsed, 1), requests=s.total_req, ok=s.total_ok,
                errors=s.total_err, output_tokens=s.total_out, outages=s.outages)
        g.logf.close()
        try:
            with open(pid_file) as f:
                if f.read().strip() == str(os.getpid()):
                    os.remove(pid_file)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
