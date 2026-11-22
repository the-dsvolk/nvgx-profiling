"""Self-test for nvgx_nvtx: counting rule on real vLLM SchedulerOutput objects, plus hook cost.

Run inside the vLLM container (PYTHONPATH must contain poc/nvtx):
    docker exec -e PYTHONPATH=/opt/nvgx-nvtx nvgx-vllm-test python3 /opt/nvgx-nvtx/selftest.py
"""

import time

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

import nvgx_nvtx


def new_req(rid, prompt_len, computed=0):
    return NewRequestData(
        req_id=rid, prompt_token_ids=list(range(prompt_len)), mm_features=[], sampling_params=None,
        pooling_params=None, block_ids=([],), num_computed_tokens=computed, lora_request=None,
    )


def make(new, cached, nst, resumed=(), spec=None):
    ids = [c[0] for c in cached]
    return SchedulerOutput(
        scheduled_new_reqs=new,
        scheduled_cached_reqs=CachedRequestData(
            req_ids=ids, resumed_req_ids=set(resumed), new_token_ids=[[] for _ in ids], all_token_ids={},
            new_block_ids=[None for _ in ids], num_computed_tokens=[0 for _ in ids],
            num_output_tokens=[c[1] for c in cached],
        ),
        num_scheduled_tokens=nst, total_num_scheduled_tokens=sum(nst.values()),
        scheduled_spec_decode_tokens=spec or {}, scheduled_encoder_inputs={}, finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


# 1 new request (first 1024-token chunk), 1 chunked-prefill continuation, 16 decodes.
dec = [(f"d{i}", 5 + i) for i in range(16)]
so = make([new_req("n0", 1024)], [("c0", 0)] + dec, {"n0": 1024, "c0": 1008, **{r: 1 for r, _ in dec}})
assert nvgx_nvtx.count_step(so) == (2032, 16, 2, 16, 0), nvgx_nvtx.count_step(so)
print(nvgx_nvtx.step_message(so)[0])

# Decode only.
so = make([], dec, {r: 1 for r, _ in dec})
assert nvgx_nvtx.count_step(so) == (0, 16, 0, 16, 0)
print(nvgx_nvtx.step_message(so)[0])

# Resumed after preemption (has outputs, recomputes 300 tokens) + a spec-decode request (1 + 3 drafts).
so = make([], [("r0", 40), ("s0", 9)], {"r0": 300, "s0": 4}, resumed={"r0"}, spec={"s0": [1, 2, 3]})
assert nvgx_nvtx.count_step(so) == (300, 4, 1, 1, 3), nvgx_nvtx.count_step(so)
print(nvgx_nvtx.step_message(so)[0])

# Streaming-session request with outputs that got 50 new prompt tokens: 1 decode + 49 prefill.
so = make([], [("x0", 12)], {"x0": 50})
assert nvgx_nvtx.count_step(so) == (49, 1, 1, 0, 0)

# Cost: message building for a 256-request decode batch, and push/pop (real NVTX if a tool is attached).
big = make([], [(f"d{i}", 3) for i in range(256)], {f"d{i}": 1 for i in range(256)})
n = 20000
t = time.perf_counter()
for _ in range(n):
    nvgx_nvtx.step_message(big)
print(f"step_message, 256 reqs: {(time.perf_counter() - t) / n * 1e6:.1f} us")
st = nvgx_nvtx._nvtx
st._init()
print(f"NVTX enabled: {st.enabled}")
if st.enabled:
    msg, color = nvgx_nvtx.step_message(big)
    t = time.perf_counter()
    for _ in range(n):
        st.push(msg, color)
        st.pop()
    print(f"push+pop (cached message): {(time.perf_counter() - t) / n * 1e6:.2f} us")
print("selftest OK")
