"""NVTX step ranges for vLLM: one range per engine step, with exact prefill/decode token counts.

Loaded as a vLLM general plugin (entry point ``vllm.general_plugins``, see ../nvgx_nvtx-0.1.dist-info):
put the parent folder on ``PYTHONPATH`` and vLLM calls ``register()`` in every process it starts.

What it does
------------
Wraps ``vllm.v1.worker.gpu_worker.Worker.execute_model`` and ``Worker.sample_tokens`` (shared by the
V1 and V2 GPU model runners) in one NVTX push/pop range per step, in the domain ``nvgx``::

    step prefill_tokens=2048 decode_tokens=16 reqs=17 prefill_reqs=1 decode_reqs=16

The range is pushed when ``execute_model`` starts and popped when the step's ``sample_tokens``
returns (or right away if ``execute_model`` returned the output itself), so every kernel and CUDA
graph the step launches, forward pass and sampler, falls inside it. The Nsight Perf SDK projects
those launches onto the range (``nvtx_gpu_projection_segments``).

Counting rule (per scheduled request, from vLLM 0.33's ``SchedulerOutput``)
----------------------------------------------------------------------
* ``n`` = ``num_scheduled_tokens[req]``.
* Prefill, all ``n`` tokens: the request is new this step (``scheduled_new_reqs``), was resumed
  after preemption (``scheduled_cached_reqs.resumed_req_ids``: its tokens are recomputed), or has
  no output token yet (``num_output_tokens == 0``: a chunked-prefill continuation). This matches
  vLLM's own ``compute_iteration_details`` context/generation split, plus the resumed case.
* Decode, otherwise: the request already produces tokens, so ``1 + len(spec_decode_tokens[req])``
  tokens are decode (the next token plus any speculative drafts). Anything scheduled beyond that
  (e.g. prompt tokens appended to a streaming session) counts as prefill.
* A request with any prefill tokens counts in ``prefill_reqs``, otherwise in ``decode_reqs``.
  ``prefill_tokens + decode_tokens == total_num_scheduled_tokens``. ``spec_tokens=`` is appended
  only when speculative decoding schedules drafts.

Safety
------
* NVTX is only called when a step runs, which happens only in the engine-core / worker process,
  never in the API server. The ``nvtx`` domain (which initializes NVTX and loads the injection)
  is created lazily on the first step. So there is no NVTX use before vLLM forks its engine.
* Only CPU-side push/pop around Python calls: nothing is captured into CUDA graphs.
* Core push/pop only (no payload extension), one domain, strict nesting on the busy-loop thread.
* Cost per step: one pass over the scheduled requests plus one push and one pop (~10 us).

Environment
-----------
``NVGX_NVTX=0`` disables the hook. ``NVGX_NVTX_DOMAIN`` overrides the domain name (default ``nvgx``).
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.util
import logging
import os
import sys

__all__ = ["register", "count_step", "step_message"]

_TARGET = "vllm.v1.worker.gpu_worker"
_DOMAIN_NAME = os.environ.get("NVGX_NVTX_DOMAIN", "nvgx")
_MAX_CACHED_MESSAGES = 65536

log = logging.getLogger("vllm.nvgx_nvtx")  # child of vLLM's configured logger

# ARGB colors for the Perfetto / Nsight timelines.
_COLOR_PREFILL = 0xFFD95F02
_COLOR_DECODE = 0xFF1B9E77
_COLOR_MIXED = 0xFF7570B3


def count_step(so) -> tuple[int, int, int, int, int]:
    """Return (prefill_tokens, decode_tokens, prefill_reqs, decode_reqs, spec_tokens) for a SchedulerOutput."""
    nst = so.num_scheduled_tokens
    if not nst:
        return 0, 0, 0, 0, 0
    new_ids = {r.req_id for r in so.scheduled_new_reqs}
    cached = so.scheduled_cached_reqs
    out_tokens = dict(zip(cached.req_ids, cached.num_output_tokens))
    resumed = cached.resumed_req_ids
    spec = so.scheduled_spec_decode_tokens
    p_tok = d_tok = p_req = d_req = s_tok = 0
    for rid, n in nst.items():
        if rid in new_ids or rid in resumed or not out_tokens.get(rid, 0):
            p_tok += n
            p_req += 1
            continue
        drafts = spec.get(rid)
        k = len(drafts) if drafts else 0
        s_tok += k
        d = k + 1
        if n <= d:
            d_tok += n
            d_req += 1
        else:
            d_tok += d
            p_tok += n - d
            p_req += 1
    return p_tok, d_tok, p_req, d_req, s_tok


def step_message(so) -> tuple[str, int]:
    p_tok, d_tok, p_req, d_req, s_tok = count_step(so)
    msg = (
        f"step prefill_tokens={p_tok} decode_tokens={d_tok} reqs={p_req + d_req} "
        f"prefill_reqs={p_req} decode_reqs={d_req}"
    )
    if s_tok:
        msg += f" spec_tokens={s_tok}"
    color = _COLOR_MIXED if (p_tok and d_tok) else (_COLOR_PREFILL if p_tok else _COLOR_DECODE)
    return msg, color


class _Nvtx:
    """Lazily created NVTX domain; push/pop with a bounded cache of event attributes."""

    def __init__(self) -> None:
        self.ready = False
        self.enabled = False
        self.pending = False  # a step range is open, waiting for sample_tokens()
        self._attrs: dict[str, object] = {}

    def _init(self) -> None:
        self.ready = True
        try:
            from nvtx._lib.lib import (  # noqa: PLC0415  (import only in the engine process)
                Domain,
                EventAttributes,
                RegisteredString,
                dummy_domain,
                pop_range,
                push_range,
            )
        except Exception as e:  # pragma: no cover
            log.warning("nvgx_nvtx: nvtx package unavailable (%s); hook disabled", e)
            return
        dom = Domain(_DOMAIN_NAME)  # first NVTX call in this process: loads the injection
        if dom is dummy_domain:
            log.warning("nvgx_nvtx: no NVTX tool attached; hook is a no-op")
            return
        self._dom = dom
        self._handle = dom.handle
        self._EventAttributes = EventAttributes
        self._RegisteredString = RegisteredString
        self._push = push_range
        self._pop = pop_range
        self.enabled = True
        log.info("nvgx_nvtx: NVTX step ranges enabled in domain %r (pid %d)", _DOMAIN_NAME, os.getpid())

    def push(self, msg: str, color: int) -> None:
        attrs = self._attrs.get(msg)
        if attrs is None:
            if len(self._attrs) >= _MAX_CACHED_MESSAGES:
                self._attrs.clear()
            attrs = self._EventAttributes(self._dom, self._RegisteredString(self._handle, msg), color)
            self._attrs[msg] = attrs
        self._push(attrs, self._handle)

    def pop(self) -> None:
        self._pop(self._handle)


_nvtx = _Nvtx()


def _wrap_execute_model(orig):
    @functools.wraps(orig)
    def execute_model(self, scheduler_output, *args, **kwargs):
        st = _nvtx
        if not st.ready:
            st._init()
        if not st.enabled:
            return orig(self, scheduler_output, *args, **kwargs)
        if st.pending:  # unbalanced (should not happen): close the previous step first
            st.pending = False
            st.pop()
        if not scheduler_output.total_num_scheduled_tokens:
            return orig(self, scheduler_output, *args, **kwargs)
        msg, color = step_message(scheduler_output)
        st.push(msg, color)
        try:
            out = orig(self, scheduler_output, *args, **kwargs)
        except BaseException:
            st.pop()
            raise
        if out is None:
            st.pending = True  # sampling follows in sample_tokens()
        else:
            st.pop()
        return out

    execute_model.__nvgx_wrapped__ = True
    return execute_model


def _wrap_sample_tokens(orig):
    @functools.wraps(orig)
    def sample_tokens(self, *args, **kwargs):
        st = _nvtx
        if not st.pending:
            return orig(self, *args, **kwargs)
        try:
            return orig(self, *args, **kwargs)
        finally:
            st.pending = False
            st.pop()

    sample_tokens.__nvgx_wrapped__ = True
    return sample_tokens


def _patch(mod) -> None:
    worker = getattr(mod, "Worker", None)
    if worker is None or getattr(worker.execute_model, "__nvgx_wrapped__", False):
        return
    worker.execute_model = _wrap_execute_model(worker.execute_model)
    worker.sample_tokens = _wrap_sample_tokens(worker.sample_tokens)
    log.info("nvgx_nvtx: patched %s.Worker (pid %d)", _TARGET, os.getpid())


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Patch gpu_worker right after it is imported, without importing it ourselves.

    The API server process also loads general plugins; it never imports gpu_worker,
    so nothing is patched there and no NVTX call can happen in it.
    """

    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def register() -> None:
    """vLLM general-plugin entry point."""
    if os.environ.get("NVGX_NVTX", "1") == "0":
        return
    mod = sys.modules.get(_TARGET)
    if mod is not None:
        _patch(mod)
    elif not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())
