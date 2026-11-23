"use strict";
// nvgx-fleet live dashboard (POC). Vanilla JS + SVG. Data: /v1/profiles (history) + /events (live).

const MAX_POINTS = 180; // 15 minutes of 5 s snapshots
const FAMILIES = [
  ["gemm", "GEMM (CUTLASS / nvjet)"], ["flash_attention", "FlashAttention"], ["pytorch_native", "PyTorch native"],
  ["flashinfer_sampling", "FlashInfer sampling"], ["vllm_custom", "vLLM custom"], ["other", "Other"],
  ["unresolved", "Unresolved names"], ["nccl", "NCCL collectives"],
];
const VERDICTS = {
  idle: ["Idle", "GPU busy < 10%"],
  compute_bound: ["Compute-bound", "SM or tensor utilization ≥ 50%"],
  copy_bound: ["Copy-bound", "memcpy ≥ 20% of the window"],
  launch_sync_bound: ["Launch / sync-bound", "busy GPU, tiny kernels, CPU waiting on syncs"],
  host_bound: ["Host-bound", "GPU gaps: the CPU isn't feeding it"],
  mixed: ["Mixed", "no single bottleneck"],
};
const SVG = "http://www.w3.org/2000/svg";
const seriesColor = (i) => `var(--series-${i + 1})`;
const NCCL_COLOR = seriesColor(7); // "nccl" is family slot 8: same entity color in the family and phase charts
// Step phases from the NVTX hook (poc/nvtx); "unattributed" stays neutral gray.
const PHASES = [
  ["decode", "Decode-only steps", seriesColor(0)],
  ["mixed", "Mixed steps (prefill + decode)", seriesColor(1)],
  ["prefill", "Prefill-only steps", seriesColor(2)],
  ["unattributed", "Outside any step", "var(--text-muted)"],
];
// Tensor parallel: NCCL time is taken out of the phases above and stacked as its own segment.
const PHASES_WITH_NCCL = [...PHASES, ["nccl", "NCCL all-reduce (any step)", NCCL_COLOR]];
// Latency histograms from FleetSink: fixed log2 buckets, bucket i = [2^i, 2^(i+1)) us (21 buckets, last >= ~1 s).
const STEP_SERIES = [
  ["decode", "Decode-only steps", seriesColor(0)],
  ["mixed", "Mixed steps", seriesColor(1)],
  ["prefill", "Prefill-only steps", seriesColor(2)],
];
const HIST_METRICS = [
  { key: "nccl", label: "NCCL kernel duration", field: "histograms.nccl_kernel_us",
    series: [...STEP_SERIES, ["unattributed", "Outside any step", "var(--text-muted)"]],
    desc: "GPU time of each NCCL collective kernel, by the engine step it ran in. Includes waiting for the other ranks, so a long tail means a late peer, not a slow link." },
  { key: "interval", label: "Engine step interval", field: "histograms.step_interval_us", series: STEP_SERIES,
    desc: "Time from one vLLM engine step's start to the next (NVTX step ranges, same thread), by the later step's type: the real per-step cadence, CPU and GPU, i.e. what each decode token waits for. Gaps of 1 s or more (idle) are left out." },
  { key: "step", label: "Engine step CPU time", field: "histograms.step_us", series: STEP_SERIES,
    desc: "Host-side time of each engine step's NVTX range (execute_model → sample_tokens). With async scheduling the GPU work continues after the range closes, so this is CPU time, not token latency." },
  { key: "launch", label: "Launch → GPU start", field: "histograms.launch_to_start_us", series: null,
    desc: "Time from the CPU launch call (or CUDA graph launch) to the kernel starting on the GPU: queueing on the GPU." },
  { key: "sync", label: "CPU sync wait", field: "histograms.sync_wait_us", series: null,
    desc: "Duration of each cuda*Synchronize call: how long the host blocks waiting for the GPU." },
];
const HIST_BUCKETS = 21;
const HIST_RECENT = 12; // histogram panel: sum of the last 12 snapshots (60 s)
let histMetric = "nccl";

let allProfiles = []; // every node
let profiles = [];    // the selected node's, used by all charts and tables
let selectedNode = null;
let renderQueued = false;

// ---------- helpers ----------
function h(tag, attrs = {}, parent = null, text = null) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function s(tag, attrs = {}, parent = null, style = null) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (style) Object.assign(e.style, style);
  if (parent) parent.appendChild(e);
  return e;
}
const get = (o, path) => path.split(".").reduce((a, k) => (a == null ? a : a[k]), o);
const fmtNum = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : v >= 100 ? Math.round(v).toLocaleString() : v.toFixed(d));
const fmtPct = (v) => (v == null ? "–" : `${fmtNum(v)}%`);
const fmtMs = (v) => (v == null ? "–" : `${fmtNum(v)} ms`);
const fmtS = (v) => (v == null ? "–" : `${fmtNum(v)} s`);
const timeOf = (p) => new Date(p.window.end_utc_ns / 1e6);
const timeLabel = (p) => timeOf(p).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

function niceMax(v) {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}
function ticks(max) {
  // Pick a tick count whose step is a round number (1, 2, 2.5 or 5 × 10^k).
  const isNice = (step) => {
    const m = step / Math.pow(10, Math.floor(Math.log10(step)));
    return [1, 2, 2.5, 5, 10].some((x) => Math.abs(m - x) < 1e-9);
  };
  const n = [4, 5, 3, 6, 2].find((k) => isNice(max / k)) || 4;
  return Array.from({ length: n + 1 }, (_, i) => (max * i) / n);
}
function tickLabel(v) {
  if (v >= 10000) return `${Math.round(v / 1000)}K`;
  return v >= 10 || v === 0 ? Math.round(v).toLocaleString() : v.toFixed(1);
}

function tooltip(card) {
  let t = card.querySelector(".tooltip");
  if (!t) t = h("div", { class: "tooltip", role: "status" }, card);
  return t;
}
function placeTooltip(t, card, x, y) {
  t.style.display = "block";
  const cw = card.clientWidth, tw = t.offsetWidth;
  t.style.left = `${Math.min(Math.max(8, x + 14), cw - tw - 8)}px`;
  t.style.top = `${Math.max(8, y - 10)}px`;
}
function fillTooltip(t, title, rows) {
  t.replaceChildren();
  h("div", { class: "t" }, t, title);
  for (const r of rows) {
    const row = h("div", { class: "row" }, t);
    const key = h("i", {}, row);
    key.style.background = r.color;
    h("b", {}, row, r.value);
    h("span", {}, row, r.label);
  }
}
function legend(el, items, kind) {
  el.replaceChildren();
  for (const it of items) {
    const sp = h("span", {}, el);
    const sw = h("i", kind === "line" ? { class: "line" } : {}, sp);
    sw.style.background = it.color;
    sp.appendChild(document.createTextNode(it.label));
  }
}

// ---------- charts ----------
function frame(container, height, m) {
  container.replaceChildren();
  const width = Math.max(320, container.clientWidth);
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img" }, container);
  return { svg, width, height, pw: width - m.l - m.r, ph: height - m.t - m.b, m };
}

function axes(f, yMax, n, fmtY) {
  const { svg, m, pw, ph } = f;
  for (const v of ticks(yMax)) {
    const y = m.t + ph - (v / yMax) * ph;
    s("line", { x1: m.l, x2: m.l + pw, y1: y, y2: y, "stroke-width": 1 }, svg, { stroke: v === 0 ? "var(--axis)" : "var(--grid)" });
    const t = s("text", { x: m.l - 6, y: y + 4, "text-anchor": "end" }, svg);
    t.textContent = fmtY ? fmtY(v) : tickLabel(v);
  }
  const labels = Math.min(5, n);
  for (let k = 0; k < labels; k++) {
    const i = labels === 1 ? n - 1 : Math.round((k * (n - 1)) / (labels - 1));
    const x = m.l + (n <= 1 ? pw / 2 : (i * pw) / (n - 1));
    const t = s("text", { x, y: m.t + ph + 16, "text-anchor": k === 0 && labels > 1 ? "start" : k === labels - 1 ? "end" : "middle" }, svg);
    t.textContent = timeLabel(profiles[i]);
  }
}

function lineChart(container, legendEl, { series, height = 210, yMax, fmt, ref }) {
  const card = container.closest(".card");
  const n = profiles.length;
  if (!n) { container.replaceChildren(h("div", { class: "empty" }, null, "No data yet")); return; }
  const f = frame(container, height, { l: 44, r: series.length > 1 ? 120 : 64, t: 10, b: 24 });
  const { svg, m, pw, ph } = f;
  const vals = series.map((sr) => profiles.map((p) => { const v = sr.get(p); return v == null ? null : +v; }));
  const top = yMax || niceMax(Math.max(ref ? ref.value : 0, ...vals.flat().filter((v) => v != null)) * 1.1);
  const xs = (i) => m.l + (n <= 1 ? pw / 2 : (i * pw) / (n - 1));
  const ys = (v) => m.t + ph - (Math.min(v, top) / top) * ph;
  axes(f, top, n, fmt.axis);
  if (ref) {
    const y = ys(ref.value);
    s("line", { x1: m.l, x2: m.l + pw, y1: y, y2: y, "stroke-width": 1 }, svg, { stroke: "var(--text-muted)" });
    const t = s("text", { x: m.l + pw + 6, y: y + 4, class: "dlabel" }, svg);
    t.textContent = ref.label;
  }
  const ends = [];
  series.forEach((sr, si) => {
    let d = "", pen = false;
    vals[si].forEach((v, i) => {
      if (v == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`;
      pen = true;
    });
    if (d) s("path", { d, fill: "none", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg, { stroke: sr.color });
    const li = vals[si].map((v, i) => (v == null ? -1 : i)).filter((i) => i >= 0).pop();
    if (li !== undefined) {
      s("circle", { cx: xs(li), cy: ys(vals[si][li]), r: 4, "stroke-width": 2 }, svg, { fill: sr.color, stroke: "var(--surface-1)" });
      ends.push({ y: ys(vals[si][li]), x: xs(li), text: series.length > 1 ? `${sr.label} ${fmt.value(vals[si][li])}` : fmt.value(vals[si][li]) });
    }
  });
  // Direct end labels (supplement the legend); skip them if they would collide.
  const sorted = [...ends].sort((a, b) => a.y - b.y);
  const collide = sorted.some((e, i) => i > 0 && e.y - sorted[i - 1].y < 13);
  if (!collide) for (const e of ends) { const t = s("text", { x: e.x + 8, y: e.y + 4, class: "dlabel" }, svg); t.textContent = e.text; }
  if (legendEl) legend(legendEl, series.map((sr) => ({ label: sr.label, color: sr.color })), "line");

  // Crosshair + one tooltip listing every series.
  const cross = s("line", { y1: m.t, y2: m.t + ph, "stroke-width": 1, visibility: "hidden" }, svg, { stroke: "var(--text-muted)" });
  const hit = s("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" }, svg);
  const tt = tooltip(card);
  hit.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const sx = ((ev.clientX - r.left) / r.width) * f.width;
    const i = n <= 1 ? 0 : Math.max(0, Math.min(n - 1, Math.round(((sx - m.l) / pw) * (n - 1))));
    cross.setAttribute("x1", xs(i)); cross.setAttribute("x2", xs(i)); cross.setAttribute("visibility", "visible");
    fillTooltip(tt, timeLabel(profiles[i]), series.map((sr, si) => ({ color: sr.color, value: fmt.value(vals[si][i]), label: sr.label })));
    const cr = card.getBoundingClientRect();
    placeTooltip(tt, card, ev.clientX - cr.left, ev.clientY - cr.top);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tt.style.display = "none"; });
}

function stackedChart(container, legendEl, { series, field, height = 230 }) {
  // series: [[key, label, color]]; field: profile path of the {key: ms} object.
  const card = container.closest(".card");
  const n = profiles.length;
  legend(legendEl, series.map(([, label, color]) => ({ label, color })), "rect");
  if (!n) { container.replaceChildren(h("div", { class: "empty" }, null, "No data yet")); return; }
  const f = frame(container, height, { l: 48, r: 12, t: 10, b: 24 });
  const { svg, m, pw, ph } = f;
  // A snapshot can cover more than 5 s (e.g. the first one after an exporter restart): show it per 5 s.
  const norm = (p) => { const d = get(p, "window.duration_s"); return d && d > 5.5 ? 5 / d : 1; };
  const data = profiles.map((p) => series.map(([k]) => (get(p, `${field}.${k}`) || 0) * norm(p)));
  const totals = data.map((r) => r.reduce((a, b) => a + b, 0));
  const top = niceMax(Math.max(...totals, 1) * 1.05);
  axes(f, top, n, null);
  const band = pw / Math.max(n, 1);
  const colW = Math.max(1, Math.min(24, band * 0.7));
  const ys = (v) => (v / top) * ph;
  const cols = [];
  data.forEach((row, i) => {
    const cx = m.l + (n <= 1 ? pw / 2 : (i * pw) / (n - 1));
    const x = Math.max(m.l, Math.min(m.l + pw - colW, cx - colW / 2));
    const g = s("g", {}, svg);
    let base = m.t + ph;
    const visible = row.map((v, k) => [v, k]).filter(([v]) => ys(v) >= 0.5);
    visible.forEach(([v, k], j) => {
      const isTop = j === visible.length - 1;
      const gap = colW >= 6 && !isTop ? 2 : 0;
      const hgt = Math.max(0.5, ys(v) - gap);
      const yTop = base - ys(v) + (isTop ? 0 : gap);
      if (isTop && hgt > 4 && colW >= 8) {
        const r = 4;
        s("path", { d: `M${x},${base}V${yTop + r}Q${x},${yTop} ${x + r},${yTop}H${x + colW - r}Q${x + colW},${yTop} ${x + colW},${yTop + r}V${base}Z` }, g, { fill: series[k][2] });
      } else {
        s("rect", { x, y: yTop, width: colW, height: hgt }, g, { fill: series[k][2] });
      }
      base -= ys(v);
    });
    cols.push(g);
  });
  // One tooltip per column (thin columns are hard to hit; the band is the target).
  const tt = tooltip(card);
  const hit = s("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" }, svg);
  let last = -1;
  hit.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const sx = ((ev.clientX - r.left) / r.width) * f.width;
    const i = n <= 1 ? 0 : Math.max(0, Math.min(n - 1, Math.round(((sx - m.l) / pw) * (n - 1))));
    if (last >= 0 && cols[last]) cols[last].style.opacity = "";
    cols[i].style.opacity = "0.75"; last = i;
    const rows = series.map(([, label, color], k) => ({ color, label, v: data[i][k] }))
      .filter((rw) => rw.v > 0).map((rw) => ({ color: rw.color, label: `${rw.label} · ${totals[i] ? Math.round((100 * rw.v) / totals[i]) : 0}%`, value: fmtMs(rw.v) }));
    fillTooltip(tt, `${timeLabel(profiles[i])} · ${fmtMs(totals[i])} total`, rows.length ? rows : [{ color: "transparent", label: "no kernels", value: "0 ms" }]);
    const cr = card.getBoundingClientRect();
    placeTooltip(tt, card, ev.clientX - cr.left, ev.clientY - cr.top);
  });
  hit.addEventListener("pointerleave", () => { if (last >= 0 && cols[last]) cols[last].style.opacity = ""; tt.style.display = "none"; });
}

// ---------- latency histograms ----------
function bucketEdge(i) { // lower edge of bucket i, as text
  const us = Math.pow(2, i);
  if (us < 1000) return `${us} µs`;
  if (us < 1e6) { const ms = us / 1000; return `${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms`; }
  return `${(us / 1e6).toFixed(1)} s`;
}
const bucketRange = (i) => (i === 0 ? `< 2 µs` : i === HIST_BUCKETS - 1 ? `≥ ${bucketEdge(i)}` : `${bucketEdge(i)} – ${bucketEdge(i + 1)}`);
// Per profile: [series][bucket] counts, or null if the profile has no such histogram.
function histOf(p, m) {
  const h = get(p, m.field);
  if (!h) return null;
  if (!m.series) return Array.isArray(h) ? [h] : null;
  return m.series.map(([k]) => h[k] || new Array(HIST_BUCKETS).fill(0));
}
function quantileBucket(counts, q) {
  const tot = counts.reduce((a, b) => a + b, 0);
  if (!tot) return null;
  let cum = 0;
  for (let i = 0; i < counts.length; i++) { cum += counts[i]; if (cum >= q * tot) return i; }
  return counts.length - 1;
}

function renderHist() {
  const m = HIST_METRICS.find((x) => x.key === histMetric) || HIST_METRICS[0];
  const sel = document.getElementById("histSel");
  if (!sel.options.length) {
    for (const x of HIST_METRICS) h("option", { value: x.key }, sel, x.label);
    sel.addEventListener("change", () => { histMetric = sel.value; scheduleRender(); });
  }
  sel.value = m.key;
  document.getElementById("hist-desc").textContent = m.desc;
  const series = m.series || [["all", m.label, seriesColor(0)]];
  legend(document.getElementById("lg-hist"), series.map(([, label, color]) => ({ label, color })), "rect");
  const histEl = document.getElementById("ch-hist"), heatEl = document.getElementById("ch-heat");
  const per = profiles.map((p) => histOf(p, m));
  const have = per.filter(Boolean);
  if (!have.length) {
    for (const el of [histEl, heatEl]) el.replaceChildren(h("div", { class: "empty" }, null, "No histograms yet (needs the exporter build with latency histograms)"));
    return;
  }
  // Shared bucket range for both panels: buckets holding at least 0.1% of the visible window's events (stray
  // outliers don't stretch the axis; they still count in shares and quantiles), at least 8 wide.
  const colTot = (hs) => (hs ? hs[0].map((_, b) => hs.reduce((a, s) => a + s[b], 0)) : null);
  const totals = per.map(colTot);
  const winTot = new Array(HIST_BUCKETS).fill(0);
  for (const t of totals) if (t) t.forEach((c, b) => { winTot[b] += c; });
  const winSum = winTot.reduce((a, b) => a + b, 0);
  let lo = HIST_BUCKETS, hi = -1;
  winTot.forEach((c, b) => { if (c && c >= 0.001 * winSum) { lo = Math.min(lo, b); hi = Math.max(hi, b); } });
  if (hi < 0) { lo = 0; hi = 7; }
  while (hi - lo < 7) { if (lo > 0) lo--; if (hi - lo < 7 && hi < HIST_BUCKETS - 1) hi++; if (lo === 0 && hi === HIST_BUCKETS - 1) break; }
  const nb = hi - lo + 1;

  // Panel 1: sum of the last HIST_RECENT snapshots, stacked by series, as share of events.
  const recent = per.slice(-HIST_RECENT).filter(Boolean);
  document.getElementById("hist-sub").textContent = `Last ${recent.length * 5} s (${recent.length} snapshots)`;
  const agg = series.map((_, si) => Array.from({ length: HIST_BUCKETS }, (_, b) => recent.reduce((a, hs) => a + hs[si][b], 0)));
  const aggTot = agg[0].map((_, b) => agg.reduce((a, s) => a + s[b], 0));
  const events = aggTot.reduce((a, b) => a + b, 0);
  {
    const card = histEl.closest(".card");
    const f = frame(histEl, 252, { l: 40, r: 10, t: 34, b: 30 });
    const { svg, m: mg, pw, ph } = f;
    const shares = aggTot.slice(lo, hi + 1).map((c) => (events ? (100 * c) / events : 0));
    const top = niceMax(Math.max(...shares, 1) * 1.1);
    for (const v of ticks(top)) {
      const y = mg.t + ph - (v / top) * ph;
      s("line", { x1: mg.l, x2: mg.l + pw, y1: y, y2: y, "stroke-width": 1 }, svg, { stroke: v === 0 ? "var(--axis)" : "var(--grid)" });
      s("text", { x: mg.l - 6, y: y + 4, "text-anchor": "end" }, svg).textContent = `${tickLabel(v)}%`;
    }
    const band = pw / nb, bw = Math.max(2, band - 2);
    for (let k = 0; k < nb; k++) {
      const b = lo + k, x = mg.l + k * band + 1;
      let base = mg.t + ph;
      series.forEach(([, , color], si) => {
        const v = events ? (100 * agg[si][b]) / events : 0;
        const hgt = (v / top) * ph;
        if (hgt < 0.5) return;
        s("rect", { x, y: base - hgt, width: bw, height: Math.max(0.5, hgt - (si ? 1 : 0)), rx: 1.5 }, svg, { fill: color });
        base -= hgt;
      });
      if (k % 2 === 0 || nb <= 10) {
        s("text", { x: x + bw / 2, y: mg.t + ph + 14, "text-anchor": "middle" }, svg).textContent = bucketEdge(b);
      }
    }
    // p50 / p99 markers (bucket resolution), labels on separate lines so adjacent buckets don't collide.
    [[0.5, "p50", 22], [0.99, "p99", 8]].forEach(([q, name, dy]) => {
      const qb = quantileBucket(aggTot, q);
      if (qb == null || qb < lo || qb > hi) return;
      const x = mg.l + (qb - lo + 0.5) * band;
      s("line", { x1: x, x2: x, y1: mg.t - dy + 4, y2: mg.t + ph, "stroke-width": 1, "stroke-dasharray": "3 3" }, svg, { stroke: "var(--text-secondary)" });
      const anchor = x < mg.l + pw * 0.25 ? "start" : x > mg.l + pw * 0.75 ? "end" : "middle";
      s("text", { x: anchor === "start" ? x - 4 : anchor === "end" ? x + 4 : x, y: mg.t - dy, "text-anchor": anchor, class: "dlabel" }, svg).textContent = `${name} ${bucketRange(qb)}`;
    });
    s("text", { x: mg.l + pw, y: mg.t + ph + 28, "text-anchor": "end" }, svg).textContent = `${events.toLocaleString()} events · log scale`;
    const hit = s("rect", { x: mg.l, y: mg.t, width: pw, height: ph, fill: "transparent" }, svg);
    const tt = tooltip(card);
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const k = Math.max(0, Math.min(nb - 1, Math.floor((((ev.clientX - r.left) / r.width) * f.width - mg.l) / band)));
      const b = lo + k;
      fillTooltip(tt, `${bucketRange(b)} · ${events ? fmtPct((100 * aggTot[b]) / events) : "–"} of events`,
        series.map(([, label, color], si) => ({ color, label, value: agg[si][b].toLocaleString() })));
      const cr = card.getBoundingClientRect();
      placeTooltip(tt, card, ev.clientX - cr.left, ev.clientY - cr.top);
    });
    hit.addEventListener("pointerleave", () => { tt.style.display = "none"; });
  }

  // Panel 2: heatmap, time x bucket; shade = share of that snapshot's events; line = median bucket.
  {
    const card = heatEl.closest(".card");
    const n = profiles.length;
    const f = frame(heatEl, 252, { l: 56, r: 10, t: 34, b: 30 });
    const { svg, m: mg, pw, ph } = f;
    const cw = pw / Math.max(n, 1), rh = ph / nb;
    const shareAt = (i, b) => { const t = totals[i]; if (!t) return 0; const sum = t.reduce((a, c) => a + c, 0); return sum ? t[b] / sum : 0; };
    let maxShare = 0;
    for (let i = 0; i < n; i++) for (let b = lo; b <= hi; b++) maxShare = Math.max(maxShare, shareAt(i, b));
    for (let i = 0; i < n; i++) {
      if (!totals[i]) continue;
      for (let b = lo; b <= hi; b++) {
        const v = shareAt(i, b);
        if (!v) continue;
        s("rect", { x: mg.l + i * cw, y: mg.t + (hi - b) * rh, width: Math.max(1, cw - (cw > 4 ? 1 : 0)), height: Math.max(1, rh - 1) }, svg,
          { fill: "var(--series-1)", fillOpacity: String(0.08 + 0.92 * Math.sqrt(v / (maxShare || 1))) });
      }
    }
    // Median line.
    let d = "", pen = false;
    for (let i = 0; i < n; i++) {
      const qb = totals[i] ? quantileBucket(totals[i], 0.5) : null;
      if (qb == null) { pen = false; continue; }
      const x = mg.l + (i + 0.5) * cw, y = mg.t + (hi - qb + 0.5) * rh;
      d += `${pen ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
      pen = true;
    }
    if (d) s("path", { d, fill: "none", "stroke-width": 1.5, "stroke-linejoin": "round" }, svg, { stroke: "var(--text-primary)" });
    for (let b = lo; b <= hi; b += nb > 10 ? 2 : 1) {
      s("text", { x: mg.l - 6, y: mg.t + (hi - b + 0.5) * rh + 4, "text-anchor": "end" }, svg).textContent = bucketEdge(b);
    }
    const labels = Math.min(5, n);
    for (let k = 0; k < labels; k++) {
      const i = labels === 1 ? n - 1 : Math.round((k * (n - 1)) / (labels - 1));
      s("text", { x: mg.l + (i + 0.5) * cw, y: mg.t + ph + 16, "text-anchor": k === 0 && labels > 1 ? "start" : k === labels - 1 ? "end" : "middle" }, svg).textContent = timeLabel(profiles[i]);
    }
    const hit = s("rect", { x: mg.l, y: mg.t, width: pw, height: ph, fill: "transparent" }, svg);
    const tt = tooltip(card);
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const sx = ((ev.clientX - r.left) / r.width) * f.width, sy = ((ev.clientY - r.top) / r.height) * f.height;
      const i = Math.max(0, Math.min(n - 1, Math.floor((sx - mg.l) / cw)));
      const b = Math.max(lo, Math.min(hi, hi - Math.floor((sy - mg.t) / rh)));
      const t = totals[i];
      const sum = t ? t.reduce((a, c) => a + c, 0) : 0;
      const med = t ? quantileBucket(t, 0.5) : null;
      fillTooltip(tt, `${timeLabel(profiles[i])} · ${bucketRange(b)}`, [
        { color: "var(--series-1)", label: "of this snapshot's events", value: t && sum ? fmtPct((100 * t[b]) / sum) : "–" },
        { color: "transparent", label: "events in bucket", value: t ? t[b].toLocaleString() : "–" },
        { color: "var(--text-primary)", label: "median", value: med == null ? "–" : bucketRange(med) },
      ]);
      const cr = card.getBoundingClientRect();
      placeTooltip(tt, card, ev.clientX - cr.left, ev.clientY - cr.top);
    });
    hit.addEventListener("pointerleave", () => { tt.style.display = "none"; });
  }
}

function sparkline(parent, values) {
  const w = 140, hgt = 26, pts = values.slice(-24);
  const svg = s("svg", { viewBox: `0 0 ${w} ${hgt}`, width: w, height: hgt, "aria-hidden": "true" }, parent);
  const nums = pts.filter((v) => v != null);
  if (nums.length < 2) return;
  const lo = Math.min(...nums), hi = Math.max(...nums), span = hi - lo || 1;
  let d = "", pen = false;
  pts.forEach((v, i) => {
    if (v == null) { pen = false; return; }
    d += `${pen ? "L" : "M"}${((i * (w - 6)) / (pts.length - 1) + 1).toFixed(1)},${(hgt - 4 - ((v - lo) / span) * (hgt - 8)).toFixed(1)}`;
    pen = true;
  });
  s("path", { d, fill: "none", "stroke-width": 1.5, "stroke-linejoin": "round" }, svg, { stroke: "var(--text-muted)" });
  const lv = pts[pts.length - 1];
  if (lv != null) s("circle", { cx: w - 5, cy: hgt - 4 - ((lv - lo) / span) * (hgt - 8), r: 3 }, svg, { fill: "var(--series-1)" });
}

// ---------- tiles & tables ----------
function renderKpis() {
  const el = document.getElementById("kpis");
  el.replaceChildren();
  const p = profiles[profiles.length - 1];
  if (!p) { h("div", { class: "empty" }, el, "Waiting for the first profile from a GPU node"); return; }
  const tile = (label, value, note, key) => {
    const t = h("div", { class: "tile" }, el);
    h("div", { class: "label" }, t, label);
    h("div", { class: "value" }, t, value);
    if (note) h("div", { class: "note" }, t, note);
    if (key) sparkline(t, profiles.map((q) => get(q, key)));
  };
  const v = VERDICTS[p.verdict] || [p.verdict || "–", ""];
  tile("Bottleneck verdict", v[0], v[1], null);
  tile("GPU busy (kernel time)", fmtPct(get(p, "attribution.gpu_busy_pct")), "what device metrics call utilization", "attribution.gpu_busy_pct");
  tile("SM utilization", fmtPct(get(p, "counters.sm_util_pct.avg")), `p99 ${fmtPct(get(p, "counters.sm_util_pct.p99"))}`, "counters.sm_util_pct.avg");
  tile("Tensor Active", fmtPct(get(p, "counters.tensor_active_pct.avg")), `p99 ${fmtPct(get(p, "counters.tensor_active_pct.p99"))}`, "counters.tensor_active_pct.avg");
  tile("Launch → start p95", fmtMs(get(p, "launch_path.launch_to_start_ms_p95")), `p50 ${fmtMs(get(p, "launch_path.launch_to_start_ms_p50"))}`, "launch_path.launch_to_start_ms_p95");
  if (get(p, "attribution.by_phase_ms")) {
    tile("Decode share of GPU time", fmtPct(get(p, "derived.decode_pct")),
      `mixed ${fmtPct(get(p, "derived.mixed_pct"))} · prefill-only ${fmtPct(get(p, "derived.prefill_pct"))}`, "derived.decode_pct");
    tile("Tokens per second", `${fmtNum(get(p, "derived.prefill_tps"), 0)} / ${fmtNum(get(p, "derived.decode_tps"), 0)}`,
      "prefill / decode (from NVTX step ranges)", "derived.decode_tps");
  }
  if (get(p, "attribution.by_kernel_family_ms.nccl") > 0) {
    tile("NCCL share of GPU time", fmtPct(get(p, "derived.nccl_pct")),
      `decode steps ${fmtPct(get(p, "derived.nccl_in_decode_pct"))} · mixed ${fmtPct(get(p, "derived.nccl_in_mixed_pct"))} · incl. waiting for peers`,
      "derived.nccl_pct");
  }
  tile("Dashboard lag", fmtS(get(p, "fleet.lag_s")), "target 5–10 s", "fleet.lag_s");
}

function table(container, head, rows) {
  container.replaceChildren();
  if (!rows.length) { h("div", { class: "empty" }, container, "No data yet"); return; }
  const t = h("table", {}, container);
  const tr = h("tr", {}, h("thead", {}, t));
  head.forEach(([label, num]) => h("th", num ? { class: "num" } : {}, tr, label));
  const tb = h("tbody", {}, t);
  for (const r of rows) {
    const row = h("tr", {}, tb);
    r.forEach((cell, i) => {
      const td = h("td", head[i][1] ? { class: "num" } : {}, row);
      if (cell && typeof cell === "object") {
        const sw = h("span", { class: "sw" }, td);
        sw.style.background = cell.color;
        td.appendChild(document.createTextNode(cell.text));
      } else td.textContent = cell;
    });
  }
}

function renderTables() {
  const p = profiles[profiles.length - 1];
  const fam = p ? FAMILIES.map(([k, label], i) => ({ label, color: seriesColor(i), v: get(p, `attribution.by_kernel_family_ms.${k}`) || 0 })) : [];
  const tot = fam.reduce((a, r) => a + r.v, 0);
  table(document.getElementById("tb-fam"), [["Family"], ["GPU time", 1], ["Share", 1]],
    fam.filter((r) => r.v > 0).map((r) => [{ color: r.color, text: r.label }, fmtMs(r.v), tot ? `${Math.round((100 * r.v) / tot)}%` : "–"]));
  table(document.getElementById("tb-top"), [["Kernel"], ["Launches", 1], ["GPU time", 1]],
    (p ? get(p, "kernels.top") || [] : []).map((k) => [k.name, (k.launches || 0).toLocaleString(), fmtMs(k.time_ms)]));
  table(document.getElementById("tb-recent"),
    [["Snapshot end"], ["Verdict"], ["GPU busy", 1], ["SM util", 1], ["Tensor", 1], ["Decode %", 1], ["Launches/s", 1], ["Delay p95", 1], ["Lag", 1]],
    profiles.slice(-12).reverse().map((q) => [timeLabel(q), (VERDICTS[q.verdict] || [q.verdict])[0],
      fmtPct(get(q, "attribution.gpu_busy_pct")), fmtPct(get(q, "counters.sm_util_pct.avg")), fmtPct(get(q, "counters.tensor_active_pct.avg")),
      fmtPct(get(q, "derived.decode_pct")), fmtNum(get(q, "kernels.launches_per_s"), 0), fmtMs(get(q, "launch_path.launch_to_start_ms_p95")), fmtS(get(q, "fleet.lag_s"))]));
}

function render() {
  renderQueued = false;
  const p = profiles[profiles.length - 1];
  if (p) {
    const procs = (p.processes || []).length ? ` · processes ${p.processes.join(", ")}` : "";
    const role = p.role ? ` · ${p.role}` : "";
    document.getElementById("ident").textContent = `${p.node || "node"}${role} · GPU ${get(p, "gpu.index") ?? 0} ${get(p, "gpu.chip") || ""}${procs}`;
    document.getElementById("updated").textContent = `last snapshot ${timeLabel(p)} · ${profiles.length} shown`;
  }
  renderKpis();
  lineChart(document.getElementById("ch-util"), document.getElementById("lg-util"), {
    yMax: 100,
    fmt: { value: fmtPct, axis: (v) => `${Math.round(v)}%` },
    series: [
      { label: "GPU busy", color: seriesColor(0), get: (q) => get(q, "attribution.gpu_busy_pct") },
      { label: "Tensor Active", color: seriesColor(1), get: (q) => get(q, "counters.tensor_active_pct.avg") },
      { label: "SM utilization", color: seriesColor(2), get: (q) => get(q, "counters.sm_util_pct.avg") },
    ],
  });
  stackedChart(document.getElementById("ch-fam"), document.getElementById("lg-fam"), {
    field: "attribution.by_kernel_family_ms",
    series: FAMILIES.map(([k, label], i) => [k, label, seriesColor(i)]),
  });
  const anyNccl = profiles.some((q) => get(q, "derived.phase_split.nccl") > 0);
  stackedChart(document.getElementById("ch-phase"), document.getElementById("lg-phase"), {
    field: "derived.phase_split",
    series: anyNccl ? PHASES_WITH_NCCL : PHASES,
  });
  lineChart(document.getElementById("ch-delay"), document.getElementById("lg-delay"), {
    fmt: { value: fmtMs, axis: null },
    series: [
      { label: "p50", color: seriesColor(0), get: (q) => get(q, "launch_path.launch_to_start_ms_p50") },
      { label: "p95", color: seriesColor(1), get: (q) => get(q, "launch_path.launch_to_start_ms_p95") },
    ],
  });
  renderHist();
  lineChart(document.getElementById("ch-lag"), null, {
    fmt: { value: fmtS, axis: null },
    ref: { value: 10, label: "10 s target" },
    series: [{ label: "Lag", color: seriesColor(0), get: (q) => get(q, "fleet.lag_s") }],
  });
  renderTables();
}
function scheduleRender() {
  if (renderQueued) return;
  renderQueued = true;
  // requestAnimationFrame pauses in background tabs; keep monitoring views current there too.
  if (document.hidden) setTimeout(render, 100); else requestAnimationFrame(render);
}
document.addEventListener("visibilitychange", () => { if (!document.hidden) scheduleRender(); });

// ---------- data ----------
function setConn(state, text) {
  const c = document.getElementById("conn");
  c.dataset.state = state;
  document.getElementById("connText").textContent = text;
}
// Derived per-profile values used by tiles, sparklines and tables.
function derive(p) {
  const ph = get(p, "attribution.by_phase_ms");
  const tok = p.phase_tokens;
  const secs = get(p, "window.duration_s") || 5;
  const d = {};
  if (ph) {
    const tot = PHASES.reduce((a, [k]) => a + (ph[k] || 0), 0);
    for (const [k] of PHASES) d[`${k}_pct`] = tot ? (100 * (ph[k] || 0)) / tot : null;
  }
  if (tok) { d.prefill_tps = (tok.prefill || 0) / secs; d.decode_tps = (tok.decode || 0) / secs; }
  // NCCL (tensor-parallel all-reduce): share of kernel time, share within decode / mixed steps, and the phase
  // chart's split with NCCL carved out of each phase.
  const nccl = get(p, "attribution.by_kernel_family_ms.nccl");
  const ktime = get(p, "attribution.kernel_time_ms");
  if (nccl != null && ktime) d.nccl_pct = (100 * nccl) / ktime;
  const pn = get(p, "attribution.by_phase_nccl_ms");
  if (ph) {
    d.phase_split = {};
    for (const [k] of PHASES) d.phase_split[k] = Math.max(0, (ph[k] || 0) - ((pn && pn[k]) || 0));
    d.phase_split.nccl = pn ? PHASES.reduce((a, [k]) => a + (pn[k] || 0), 0) : 0;
    if (pn) for (const k of ["decode", "mixed"]) d[`nccl_in_${k}_pct`] = ph[k] ? (100 * (pn[k] || 0)) / ph[k] : null;
  }
  p.derived = d;
  return p;
}

function refreshNodes() {
  const order = [];
  for (let i = allProfiles.length - 1; i >= 0; i--) {
    const n = allProfiles[i].node || "node";
    if (!order.includes(n)) order.push(n);
  }
  if (!selectedNode || !order.includes(selectedNode)) selectedNode = order[0] || null;
  const sel = document.getElementById("nodeSel");
  const current = [...sel.options].map((o) => o.value).join("\n");
  if (current !== order.join("\n")) {
    sel.replaceChildren();
    for (const n of order) {
      const o = document.createElement("option");
      o.value = n;
      o.textContent = n;
      sel.appendChild(o);
    }
  }
  if (selectedNode) sel.value = selectedNode;
  profiles = allProfiles.filter((p) => (p.node || "node") === selectedNode).slice(-MAX_POINTS);
}
function add(p) {
  allProfiles.push(derive(p));
  if (allProfiles.length > MAX_POINTS * 4) allProfiles.splice(0, allProfiles.length - MAX_POINTS * 4);
  refreshNodes();
  scheduleRender();
}

// ---------- replay / frame mode (for rendering time-lapse frames) ----------
// ?replay=<recording.json>&frame=<N>[&window=<snapshots>] draws the window ending at snapshot N of a recording
// (tools/timelapse/export_recording.py) instead of the live feed, with a caption of the active load regime.
// Sets <html data-ready="1"> when drawn, for headless screenshots.
async function startReplay(params) {
  document.documentElement.classList.add("frame");
  const rec = await (await fetch(params.get("replay"))).json();
  const n = Math.max(0, Math.min(rec.profiles.length - 1, parseInt(params.get("frame") || "0", 10)));
  const win = parseInt(params.get("window") || "120", 10);
  allProfiles = rec.profiles.slice(Math.max(0, n - win + 1), n + 1).map(derive);
  refreshNodes();
  const p = rec.profiles[n];
  const tEnd = p.window.end_utc_ns;
  const regime = [...(rec.regimes || [])].reverse().find((r) => r.t_utc_ns <= tEnd);
  const elapsed = Math.max(0, (tEnd - rec.t0_utc_ns) / 1e9);
  const total = rec.duration_s || 1800;
  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const cap = document.getElementById("caption");
  cap.replaceChildren();
  h("b", {}, cap, regime ? `Load regime: ${regime.name}` : "Load regime: warm-up");
  if (regime && regime.desc) h("span", {}, cap, regime.desc);
  h("span", { class: "clock" }, cap, `${mmss(elapsed)} / ${mmss(total)}`);
  setConn("live", "replay of a recorded run");
  render();
  document.documentElement.dataset.ready = "1";
}

async function start() {
  const params = new URLSearchParams(location.search);
  if (params.get("replay")) return startReplay(params);
  try {
    const r = await fetch(`/v1/profiles?limit=${MAX_POINTS * 4}`);
    allProfiles = (await r.json()).map(derive);
  } catch (e) { allProfiles = []; }
  refreshNodes();
  document.getElementById("nodeSel").addEventListener("change", (ev) => {
    selectedNode = ev.target.value;
    refreshNodes();
    scheduleRender();
  });
  scheduleRender();
  const es = new EventSource("/events");
  es.onopen = () => setConn("live", "live");
  es.onerror = () => setConn("down", "reconnecting");
  es.addEventListener("profile", (ev) => { add(JSON.parse(ev.data)); setConn("live", "live"); });
  setInterval(() => {
    const p = profiles[profiles.length - 1];
    if (!p) return;
    const age = (Date.now() - p.fleet.received_utc_ns / 1e6) / 1000;
    if (age > 20 && document.getElementById("conn").dataset.state === "live") setConn("waiting", `no data for ${Math.round(age)} s`);
  }, 2000);
  window.addEventListener("resize", scheduleRender);
}
start();
