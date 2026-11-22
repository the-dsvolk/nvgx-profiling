"use strict";
// nvgx-fleet live dashboard (POC). Vanilla JS + SVG. Data: /v1/profiles (history) + /events (live).

const MAX_POINTS = 180; // 15 minutes of 5 s snapshots
const FAMILIES = [
  ["gemm", "GEMM (CUTLASS / nvjet)"], ["flash_attention", "FlashAttention"], ["pytorch_native", "PyTorch native"],
  ["flashinfer_sampling", "FlashInfer sampling"], ["vllm_custom", "vLLM custom"], ["other", "Other"],
  ["unresolved", "Unresolved names"],
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

function stackedChart(container, legendEl, { height = 230 }) {
  const card = container.closest(".card");
  const n = profiles.length;
  legend(legendEl, FAMILIES.map(([, label], i) => ({ label, color: seriesColor(i) })), "rect");
  if (!n) { container.replaceChildren(h("div", { class: "empty" }, null, "No data yet")); return; }
  const f = frame(container, height, { l: 48, r: 12, t: 10, b: 24 });
  const { svg, m, pw, ph } = f;
  const data = profiles.map((p) => FAMILIES.map(([k]) => get(p, `attribution.by_kernel_family_ms.${k}`) || 0));
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
        s("path", { d: `M${x},${base}V${yTop + r}Q${x},${yTop} ${x + r},${yTop}H${x + colW - r}Q${x + colW},${yTop} ${x + colW},${yTop + r}V${base}Z` }, g, { fill: seriesColor(k) });
      } else {
        s("rect", { x, y: yTop, width: colW, height: hgt }, g, { fill: seriesColor(k) });
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
    const rows = FAMILIES.map(([, label], k) => ({ color: seriesColor(k), label, v: data[i][k] }))
      .filter((rw) => rw.v > 0).map((rw) => ({ color: rw.color, label: `${rw.label} · ${totals[i] ? Math.round((100 * rw.v) / totals[i]) : 0}%`, value: fmtMs(rw.v) }));
    fillTooltip(tt, `${timeLabel(profiles[i])} · ${fmtMs(totals[i])} total`, rows.length ? rows : [{ color: "transparent", label: "no kernels", value: "0 ms" }]);
    const cr = card.getBoundingClientRect();
    placeTooltip(tt, card, ev.clientX - cr.left, ev.clientY - cr.top);
  });
  hit.addEventListener("pointerleave", () => { if (last >= 0 && cols[last]) cols[last].style.opacity = ""; tt.style.display = "none"; });
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
    [["Snapshot end"], ["Verdict"], ["GPU busy", 1], ["SM util", 1], ["Tensor", 1], ["Launches/s", 1], ["Delay p95", 1], ["Lag", 1]],
    profiles.slice(-12).reverse().map((q) => [timeLabel(q), (VERDICTS[q.verdict] || [q.verdict])[0],
      fmtPct(get(q, "attribution.gpu_busy_pct")), fmtPct(get(q, "counters.sm_util_pct.avg")), fmtPct(get(q, "counters.tensor_active_pct.avg")),
      fmtNum(get(q, "kernels.launches_per_s"), 0), fmtMs(get(q, "launch_path.launch_to_start_ms_p95")), fmtS(get(q, "fleet.lag_s"))]));
}

function render() {
  renderQueued = false;
  const p = profiles[profiles.length - 1];
  if (p) {
    const procs = (p.processes || []).length ? ` · processes ${p.processes.join(", ")}` : "";
    document.getElementById("ident").textContent = `${p.node || "node"} · GPU ${get(p, "gpu.index") ?? 0} ${get(p, "gpu.chip") || ""}${procs}`;
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
  stackedChart(document.getElementById("ch-fam"), document.getElementById("lg-fam"), {});
  lineChart(document.getElementById("ch-delay"), document.getElementById("lg-delay"), {
    fmt: { value: fmtMs, axis: null },
    series: [
      { label: "p50", color: seriesColor(0), get: (q) => get(q, "launch_path.launch_to_start_ms_p50") },
      { label: "p95", color: seriesColor(1), get: (q) => get(q, "launch_path.launch_to_start_ms_p95") },
    ],
  });
  lineChart(document.getElementById("ch-lag"), null, {
    fmt: { value: fmtS, axis: null },
    ref: { value: 10, label: "10 s target" },
    series: [{ label: "Lag", color: seriesColor(0), get: (q) => get(q, "fleet.lag_s") }],
  });
  renderTables();
}
function scheduleRender() { if (!renderQueued) { renderQueued = true; requestAnimationFrame(render); } }

// ---------- data ----------
function setConn(state, text) {
  const c = document.getElementById("conn");
  c.dataset.state = state;
  document.getElementById("connText").textContent = text;
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
  allProfiles.push(p);
  if (allProfiles.length > MAX_POINTS * 4) allProfiles.splice(0, allProfiles.length - MAX_POINTS * 4);
  refreshNodes();
  scheduleRender();
}

async function start() {
  try {
    const r = await fetch(`/v1/profiles?limit=${MAX_POINTS * 4}`);
    allProfiles = await r.json();
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
