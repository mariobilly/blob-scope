"use strict";
// Blob Scope window: preview, presets, controls, playback and export. The server does all rendering.

const $ = (id) => document.getElementById(id);
const S = {
  defaults: null, presets: [], P: {}, preset: "tracker", media: null,
  t: 0, playing: false, playStart: 0, playFrom: 0, showOrig: false,
  seq: 0, shown: 0, inflight: false, again: false, thumbSeq: 0,
};

const PALETTE_NAMES = [["source", "Own colours"], ["cobalt", "Cobalt"], ["ink", "Ink"], ["ember", "Ember"],
  ["acid", "Acid"], ["riso", "Riso"], ["uv", "Ultraviolet"], ["night", "Night"]];

// k: param key. show: when the control is visible. fmt: value label.
const SECTIONS = [
  { title: "Dots", items: [
    { k: "dots", type: "toggle", label: "Dots" },
    { k: "palette", type: "select", label: "Colours", options: PALETTE_NAMES },
    { k: "cols", type: "range", label: "Columns", min: 24, max: 320, step: 1, fmt: (v) => v },
    { k: "dot", type: "range", label: "Dot size", min: 0.15, max: 1, step: 0.01 },
    { k: "shape", type: "seg", label: "Shape", options: [["square", "■ square"], ["round", "● round"], ["plus", "✚ plus"]] },
    { k: "pattern", type: "seg", label: "Pattern", options: [["bayer2", "2×2"], ["bayer4", "4×4"], ["bayer8", "8×8"]] },
    { k: "spread", type: "range", label: "Dither", min: 0, max: 1.5, step: 0.01 },
    { k: "levels", type: "range", label: "Levels", min: 2, max: 6, step: 1, fmt: (v) => v, show: (P) => P.palette === "source" },
    { k: "warm", type: "toggle", label: "Keep fire warm", show: (P) => P.palette !== "source" },
    { k: "grid", type: "range", label: "Panel gaps", min: 0, max: 1, step: 0.01 },
  ]},
  { title: "Tone", items: [
    { k: "contrast", type: "range", label: "Contrast", min: 0.2, max: 3, step: 0.01 },
    { k: "brightness", type: "range", label: "Brightness", min: -0.6, max: 0.6, step: 0.01 },
    { k: "saturation", type: "range", label: "Saturation", min: 0, max: 3, step: 0.01 },
    { k: "hue", type: "range", label: "Hue shift", min: -180, max: 180, step: 1, fmt: (v) => v + "°" },
  ]},
  { title: "Tracker", items: [
    { k: "hud", type: "toggle", label: "Tracker HUD" },
    { k: "tags", type: "range", label: "Tags", min: 0, max: 20, step: 1, fmt: (v) => v },
    { k: "life", type: "range", label: "Tag life", min: 0.2, max: 4, step: 0.05, fmt: (v) => (+v).toFixed(2) + "s" },
    { k: "tag_scale", type: "range", label: "Tag size", min: 0.5, max: 2.5, step: 0.05 },
    { k: "links", type: "toggle", label: "Links" },
    { k: "pair_labels", type: "toggle", label: "Pair labels" },
    { k: "main_tag", type: "toggle", label: "Main tag" },
    { k: "brackets", type: "toggle", label: "Brackets" },
    { k: "tag_color", type: "color", label: "Tag colour" },
    { k: "link_color", type: "color", label: "Link colour" },
    { k: "id_base", type: "number", label: "ID start", min: 0, max: 99000 },
  ]},
  { title: "Frame", items: [
    { k: "frame", type: "select", label: "Shape", options: [["source", "Original"], ["reel", "Letterbox (3:4 on 9:16)"],
      ["9:16", "9:16"], ["4:5", "4:5"], ["1:1", "1:1"], ["16:9", "16:9"]] },
    { k: "paper", type: "toggle", label: "Cut out to paper" },
    { k: "paper_color", type: "color", label: "Paper", show: (P) => P.paper },
    { k: "ornament", type: "select", label: "Corner dots", options: [["off", "Off"], ["top-right", "Top right"], ["corners", "All corners"]] },
  ]},
  { title: "Motion", items: [
    { k: "motion", type: "select", label: "Camera", options: [["off", "Off"], ["drift", "Drift"], ["sway", "Sway"],
      ["shake", "Shake"], ["pulse", "Pulse"]] },
    { k: "motion_amt", type: "range", label: "Strength", min: 0, max: 3, step: 0.05 },
    { k: "cycle", type: "range", label: "Loop / cycle", min: 2, max: 20, step: 0.5, fmt: (v) => (+v).toFixed(1) + "s" },
    { k: "glitch", type: "range", label: "Glitch", min: 0, max: 1, step: 0.01 },
    { k: "seed", type: "seed", label: "Seed" },
  ]},
];
const CTRL = {};  // key -> {row, set(v)}

// --------------------------------------------------------------------------- helpers

async function api(path, body, raw) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* not JSON */ }
    throw new Error(msg);
  }
  return raw ? r : r.json();
}

let toastTimer = 0;
function toast(html, err, ms) {
  const el = $("toast");
  el.innerHTML = html;
  el.classList.toggle("err", !!err);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), ms || 4500);
  return el;
}

const isVideo = () => S.media && S.media.kind === "video";
const period = () => (isVideo() ? S.media.duration : +S.P.cycle);
function frameAt(t) {
  if (!isVideo()) return 0;
  return Math.min(S.media.frames - 1, Math.max(0, Math.floor(t * S.media.fps + 1e-6)));
}
function tOf() {  // the time the preview shows: a video frame's own time, or the position in the photo loop
  return isVideo() ? frameAt(S.t) / S.media.fps : S.t;
}

// --------------------------------------------------------------------------- controls

function buildControls() {
  const host = $("controls");
  host.innerHTML = "";
  for (const sec of SECTIONS) {
    const box = document.createElement("div");
    box.className = "sec";
    const h = document.createElement("h3");
    h.textContent = sec.title;
    h.onclick = () => box.classList.toggle("closed");
    const rows = document.createElement("div");
    rows.className = "rows";
    box.append(h, rows);
    for (const it of sec.items) rows.append(makeRow(it));
    host.append(box);
  }
}

function makeRow(it) {
  const row = document.createElement("div");
  row.className = "row";
  const lab = document.createElement("label");
  lab.textContent = it.label;
  row.append(lab);
  let set;
  if (it.type === "range") {
    const inp = document.createElement("input");
    Object.assign(inp, { type: "range", min: it.min, max: it.max, step: it.step });
    const val = document.createElement("span");
    val.className = "val";
    const fmt = it.fmt || ((v) => (+v).toFixed(2));
    inp.oninput = () => { val.textContent = fmt(inp.value); change(it.k, +inp.value); };
    inp.ondblclick = () => { set(S.defaults[it.k]); change(it.k, S.defaults[it.k]); };
    row.append(inp, val);
    set = (v) => { inp.value = v; val.textContent = fmt(v); };
  } else if (it.type === "toggle") {
    const b = document.createElement("button");
    b.className = "toggle";
    b.setAttribute("role", "switch");
    b.onclick = () => { const v = !S.P[it.k]; set(v); change(it.k, v); };
    row.append(b);
    set = (v) => { b.classList.toggle("on", !!v); b.setAttribute("aria-checked", !!v); };
  } else if (it.type === "select") {
    const sel = document.createElement("select");
    for (const [v, n] of it.options) sel.append(new Option(n, v));
    sel.onchange = () => change(it.k, sel.value);
    row.append(sel);
    set = (v) => { sel.value = v; };
  } else if (it.type === "seg") {
    const seg = document.createElement("div");
    seg.className = "seg";
    const btns = it.options.map(([v, n]) => {
      const b = document.createElement("button");
      b.textContent = n;
      b.onclick = () => { set(v); change(it.k, v); };
      seg.append(b);
      return [v, b];
    });
    row.append(seg);
    set = (v) => btns.forEach(([bv, b]) => b.classList.toggle("on", bv === v));
  } else if (it.type === "color") {
    const wrap = document.createElement("div");
    wrap.className = "colorpick";
    const c = document.createElement("input");
    c.type = "color";
    const hex = document.createElement("span");
    hex.className = "mono";
    c.oninput = () => { hex.textContent = c.value; change(it.k, c.value); };
    wrap.append(c, hex);
    row.append(wrap);
    set = (v) => { c.value = v; hex.textContent = v; };
  } else if (it.type === "number") {
    const n = document.createElement("input");
    Object.assign(n, { type: "number", min: it.min, max: it.max, step: 1 });
    n.onchange = () => change(it.k, Math.round(+n.value || 0));
    row.append(n);
    set = (v) => { n.value = v; };
  } else if (it.type === "seed") {
    const wrap = document.createElement("div");
    wrap.className = "seedrow";
    const n = document.createElement("input");
    Object.assign(n, { type: "number", min: 0, max: 1000000, step: 1 });
    n.onchange = () => change("seed", Math.round(+n.value || 0));
    const dice = document.createElement("button");
    dice.textContent = "⚄ New";
    dice.title = "New tag positions, IDs and glitch timing";
    dice.onclick = () => { const v = Math.floor(Math.random() * 99999); set(v); change("seed", v); };
    wrap.append(n, dice);
    row.append(wrap);
    set = (v) => { n.value = v; };
  }
  CTRL[it.k] = { row, set, it };
  return row;
}

function syncControls() {
  for (const k in CTRL) {
    const c = CTRL[k];
    c.set(S.P[k]);
    c.row.hidden = c.it.show ? !c.it.show(S.P) : false;
  }
  document.querySelectorAll(".preset").forEach((el) => el.classList.toggle("on", el.dataset.id === S.preset));
}

let matteNoted = false;
function noteMatte() {  // the cut-out models load on first use
  if (matteNoted) return;
  matteNoted = true;
  toast("Cutting out the subject… the first time loads the models (about 15 s, longer while they download).", false, 15000);
}

function change(k, v) {
  S.P[k] = v;
  if (k === "paper" && v) noteMatte();
  if (k !== "seed" && k !== "frame") S.preset = null;
  syncControls();
  if (k === "cycle") updateTimeLabel();
  if (k === "frame" || k === "paper" || k === "seed") loadThumbs();
  preview();
}

// --------------------------------------------------------------------------- presets

function applyPreset(id) {
  const pr = S.presets.find((x) => x.id === id);
  if (!pr) return;
  const keep = { frame: S.P.frame, seed: S.P.seed, motion: S.P.motion, cycle: S.P.cycle };
  S.P = Object.assign({}, S.defaults, keep, pr.p);
  if (pr.p.motion === undefined && isVideo()) S.P.motion = keep.motion;
  S.preset = id;
  if (S.P.paper && S.media) noteMatte();
  syncControls();
  preview();
}

function buildPresets() {
  const host = $("presets");
  host.innerHTML = "";
  for (const pr of S.presets) {
    const b = document.createElement("button");
    b.className = "preset";
    b.dataset.id = pr.id;
    b.title = pr.note;
    b.innerHTML = `<img alt=""><span class="pn"></span>`;
    b.querySelector(".pn").textContent = pr.name;
    b.onclick = () => applyPreset(pr.id);
    host.append(b);
  }
}

let thumbTimer = 0;
function loadThumbs() {
  clearTimeout(thumbTimer);
  thumbTimer = setTimeout(async () => {
    if (!S.media) return;
    const seq = ++S.thumbSeq;
    const keep = { frame: S.P.frame === "reel" ? "1:1" : S.P.frame, seed: S.P.seed, motion: "off",
      cycle: S.P.cycle };
    for (const el of document.querySelectorAll(".preset")) {
      if (seq !== S.thumbSeq) return;
      try {
        const r = await api("/api/thumb", { preset: el.dataset.id, frame: frameAt(S.t), t: 0.9, side: 300,
          keep }, true);
        const url = URL.createObjectURL(await r.blob());
        const img = el.querySelector("img");
        if (img.src) URL.revokeObjectURL(img.src);
        img.src = url;
      } catch (e) { /* a thumbnail is optional */ }
    }
  }, 500);
}

// --------------------------------------------------------------------------- preview

function previewSide() {
  const v = $("view");
  const big = Math.max(v.clientWidth, v.clientHeight) * Math.min(2, window.devicePixelRatio || 1);
  return Math.round(Math.min(S.playing ? 900 : 1600, Math.max(480, big)));
}

async function preview() {
  if (!S.media) return;
  if (S.inflight) { S.again = true; return; }
  S.inflight = true;
  S.again = false;
  const seq = ++S.seq;
  $("busy").hidden = S.playing;
  try {
    const body = { frame: frameAt(S.t), t: tOf(), side: previewSide(), params: S.P };
    const r = await api("/api/preview", body, true);
    const blob = await r.blob();
    if (seq > S.shown) {
      S.shown = seq;
      const img = $("shot");
      const old = img.src;
      img.src = URL.createObjectURL(blob);
      if (old) setTimeout(() => URL.revokeObjectURL(old), 1000);
      $("ms").textContent = r.headers.get("X-Render-Ms") + " ms";
    }
  } catch (e) {
    toast("Preview failed: " + e.message, true);
    stopPlay();
  } finally {
    S.inflight = false;
    $("busy").hidden = true;
    if (S.playing) requestAnimationFrame(tick);
    else if (S.again) preview();
  }
}

async function showOriginal(on) {
  if (!S.media) return;
  S.showOrig = on;
  $("view").classList.toggle("showorig", on);
  $("badge").hidden = !on;
  if (on) {
    try {
      const r = await api("/api/preview", { frame: frameAt(S.t), side: previewSide(), original: true }, true);
      const img = $("orig");
      if (img.src) URL.revokeObjectURL(img.src);
      img.src = URL.createObjectURL(await r.blob());
    } catch (e) { /* ignore */ }
  }
}

// --------------------------------------------------------------------------- playback

function updateTimeLabel() {
  const per = period() || 1;
  $("tlabel").textContent = `${tOf().toFixed(2)} / ${per.toFixed(isVideo() ? 2 : 1)} s`;
  $("tpos").value = Math.round((S.t / per) * 1000) % 1001;
}

function tick() {
  if (!S.playing) return;
  const per = period() || 1;
  S.t = (S.playFrom + (performance.now() - S.playStart) / 1000) % per;
  updateTimeLabel();
  preview();
}

function startPlay() {
  if (!S.media) return;
  S.playing = true;
  S.playStart = performance.now();
  S.playFrom = S.t;
  $("play").classList.add("on");
  $("play").innerHTML = "&#10074;&#10074;";
  tick();
}

function stopPlay() {
  if (!S.playing) return;
  S.playing = false;
  $("play").classList.remove("on");
  $("play").innerHTML = "&#9654;";
  preview();  // a full-size frame where playback stopped
  loadThumbs();
}

// --------------------------------------------------------------------------- media

function setMedia(m) {
  stopPlay();
  S.media = m;
  S.t = 0;
  $("empty").hidden = !!m;
  $("export").disabled = !m;
  $("playbar").hidden = !m;
  const photo = m && m.kind === "image";
  $("photoOutWrap").hidden = !photo;
  $("fpsWrap").hidden = !photo || $("photoOut").value === "still";
  if (!m) { $("fileinfo").textContent = "No file open"; return; }
  const info = m.kind === "video"
    ? `${m.width}×${m.height} · ${m.duration.toFixed(1)} s · ${(+m.fps).toFixed(2)} fps${m.has_audio ? " · audio" : ""}`
    : `${m.width}×${m.height} photo`;
  $("fileinfo").innerHTML = `<b></b> · ${info}`;
  $("fileinfo").querySelector("b").textContent = m.name;
  // photos drift like the reel; a video already moves, so its camera starts still
  if (m.kind === "video" && S.preset) S.P.motion = "off";
  else if (m.kind === "image" && S.P.motion === "off" && S.preset) S.P.motion = "drift";
  syncControls();
  updateTimeLabel();
  preview();
  loadThumbs();
}

async function browse() {
  try {
    const r = await api("/api/browse", {});
    if (!r.cancelled) setMedia(r.media);
  } catch (e) { toast(e.message, true); }
}

async function uploadFile(file) {
  toast("Opening " + file.name.replace(/</g, "&lt;") + "…", false, 60000);
  try {
    const r = await fetch("/api/upload?name=" + encodeURIComponent(file.name), { method: "POST", body: file });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    setMedia((await r.json()).media);
    $("toast").hidden = true;
  } catch (e) { toast(e.message, true); }
}

// --------------------------------------------------------------------------- export

let pollTimer = 0;
async function doExport() {
  if (!S.media || !$("progress").hidden) return;
  stopPlay();
  try {
    const job = await api("/api/export", { params: S.P, max_side: +$("exportSize").value,
      photo_out: $("photoOut").value, fps: +$("fps").value });
    $("progress").hidden = false;
    $("ptitle").textContent = job.kind === "video" ? "Exporting the clip…"
      : job.kind === "loop" ? "Exporting the loop…" : "Exporting the still…";
    poll();
  } catch (e) { toast(e.message, true); }
}

async function poll() {
  clearTimeout(pollTimer);
  let j;
  try { j = await api("/api/job"); } catch (e) { pollTimer = setTimeout(poll, 800); return; }
  const pct = j.total ? (100 * j.done) / j.total : 0;
  $("pbar").style.width = pct.toFixed(1) + "%";
  $("pinfo").textContent = j.state === "running"
    ? `${j.done} / ${j.total} frames` + (j.fps ? ` · ${j.fps.toFixed(1)} fps · ${Math.ceil(j.eta)} s left` : "")
    : "";
  if (j.state === "running") { pollTimer = setTimeout(poll, 400); return; }
  $("progress").hidden = true;
  if (j.state === "done") {
    const probs = j.report && j.report.problems && j.report.problems.length ? ` (check: ${j.report.problems.join("; ")})` : "";
    const el = toast(`Saved <b></b>${probs} <button id="reveal">Show in folder</button>`, !!probs, 12000);
    el.querySelector("b").textContent = j.out.split(/[\\/]/).pop();
    $("reveal").onclick = () => api("/api/reveal", { path: j.out });
  } else if (j.state === "error") {
    toast("Export failed: " + (j.error || "unknown error").replace(/</g, "&lt;"), true, 12000);
  } else if (j.state === "cancelled") {
    toast("Export cancelled");
  }
}

// --------------------------------------------------------------------------- randomize

function randomize() {
  const pick = (a) => a[Math.floor(Math.random() * a.length)];
  const pr = pick(S.presets.filter((x) => x.id !== "clean" && (x.id !== "paper" || Math.random() < 0.3)));
  applyPreset(pr.id);
  const r = (a, b) => a + Math.random() * (b - a);
  Object.assign(S.P, {
    cols: Math.round(r(70, 240)), dot: +r(0.4, 0.85).toFixed(2), tags: Math.round(r(4, 12)),
    life: +r(0.6, 2).toFixed(2), glitch: +r(0, 0.6).toFixed(2), seed: Math.floor(Math.random() * 99999),
    shape: pick(["square", "square", "round", "plus"]), pattern: pick(["bayer4", "bayer4", "bayer8", "bayer2"]),
  });
  if (!isVideo()) S.P.motion = pick(["drift", "drift", "sway", "shake", "pulse"]);
  S.preset = null;
  syncControls();
  preview();
}

// --------------------------------------------------------------------------- wiring

function wire() {
  $("open").onclick = browse;
  $("open2").onclick = browse;
  $("random").onclick = randomize;
  $("export").onclick = doExport;
  $("pcancel").onclick = () => api("/api/cancel", {});
  $("photoOut").onchange = () => { $("fpsWrap").hidden = $("photoOut").value === "still"; };
  $("play").onclick = () => (S.playing ? stopPlay() : startPlay());
  $("tpos").oninput = () => {
    stopPlay();
    S.t = (+$("tpos").value / 1000) * (period() || 1);
    if (S.t >= period()) S.t = 0;
    updateTimeLabel();
    preview();
  };
  $("tpos").onchange = () => loadThumbs();

  const view = $("view");
  view.addEventListener("dragover", (e) => { e.preventDefault(); view.classList.add("drag"); });
  view.addEventListener("dragleave", () => view.classList.remove("drag"));
  view.addEventListener("drop", (e) => {
    e.preventDefault();
    view.classList.remove("drag");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => e.preventDefault());

  document.addEventListener("keydown", (e) => {
    const typing = /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) && document.activeElement.type !== "range";
    if (e.ctrlKey && e.key.toLowerCase() === "o") { e.preventDefault(); browse(); return; }
    if (e.ctrlKey && e.key.toLowerCase() === "e") { e.preventDefault(); doExport(); return; }
    if (typing || e.ctrlKey || e.altKey) return;
    if (e.code === "Space") { e.preventDefault(); if (!e.repeat) showOriginal(true); }
    else if (e.key.toLowerCase() === "p") { S.playing ? stopPlay() : startPlay(); }
    else if (e.key.toLowerCase() === "r") randomize();
  });
  document.addEventListener("keyup", (e) => { if (e.code === "Space") showOriginal(false); });
  let rz = 0;
  window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(preview, 200); });
  setInterval(() => api("/api/ping", {}).catch(() => {}), 5000);
  api("/api/ping", {}).catch(() => {});
}

async function init() {
  wire();
  const st = await api("/api/state");
  S.defaults = st.defaults;
  S.presets = st.presets;
  S.P = Object.assign({}, st.defaults);
  buildControls();
  buildPresets();
  syncControls();
  if (st.media) setMedia(st.media);
  if (st.job && st.job.state === "running") { $("progress").hidden = false; poll(); }
}

init().catch((e) => toast("Could not start: " + e.message, true, 30000));
