"use strict";
const $ = (id) => document.getElementById(id);
const MAX_MB = 200;
const RECENT_KEY = "forte.recent.v1";
const TYPE_COLORS = {
  T1: "#c0392b", T2: "#d35400", T3: "#e67e22", T4: "#8e44ad", T5: "#6c3483", T6: "#2471a3", T7: "#17a589",
  T8: "#b7950b", T9: "#a93226", T10: "#1f618d", T11: "#7d6608", T12: "#148f77", T13: "#943126", GENERIC: "#5d6d7e",
};
const state = { jobId: null, pollTimer: null, tickTimer: null, startedAt: 0, violations: [], frames: [], lbIndex: 0, xhr: null };
const STAGE_LABELS = {
  persons: "Детекция и трекинг людей (YOLO)", facts: "Описание событий (VLM)", states: "Состояние сцены по кадрам (VLM)",
  checks: "Проверка правил", layout: "Разметка рабочего места", evidence: "Кадры-доказательства", decode: "Декодирование видео",
};

/* ---------- helpers ---------- */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function mmss(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}
function secs(x, digits = 1) { return `${Number(x).toFixed(digits).replace(".", ",")} с`; }
function colorOf(typeId) { return TYPE_COLORS[typeId] || TYPE_COLORS.GENERIC; }
function plural(n, one, few, many) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}
function show(view) {
  for (const v of ["upload", "status", "result"]) $(`view-${v}`).hidden = v !== view;
  window.scrollTo(0, 0);
}
function stopTimers() {
  clearTimeout(state.pollTimer);
  clearInterval(state.tickTimer);
}

/* ---------- recent checks (localStorage) ---------- */
function recentLoad() {
  try { return JSON.parse(localStorage.getItem(RECENT_KEY)) || []; } catch (_) { return []; }
}
function recentSave(entry) {
  const all = recentLoad();
  const old = all.find((e) => e.id === entry.id) || {};
  const clean = Object.fromEntries(Object.entries(entry).filter(([, v]) => v !== undefined && v !== null));
  const list = [{ ...old, ...clean }, ...all.filter((e) => e.id !== entry.id)];
  try { localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 20))); } catch (_) { /* storage full or disabled */ }
}
function recentRender() {
  const list = recentLoad();
  const ul = $("recent");
  $("recent-clear").hidden = !list.length;
  if (!list.length) { ul.innerHTML = '<li class="muted small" style="border:0;padding-top:6px">Пока нет проверок.</li>'; return; }
  ul.innerHTML = list.map((e) => {
    let pill = '<span class="pill wait">в работе</span>';
    if (e.status === "failed") pill = '<span class="pill bad">ошибка</span>';
    else if (e.verdict === "violation") pill = `<span class="pill bad">нарушения: ${e.count ?? "?"}</span>`;
    else if (e.verdict === "no_violation") pill = '<span class="pill ok">нет нарушений</span>';
    const when = new Date(e.ts).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
    return `<li><a href="/#job=${encodeURIComponent(e.id)}"><span class="nm"><span>${esc(e.name || e.id)}</span><br><span class="when">${when}</span></span>${pill}</a></li>`;
  }).join("");
}

/* ---------- upload ---------- */
function uploadError(msg) {
  const el = $("upload-error");
  el.textContent = msg;
  el.hidden = !msg;
}
function upload(file) {
  uploadError("");
  const ext = (file.name.toLowerCase().match(/\.([a-z0-9]+)$/) || [])[1];
  if (!["mov", "mp4"].includes(ext)) { uploadError(`Файл «${file.name}» не принят: поддерживаются только видео .mov и .mp4.`); return; }
  if (file.size > MAX_MB * 1024 * 1024) { uploadError(`Файл «${file.name}» не принят: размер ${(file.size / 1048576).toFixed(0)} МБ больше допустимых ${MAX_MB} МБ.`); return; }
  if (!file.size) { uploadError(`Файл «${file.name}» пустой.`); return; }

  stopTimers();
  state.jobId = null;
  show("status");
  $("st-name").textContent = file.name;
  $("st-error").hidden = true;
  $("st-actions").hidden = true;
  setProgress("Загрузка файла на сервер…", 0, true);
  startElapsed(Date.now());

  const fd = new FormData();
  fd.append("file", file);
  const xhr = new XMLHttpRequest();
  state.xhr = xhr;
  xhr.open("POST", "/ui/analyze");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) setProgress(`Загрузка файла на сервер… ${Math.round((e.loaded / e.total) * 100)}%`, e.loaded / e.total, true);
  };
  xhr.onload = () => {
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch (_) { /* proxy error page */ }
    if (xhr.status !== 200 || !body.job_id) {
      stopTimers();
      show("upload");
      const raw = String(body.detail || `сервер ответил ${xhr.status} ${xhr.statusText}`).trim();
      uploadError(`Видео «${file.name}» не принято. ${raw.charAt(0).toUpperCase()}${raw.slice(1)}`);
      return;
    }
    recentSave({ id: body.job_id, name: file.name, ts: Date.now(), status: "processing" });
    location.hash = `job=${body.job_id}`; // hashchange -> openJob
  };
  xhr.onerror = () => { stopTimers(); show("upload"); uploadError("Сетевая ошибка при загрузке файла. Проверьте соединение и попробуйте ещё раз."); };
  xhr.send(fd);
}

/* ---------- processing ---------- */
function setProgress(stage, p, uploading = false) {
  $("st-stage").textContent = stage;
  const pct = Math.round(Math.min(1, Math.max(0, p || 0)) * 100);
  $("st-pct").textContent = uploading ? "" : `${pct}%`;
  $("st-bar").style.width = `${pct}%`;
  $("st-bar").parentElement.classList.toggle("indeterminate", !uploading && pct === 0);
}
function startElapsed(t0) {
  clearInterval(state.tickTimer);
  state.startedAt = t0;
  const tick = () => {
    const s = Math.max(0, Math.round((Date.now() - state.startedAt) / 1000));
    $("st-elapsed").textContent = s >= 60 ? `${Math.floor(s / 60)} мин ${s % 60} с` : `${s} с`;
  };
  tick();
  state.tickTimer = setInterval(tick, 1000);
}
function showFailure(msg) {
  stopTimers();
  show("status");
  $("st-bar").parentElement.classList.remove("indeterminate");
  $("st-bar").style.width = "0";
  $("st-elapsed").textContent = "—";
  $("st-stage").textContent = "Обработка не выполнена";
  $("st-pct").textContent = "";
  $("st-error").textContent = msg;
  $("st-error").hidden = false;
  $("st-actions").hidden = false;
}
function openJob(jobId) {
  stopTimers();
  state.jobId = jobId;
  show("status");
  const known = recentLoad().find((e) => e.id === jobId);
  $("st-name").textContent = known?.name || `Задача ${jobId}`;
  $("st-error").hidden = true;
  $("st-actions").hidden = true;
  setProgress("Получение статуса…", 0);
  poll(jobId, true);
}
async function poll(jobId, first = false) {
  if (state.jobId !== jobId) return;
  let job;
  try {
    const r = await fetch(`/ui/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
    if (state.jobId !== jobId) return;
    if (r.status === 404) { showFailure(`Задача ${jobId} не найдена на сервере. Возможно, ссылка устарела.`); return; }
    if (!r.ok) throw new Error(String(r.status));
    job = await r.json();
  } catch (_) {
    if (state.jobId !== jobId) return;
    $("st-stage").textContent = "Нет связи с сервером, повторяем запрос…";
    state.pollTimer = setTimeout(() => poll(jobId), 3000);
    return;
  }
  if (state.jobId !== jobId) return;
  if (job.filename) $("st-name").textContent = job.filename;
  if (job.status === "processing") {
    if (first) startElapsed(job.created_at ? job.created_at * 1000 : Date.now());
    const q = job.queue_position ? ` · в очереди: ${job.queue_position}` : "";
    setProgress(`${job.stage || "Обработка"}${q}`, job.progress || 0);
    state.pollTimer = setTimeout(() => poll(jobId), 1500);
  } else if (job.status === "failed") {
    recentSave({ id: jobId, name: job.filename, status: "failed", ts: (job.created_at || Date.now() / 1000) * 1000 });
    showFailure(`Анализ завершился с ошибкой: ${job.error || "причина не указана"}.\nПопробуйте загрузить видео ещё раз; если ошибка повторится, файл, вероятно, повреждён или записан в неподдерживаемом формате.`);
  } else {
    stopTimers();
    recentSave({ id: jobId, name: job.filename, status: "done", verdict: job.verdict, count: (job.violations || []).length, ts: (job.created_at || Date.now() / 1000) * 1000 });
    render(jobId, job);
  }
}

/* ---------- result ---------- */
function render(jobId, job) {
  show("result");
  const meta = job.meta || {};
  const viol = (job.violations || []).slice().sort((a, b) => a.timecode_sec - b.timecode_sec);
  state.violations = viol;
  const bad = job.verdict === "violation" && viol.length > 0;

  const v = $("verdict");
  v.className = `verdict ${bad ? "bad" : "ok"}`;
  v.innerHTML = bad
    ? `<span class="ico">!</span><span>Обнаружены признаки возможных нарушений</span>`
    : `<span class="ico">✓</span><span>Нарушение не обнаружено</span>`;

  const dur = meta.duration_sec ?? job.video_duration_sec;
  const proc = job.processing_sec ?? meta.processing_sec;
  $("r-name").textContent = job.filename || jobId;
  $("r-dur").textContent = dur ? `${mmss(dur)} (${secs(dur)})` : "—";
  $("r-proc").textContent = proc !== undefined ? secs(proc) : "—";
  $("r-count").textContent = bad ? `${viol.length} ${plural(viol.length, "событие", "события", "событий")}` : "0";
  $("r-models").innerHTML = (meta.models || []).length ? meta.models.map((m) => `<span class="chip">${esc(m)}</span>`).join("") : '<span class="muted">—</span>';

  // technical details
  const rows = [];
  if (meta.fps) rows.push(["Частота кадров видео", `${meta.fps} к/с`]);
  if (meta.frames_analyzed) rows.push(["Проанализировано кадров", meta.frames_analyzed]);
  for (const [k, s] of Object.entries(meta.stage_sec || {})) rows.push([`Этап: ${STAGE_LABELS[k] || k}`, secs(s)]);
  rows.push(["Идентификатор задачи", jobId]);
  const parallel = Object.keys(meta.stage_sec || {}).length > 1 ? '<div class="muted">Этапы частично выполняются параллельно, поэтому их сумма может превышать общее время.</div>' : "";
  $("tech-body").innerHTML = `<table>${rows.map(([a, b]) => `<tr><td class="muted">${esc(a)}</td><td>${esc(b)}</td></tr>`).join("")}</table>${parallel}`;
  $("tech").hidden = false;

  const video = $("video");
  video.src = `/ui/jobs/${encodeURIComponent(jobId)}/video`;
  renderTimeline(dur || 0);
  renderCards(viol);
}

function renderTimeline(duration) {
  const tl = $("timeline");
  tl.querySelectorAll(".tl-mark").forEach((m) => m.remove());
  tl.dataset.duration = duration;
  const seen = new Map();
  state.violations.forEach((x, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tl-mark";
    b.style.background = colorOf(x.type_id);
    b.style.left = `${duration ? Math.min(100, (x.timecode_sec / duration) * 100) : 0}%`;
    b.title = `${i + 1}. ${x.type} — ${mmss(x.timecode_sec)}`;
    b.setAttribute("aria-label", b.title);
    b.dataset.idx = i;
    b.onclick = (e) => { e.stopPropagation(); focusViolation(i, true); };
    tl.append(b);
    if (!seen.has(x.type_id || x.type)) seen.set(x.type_id || x.type, x);
  });
  $("tl-legend").innerHTML = [...seen.values()].map((x) => `<span><i style="background:${colorOf(x.type_id)}"></i>${esc(x.type_id || "")} ${esc(shortType(x.type))}</span>`).join("");
  updatePlayhead();
}
function shortType(t) { const s = String(t || ""); return s.length > 48 ? `${s.slice(0, 46)}…` : s; }
// Timecodes written by the analyzer as mm:ss.s (e.g. «(00:15.4)») become seek links; input is already HTML-escaped.
function linkTimecodes(html) {
  return html.replace(/\b(\d{1,2}):(\d{2}(?:\.\d)?)\b/g, (m, mm, ss) => {
    const t = Number(mm) * 60 + Number(ss);
    return `<button type="button" class="tlink" data-t="${t}" title="Перейти к ${m}">${m}</button>`;
  });
}

function updatePlayhead() {
  const video = $("video");
  const dur = video.duration && isFinite(video.duration) ? video.duration : Number($("timeline").dataset.duration) || 0;
  const t = video.currentTime || 0;
  const pct = dur ? Math.min(100, (t / dur) * 100) : 0;
  $("tl-head").style.left = `${pct}%`;
  $("tl-played").style.width = `${pct}%`;
  $("tl-time").textContent = `${mmss(t)} / ${mmss(dur)}`;
}

function renderCards(viol) {
  const box = $("violations");
  box.innerHTML = "";
  state.frames = [];
  if (!viol.length) {
    box.innerHTML = '<div class="card empty-card"><b>Признаков нарушений кассовой дисциплины не найдено.</b><br><span class="muted">Сервис проверил видео по всем типам из справочника «Типы нарушений».</span></div>';
    return;
  }
  viol.forEach((x, i) => {
    const card = document.createElement("article");
    card.className = "card viol";
    card.id = `viol-${i}`;
    card.style.borderLeftColor = colorOf(x.type_id);
    const rule = x.rule || {};
    const strength = rule.strength || x.strength;
    const conf = typeof x.confidence === "number" ? Math.round(Math.min(1, Math.max(0, x.confidence)) * 100) : null;
    card.innerHTML = `
      <div class="viol-head">
        <h3>${x.type_id ? `<span class="badge" style="background:${colorOf(x.type_id)}">${esc(x.type_id)}</span>` : ""}${esc(x.type)}</h3>
        <button type="button" class="tc" title="Перейти к моменту в видео">▶ ${mmss(x.timecode_sec)}<small>${secs(x.timecode_sec)}</small></button>
      </div>
      ${conf !== null ? `<div class="conf">Уверенность <span class="track"><span class="fill" style="width:${conf}%;display:block"></span></span> ${conf}%</div>` : ""}
      <p class="desc">${linkTimecodes(esc(x.description))}</p>
      ${rule.document || rule.clause ? `
      <div class="rule">
        <div class="rule-title">
          <b>${esc(rule.short || "Норма")}${rule.clause ? `, ${esc(rule.clause)}` : ""}</b>
          ${strength ? `<span class="strength ${strength === "прямая" ? "direct" : "indirect"}" title="${strength === "прямая" ? "Норма прямо регулирует это действие" : "Норма применена по аналогии или из общего требования"}">связь с нормой: ${esc(strength)}</span>` : ""}
          ${rule.url ? `<a href="${esc(rule.url)}" target="_blank" rel="noopener">открыть на adilet.zan.kz ↗</a>` : ""}
        </div>
        ${rule.document ? `<div class="doc">${esc(rule.document)}</div>` : ""}
        ${rule.quote ? `<details><summary>Текст нормы</summary><blockquote>${esc(rule.quote)}</blockquote></details>` : ""}
      </div>` : ""}
      <div class="frames"></div>`;
    const frames = card.querySelector(".frames");
    (x.evidence_frames || []).forEach((f) => {
      const idx = state.frames.length;
      state.frames.push({ src: `data:image/jpeg;base64,${f.image_base64}`, caption: `${i + 1}. ${x.type} — ${mmss(f.timestamp_sec)} (${secs(f.timestamp_sec)})`, t: f.timestamp_sec });
      const fig = document.createElement("figure");
      fig.innerHTML = `<img alt="Кадр ${mmss(f.timestamp_sec)}" src="${state.frames[idx].src}"><figcaption>${mmss(f.timestamp_sec)}</figcaption>`;
      fig.querySelector("img").onclick = () => openLightbox(idx);
      frames.append(fig);
    });
    card.querySelector(".tc").onclick = () => focusViolation(i, false);
    card.querySelectorAll(".desc .tlink").forEach((a) => { a.onclick = () => seek(Number(a.dataset.t)); });
    box.append(card);
  });
}

function focusViolation(i, scrollCard) {
  const x = state.violations[i];
  if (!x) return;
  seek(x.timecode_sec);
  document.querySelectorAll(".viol.active, .tl-mark.active").forEach((e) => e.classList.remove("active"));
  const card = $(`viol-${i}`);
  card?.classList.add("active");
  document.querySelector(`.tl-mark[data-idx="${i}"]`)?.classList.add("active");
  if (scrollCard) card?.scrollIntoView({ behavior: "smooth", block: "start" });
}
function seek(t) {
  const video = $("video");
  const go = () => { video.currentTime = Math.max(0, Number(t) || 0); video.pause(); updatePlayhead(); };
  if (video.readyState >= 1) go(); else video.addEventListener("loadedmetadata", go, { once: true });
}

/* ---------- lightbox ---------- */
function openLightbox(idx) {
  if (!state.frames.length) return;
  state.lbIndex = (idx + state.frames.length) % state.frames.length;
  const f = state.frames[state.lbIndex];
  const lb = $("lightbox");
  lb.querySelector("img").src = f.src;
  lb.querySelector("figcaption").textContent = `${f.caption} · кадр ${state.lbIndex + 1} из ${state.frames.length}`;
  const multi = state.frames.length > 1;
  lb.querySelector(".lb-prev").hidden = !multi;
  lb.querySelector(".lb-next").hidden = !multi;
  lb.hidden = false;
}
function closeLightbox() { $("lightbox").hidden = true; }

/* ---------- violation types ---------- */
let typesLoaded = false;
async function openTypes() {
  const dlg = $("types-dlg");
  if (!dlg.open) dlg.showModal();
  if (typesLoaded) return;
  try {
    const r = await fetch("/ui/types", { cache: "no-store" });
    if (!r.ok) throw new Error(String(r.status));
    const data = await r.json();
    $("types-body").innerHTML = data.types.map((t) => {
      const norms = [t.rule, ...(t.extra_rules || [])].filter((n) => n && (n.short || n.document))
        .map((n) => `${n.url ? `<a href="${esc(n.url)}" target="_blank" rel="noopener">${esc(n.short || n.document)}, ${esc(n.clause || "")}</a>` : `${esc(n.short || n.document)}, ${esc(n.clause || "")}`}`).join(" · ");
      return `<div class="type-item">
        <h3><span class="badge" style="background:${colorOf(t.type_id)}">${esc(t.type_id === "GENERIC" ? "Иное" : t.type_id)}</span>${esc(t.name)}</h3>
        <p>${esc(t.summary)}</p>
        <div class="norms">Нормы: ${norms || "—"}${t.strength ? ` · связь: ${esc(t.strength)}` : ""}</div>
      </div>`;
    }).join("");
    $("types-sources").innerHTML = (data.sources || []).length ? `<div class="sources"><b>Источники</b><ul>${data.sources.map((s) => `<li><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a><br><span class="muted">${esc(s.requisites || "")}${s.status ? ` · ${esc(s.status)}` : ""}</span></li>`).join("")}</ul></div>` : "";
    typesLoaded = true;
  } catch (_) {
    $("types-body").innerHTML = '<div class="alert error">Не удалось загрузить справочник типов. Попробуйте ещё раз.</div>';
  }
}

/* ---------- routing & events ---------- */
function goHome() {
  stopTimers();
  state.jobId = null;
  state.xhr?.abort();
  $("video").removeAttribute("src");
  $("video").load();
  $("file").value = "";
  if (location.hash) history.pushState(null, "", location.pathname);
  recentRender();
  show("upload");
}
function route() {
  const m = location.hash.match(/job=([\w-]+)/);
  if (m) openJob(m[1]); else goHome();
}

$("nav-new").onclick = goHome;
$("home").onclick = (e) => { e.preventDefault(); goHome(); };
$("st-retry").onclick = goHome;
$("r-new").onclick = goHome;
$("nav-types").onclick = openTypes;
$("types-close").onclick = () => $("types-dlg").close();
$("types-dlg").addEventListener("click", (e) => { if (e.target === e.currentTarget) e.currentTarget.close(); });
$("recent-clear").onclick = () => { localStorage.removeItem(RECENT_KEY); recentRender(); };

$("file").onchange = (e) => { if (e.target.files[0]) upload(e.target.files[0]); };
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  if (files.length > 1) uploadError("Перетащите один файл: пакетная загрузка не поддерживается.");
  else if (files[0]) upload(files[0]);
});
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

const video = $("video");
["timeupdate", "seeked", "loadedmetadata", "durationchange"].forEach((ev) => video.addEventListener(ev, updatePlayhead));
$("timeline").addEventListener("click", (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  const dur = video.duration && isFinite(video.duration) ? video.duration : Number(e.currentTarget.dataset.duration) || 0;
  if (dur) seek(((e.clientX - r.left) / r.width) * dur);
});

const lb = $("lightbox");
lb.querySelector(".lb-prev").onclick = (e) => { e.stopPropagation(); openLightbox(state.lbIndex - 1); };
lb.querySelector(".lb-next").onclick = (e) => { e.stopPropagation(); openLightbox(state.lbIndex + 1); };
lb.querySelector(".lb-close").onclick = closeLightbox;
lb.addEventListener("click", (e) => { if (e.target === lb) closeLightbox(); });
document.addEventListener("keydown", (e) => {
  if (lb.hidden) return;
  if (e.key === "Escape") closeLightbox();
  else if (e.key === "ArrowLeft") openLightbox(state.lbIndex - 1);
  else if (e.key === "ArrowRight") openLightbox(state.lbIndex + 1);
});

window.addEventListener("hashchange", route);
recentRender();
route();
