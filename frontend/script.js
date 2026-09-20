const $ = (id) => document.getElementById(id);

const dbStatusEl = $("db-status");

// --- Ingest (Card 1) ---
const ingestDrop = $("ingest-drop");
const ingestInput = $("ingest-input");
const ingestPicked = $("ingest-picked");
const ingestBtn = $("ingest-btn");
const ingestStatus = $("ingest-status");
const ingestThumbs = $("ingest-thumbs");

// --- Search (Card 2) ---
const searchDrop = $("search-drop");
const searchInput = $("search-input");
const searchPicked = $("search-picked");
const queryPreview = $("query-preview");
const topkSel = $("topk");
const searchBtn = $("search-btn");
const searchStatus = $("search-status");
const resultsEl = $("results");

const DEFAULT_TOP_K = 5;

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function filenameFromPath(p) {
  return String(p).split(/[\\/]/).pop() || p;
}

function setStatus(el, text, kind) {
  el.textContent = text || "";
  el.className = "status" + (kind ? " " + kind : "");
}

// ---------------------------------------------------------------------------
// Header: database + engine status
// ---------------------------------------------------------------------------
async function fetchDbStatus() {
  try {
    const res = await fetch("/db-status");
    const d = await res.json();
    if (!d.connected) {
      dbStatusEl.textContent = "Database unreachable" + (d.error ? ` — ${d.error}` : "");
      dbStatusEl.style.color = "#b91c1c";
      return;
    }
    const rows = d.row_count != null ? Number(d.row_count).toLocaleString() : "?";
    const indexType = d.index_type || "?";
    const metric = d.metric_type || "?";
    dbStatusEl.textContent =
      `Collection ${d.collection} · ${rows} vectors · index: ${indexType} · metric: ${metric}`;
    dbStatusEl.style.color = "";
  } catch (e) {
    dbStatusEl.textContent = "Database unreachable";
    dbStatusEl.style.color = "#b91c1c";
  }
}

// ---------------------------------------------------------------------------
// Card 1: Ingest / create embeddings
// ---------------------------------------------------------------------------
function refreshIngestControls() {
  const n = ingestInput.files ? ingestInput.files.length : 0;
  ingestBtn.disabled = n === 0;
  ingestPicked.textContent = n
    ? `${n} file${n > 1 ? "s" : ""} selected: ` +
      Array.from(ingestInput.files).map((f) => f.name).join(", ")
    : "";
}

ingestDrop.addEventListener("click", () => ingestInput.click());
ingestInput.addEventListener("change", refreshIngestControls);

ingestBtn.addEventListener("click", async () => {
  const files = ingestInput.files;
  if (!files || files.length === 0) return;

  setStatus(ingestStatus, `Embedding ${files.length} image(s)… this can take a while.`);
  ingestThumbs.innerHTML = "";
  ingestBtn.disabled = true;

  const formData = new FormData();
  for (const f of files) formData.append("files", f);

  try {
    const res = await fetch("/ingest", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      setStatus(ingestStatus, data.detail || "Ingest failed.", "err");
      return;
    }

    const skipped = data.total - data.indexed;
    const rows = data.db_row_count != null ? Number(data.db_row_count).toLocaleString() : "?";

    let msg = `Done — indexed ${data.indexed} of ${data.total} image(s). ` +
      `Database now holds ${rows} vectors.`;
    let kind = "ok";
    if (skipped > 0) { msg += `\n${skipped} image(s) were not indexed (see details below).`; kind = "warn"; }
    setStatus(ingestStatus, msg, kind);

    const lines = [];
    (data.results || []).forEach((r) => {
      const line = document.createElement("div");
      if (r.status === "indexed") {
        line.innerHTML = `${escapeHtml(r.filename)} — <span class="ok">indexed</span>` +
          (r.num_faces_detected ? ` (faces: ${r.num_faces_detected})` : "");
        const img = document.createElement("img");
        img.src = r.image_url;
        img.alt = r.filename;
        img.title = r.filename;
        ingestThumbs.appendChild(img);
      } else {
        line.innerHTML = `${escapeHtml(r.filename)} — <span style="color:var(--err)">${escapeHtml(r.reason || r.status)}</span>`;
      }
      lines.push(line);
    });
    const detail = document.createElement("div");
    detail.className = "picked";
    lines.forEach((l) => detail.appendChild(l));
    ingestStatus.appendChild(detail);

    ingestInput.value = "";
    refreshIngestControls();
    fetchDbStatus();
  } catch (e) {
    setStatus(ingestStatus, "Request failed: " + e, "err");
  } finally {
    ingestBtn.disabled = true;
  }
});

// ---------------------------------------------------------------------------
// Card 2: Similarity search
// ---------------------------------------------------------------------------
function refreshSearchControls() {
  const has = searchInput.files && searchInput.files.length > 0;
  searchBtn.disabled = !has;
  if (has) {
    const file = searchInput.files[0];
    searchPicked.textContent = `Query: ${file.name}`;
    queryPreview.innerHTML = "";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.alt = "Query image";
    queryPreview.appendChild(img);
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = "Your selected image";
    queryPreview.appendChild(label);
  } else {
    searchPicked.textContent = "";
    queryPreview.innerHTML = "";
  }
}

searchDrop.addEventListener("click", () => searchInput.click());
searchInput.addEventListener("change", () => {
  refreshSearchControls();
  setStatus(searchStatus, "", "");
  resultsEl.innerHTML = "";
});

searchBtn.addEventListener("click", async () => {
  const file = searchInput.files && searchInput.files[0];
  if (!file) return;

  setStatus(searchStatus, "Embedding query + searching…");
  resultsEl.innerHTML = "";
  searchBtn.disabled = true;

  const topK = parseInt(topkSel.value, 10) || DEFAULT_TOP_K;
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`/search?top_k=${topK}`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      setStatus(searchStatus, data.detail || "Search failed.", "err");
      return;
    }

    let msg = `Faces detected in query: ${data.num_faces_detected}`;
    if (!data.results || data.results.length === 0) {
      msg += " — no results returned (is the database empty?).";
      setStatus(searchStatus, msg, "warn");
      return;
    }
    setStatus(searchStatus, msg, "ok");

    data.results.forEach((r, i) => {
      resultsEl.appendChild(resultCard(r, i));
    });
  } catch (e) {
    setStatus(searchStatus, "Request failed: " + e, "err");
  } finally {
    refreshSearchControls();
  }
});

function resultCard(r, i) {
  const card = document.createElement("div");
  card.className = "result-card";

  const wrap = document.createElement("div");
  wrap.className = "imgwrap";

  const img = document.createElement("img");
  img.loading = "lazy";
  img.alt = filenameFromPath(r.image_path);
  img.onerror = () => {
    wrap.innerHTML = '<div class="placeholder">image unavailable<br>(not on this host)</div>';
  };
  img.src = r.image_url;
  wrap.appendChild(img);

  const meta = document.createElement("div");
  meta.className = "meta";

  const score = Number(r.score);
  const pct = Math.round(score * 100);
  const barColor = pct >= 50
    ? `linear-gradient(90deg, #4ade80, #22c55e)`
    : `linear-gradient(90deg, #f87171, #ef4444)`;

  const source = r.dataset_source || "";
  const person = r.person_id ? ` · ${escapeHtml(r.person_id)}` : "";

  meta.innerHTML =
    `<div class="rank">#${i + 1} · <span class="score">${score.toFixed(4)}</span></div>` +
    `<div class="similarity-bar-wrap">` +
      `<div class="similarity-bar-fill" style="width:${pct}%;background:${barColor}"></div>` +
    `</div>` +
    `<div class="similarity-label"><span>${pct}%</span><span>similarity</span></div>` +
    `<div class="src">${escapeHtml(source)}${person}</div>` +
    `<div class="fname" title="${escapeHtml(r.image_path)}">${escapeHtml(filenameFromPath(r.image_path))}</div>`;

  card.appendChild(wrap);
  card.appendChild(meta);
  return card;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
fetchDbStatus();
refreshIngestControls();
refreshSearchControls();
