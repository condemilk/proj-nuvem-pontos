const $ = s => document.querySelector(s);
const viewer = new Viewer($("#view"));
const state = { cloudId: null };
const STATUS = { ok: "dentro", atencao: "atenção", fora: "fora", "n/d": "n/d" };
const fmt = (v, d = 1) => v == null ? "—" : Number(v).toLocaleString("pt-BR", { maximumFractionDigits: d, minimumFractionDigits: d });
const msg = (t, bad = false) => { $("#msg").textContent = t; $("#msg").style.color = bad ? "var(--bad)" : ""; };

async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(Array.isArray(j.detail) ? j.detail.map(d => d.msg).join("; ") : (j.detail || r.statusText));
  return j;
}
const chip = s => `<span class="chip s-${s === "n/d" ? "nd" : s}">${STATUS[s]}</span>`;

// ---------- entradas ----------
function refPayload() {
  return { width: Number($("#refW").value), height: Number($("#refH").value) };
}
const tolPayload = () => ({ dimension_mm: +$("#tDim").value, plumb_mm_per_m: +$("#tPlumb").value });

async function onCloud(info) {
  state.cloudId = info.cloud_id; $("#btnRun").disabled = false;
  $("#cloudInfo").textContent = `${info.filename}: ${info.n_points.toLocaleString("pt-BR")} pontos.` + (info.warnings?.length ? " Atenção: " + info.warnings.join(" ") : "");
  const p = await api(`/api/cloud/${info.cloud_id}/preview`);
  viewer.set(p.points);
}

$("#cloudFile").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f); fd.append("unit", $("#unit").value);
  msg("Carregando nuvem…");
  try { await onCloud(await api("/api/cloud", { method: "POST", body: fd })); msg(""); } catch (err) { msg(err.message, true); }
};

// ---------- análise ----------
function renderResult(r) {
  const b = $("#banner"); b.hidden = false; b.className = `chip s-${r.overall_status === "n/d" ? "nd" : r.overall_status}`;
  b.textContent = "Situação geral: " + STATUS[r.overall_status];
  $("#tbl tbody").innerHTML = r.measurements.map(m => `<tr><td title="${m.note || ""}">${m.label}${m.note ? " ⓘ" : ""}</td>
    <td>${fmt(m.measured)} ${m.unit}</td><td>${fmt(m.reference)}</td><td>${fmt(m.deviation)}</td><td>±${fmt(m.tolerance)}</td><td>${chip(m.status)}</td></tr>`).join("");
  $("#stats").textContent = `${r.n_points_used} pontos usados no plano identificado.`;
  $("#warn").innerHTML = [...(r.cloud_warnings || []), ...r.warnings].map(w => `<li>${w}</li>`).join("");
  if (r.viz) viewer.set(r.viz.points, r.viz.corners);
}

$("#btnRun").onclick = async () => {
  msg("Analisando…");
  const body = { cloud_id: state.cloudId, reference: refPayload(), tolerances: tolPayload(),
    ransac_threshold_mm: Number($("#ransacThreshold").value) };
  try { renderResult(await api("/api/inspect", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })); msg("Inspeção salva no histórico."); }
  catch (err) { msg(err.message, true); }
};

$("#btnPrint").onclick = () => window.print();
