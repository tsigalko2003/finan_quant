const $ = (selector) => document.querySelector(selector);
const form = $("#jobForm");
let defaults = null;
let selectedRun = null;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}

async function api(path, options = {}) {
  const response = await fetch(path, {headers: {"Content-Type": "application/json"}, ...options});
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail;
    if (Array.isArray(detail)) throw new Error(detail.map(item => `${item.loc.at(-1)}: ${item.msg}`).join("\n"));
    throw new Error(detail || payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function splitTickers(value) {
  return value.split(",").map(item => item.trim().toUpperCase()).filter(Boolean);
}

function formPayload() {
  const data = new FormData(form);
  const numeric = [
    "refresh_tail_days", "short_window", "mid_window", "long_window", "correlation_window",
    "pca_window", "hmm_states", "capture_window", "rebound_horizon", "rebound_return_pct",
    "max_adverse_pct", "correction_drawdown_pct", "minimum_correlation",
    "rebound_probability", "minimum_breadth_thrust_pct", "min_coverage_pct"
  ];
  const payload = {};
  for (const [key, value] of data.entries()) payload[key] = value;
  numeric.forEach(key => payload[key] = Number(payload[key]));
  payload.max_tickers = payload.max_tickers ? Number(payload.max_tickers) : null;
  payload.start = payload.start || null;
  payload.end = payload.end || null;
  payload.include = splitTickers(payload.include || "");
  payload.exclude = splitTickers(payload.exclude || "");
  payload.force_download = data.has("force_download");
  payload.refresh_universe = data.has("refresh_universe");
  return payload;
}

function updateStageNote() {
  const prod = $("#stage").value === "prod";
  const dynamic = $("#industry").value.startsWith("nasdaq:");
  $("#stageNote").textContent = prod
    ? `${dynamic ? "Nasdaq production uses every eligible match from the cached export. " : "Production uses the full configured universe. "}The default history is 15 years; confirm runtime before submitting.`
    : "POC caps the resolved universe to six market-cap-sorted names and shorter history for workflow validation.";
}

function renderOptions(options) {
  defaults = options.defaults;
  $("#industryOptions").innerHTML = options.industries.map(item =>
    `<option value="${escapeHtml(item.name)}">${escapeHtml(item.description)} · ${item.tickers} tickers</option>`
  ).join("");
}

function resetForm() {
  form.reset();
  if (defaults) {
    for (const [key, value] of Object.entries(defaults)) {
      const input = form.elements.namedItem(key);
      if (!input || value === null || Array.isArray(value)) continue;
      if (input.type === "checkbox") input.checked = Boolean(value);
      else input.value = value;
    }
  }
  updateStageNote();
  $("#formError").textContent = "";
}

async function submitJob(action) {
  $("#formError").textContent = "";
  if ($("#stage").value === "prod" && action !== "analyze") {
    if (!window.confirm("Start a production-style job with a larger universe and history?")) return;
  }
  try {
    const job = await api(`/api/jobs/${action}`, {method: "POST", body: JSON.stringify(formPayload())});
    renderJobs([job, ...(await api("/api/jobs")).filter(item => item.id !== job.id)]);
  } catch (error) {
    $("#formError").textContent = error.message;
  }
}

function jobCard(job) {
  const request = job.request || {};
  const message = job.error?.detail || job.message || "";
  return `<article class="job">
    <div><strong>${escapeHtml(job.action.toUpperCase())} · ${escapeHtml(request.industry)} · ${escapeHtml(request.stage)}</strong>
      <p>${escapeHtml(message)}</p></div>
    <span class="status ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span>
    <div class="progress"><span style="width:${Number(job.progress || 0)}%"></span></div>
  </article>`;
}

function renderJobs(jobs) {
  $("#jobs").innerHTML = jobs.length ? jobs.slice(0, 6).map(jobCard).join("") : '<div class="empty">No jobs submitted yet.</div>';
}

function runCard(run) {
  const alert = run.alert || {};
  const state = alert.triggered ? "Rebound trigger" : (alert.watch ? "Correction watch" : "No active signal");
  const selected = selectedRun === run.id ? " selected" : "";
  return `<article class="run${selected}" data-run-id="${escapeHtml(run.id)}">
    <div><strong>${escapeHtml(run.industry || "Unknown")} · ${escapeHtml((run.stage || "").toUpperCase())}</strong>
      <p>${escapeHtml(run.actual_range?.end || "No date")} · ${escapeHtml(state)}</p></div>
    <span class="status ${alert.triggered ? "completed" : ""}">${alert.triggered ? "Trigger" : "View"}</span>
  </article>`;
}

function renderRuns(runs) {
  $("#runs").innerHTML = runs.length ? runs.map(runCard).join("") : '<div class="empty">No completed reports.</div>';
  document.querySelectorAll("[data-run-id]").forEach(item => item.addEventListener("click", () => selectRun(item.dataset.runId)));
}

function metric(label, value) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value ?? "—")}</strong></div>`;
}

async function selectRun(runId) {
  try {
    const detail = await api(`/api/runs/${encodeURIComponent(runId)}`);
    selectedRun = runId;
    const manifest = detail.manifest || {};
    const alert = detail.alert || {};
    const validation = detail.validation || {};
    $("#reportTitle").textContent = `${manifest.universe?.name || "Sector"} · ${(manifest.stage || "").toUpperCase()}`;
    $("#snapshot").innerHTML = [
      metric("Signal", alert.triggered ? "Rebound trigger" : (alert.watch ? "Correction watch" : "No active signal")),
      metric("Data through", alert.data_through),
      metric("OOS ROC-AUC", validation.roc_auc == null ? validation.status : Number(validation.roc_auc).toFixed(3)),
      metric("OOS events", validation.positive_events)
    ].join("");
    $("#snapshot").classList.remove("hidden");
    $("#reportEmpty").classList.add("hidden");
    $("#reportFrame").src = detail.report_url;
    $("#reportFrame").classList.remove("hidden");
    $("#openReport").href = detail.report_url;
    $("#openReport").classList.remove("hidden");
    renderRuns(await api("/api/runs"));
  } catch (error) {
    $("#formError").textContent = error.message;
  }
}

async function refresh() {
  try {
    const [jobs, runs, options] = await Promise.all([api("/api/jobs"), api("/api/runs"), api("/api/options")]);
    renderOptions(options);
    renderJobs(jobs);
    renderRuns(runs);
    const newestCompleted = jobs.find(job => job.status === "completed" && job.result?.analysis?.run_dir);
    if (newestCompleted && !selectedRun) {
      const runId = newestCompleted.result.analysis.run_dir.split("/").at(-1);
      if (runs.some(run => run.id === runId)) selectRun(runId);
    }
  } catch (error) {
    console.error(error);
  }
}

async function initialize() {
  try {
    const [health, options] = await Promise.all([api("/health/ready"), api("/api/options")]);
    renderOptions(options);
    $("#healthText").textContent = health.status;
    $(".health").classList.add("ready");
    resetForm();
    await refresh();
    setInterval(refresh, 2500);
  } catch (error) {
    $("#healthText").textContent = "Unavailable";
    $("#formError").textContent = error.message;
  }
}

document.querySelectorAll("[data-action]").forEach(button => button.addEventListener("click", () => submitJob(button.dataset.action)));
$("#resetButton").addEventListener("click", resetForm);
$("#refreshButton").addEventListener("click", refresh);
$("#stage").addEventListener("change", updateStageNote);
$("#industry").addEventListener("input", updateStageNote);
initialize();
