const state = { csrfToken: null, experiment: null, pollTimer: null };

const elements = {
  signin: document.querySelector("#signin-required"),
  setup: document.querySelector("#setup-section"),
  setupForm: document.querySelector("#setup-form"),
  setupMessage: document.querySelector("#setup-message"),
  recent: document.querySelector("#recent-tests"),
  workspace: document.querySelector("#workspace-section"),
  experimentName: document.querySelector("#experiment-name"),
  experimentStatus: document.querySelector("#experiment-status"),
  steps: document.querySelector("#workflow-steps"),
  newTest: document.querySelector("#new-test"),
  results: document.querySelector("#results-section"),
  metricGrid: document.querySelector("#metric-grid"),
  matchTable: document.querySelector("#match-table"),
  sourceTemplate: document.querySelector("#source-template"),
};

function formatTime(value) {
  if (value === null || value === undefined) return "—";
  const seconds = Number(value);
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

function statusText(value) {
  return String(value || "").replaceAll("_", " ");
}

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.csrfToken && options.method && options.method !== "GET") {
    headers.set("X-CSRF-Token", state.csrfToken);
  }
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "The request failed");
  return payload;
}

function setBusy(form, busy, copy = "WORKING…") {
  const button = form.querySelector("button[type='submit']");
  if (!button) return;
  if (!button.dataset.original) button.dataset.original = button.textContent;
  button.disabled = busy;
  button.textContent = busy ? copy : button.dataset.original;
}

function expectedRows(source) {
  if (source.role === "holdout") return state.experiment.holdout.actual_clips;
  return state.experiment.reference_clips.filter(
    (clip) => clip.source_key === source.source_key,
  );
}

function renderAlignments(container, rows) {
  container.replaceChildren();
  rows
    .filter((row) => row.uploaded_filename && row.source_start_seconds !== null)
    .forEach((row) => {
      const line = document.createElement("form");
      line.className = "alignment-row";
      const name = document.createElement("strong");
      name.textContent = row.content_id;
      const start = document.createElement("input");
      start.type = "number";
      start.step = "0.1";
      start.min = "0";
      start.value = Number(row.source_start_seconds).toFixed(1);
      start.setAttribute("aria-label", `${row.content_id} source start`);
      const end = document.createElement("input");
      end.type = "number";
      end.step = "0.1";
      end.min = "0";
      end.value = Number(row.source_end_seconds).toFixed(1);
      end.setAttribute("aria-label", `${row.content_id} source end`);
      const confidence = document.createElement("span");
      confidence.className = "small-print";
      confidence.textContent =
        row.alignment_method === "manual"
          ? "MANUAL"
          : `${Math.round(Number(row.alignment_score || 0) * 100)}% AUDIO MATCH${row.compound_edit_detected ? " · COMPOUND EDIT" : ""}`;
      const button = document.createElement("button");
      button.type = "submit";
      button.className = "plain-button";
      button.textContent = "SAVE TIMING";
      line.append(name, start, end, confidence, button);
      line.addEventListener("submit", async (event) => {
        event.preventDefault();
        setBusy(line, true, "SAVING…");
        try {
          const payload = await request(
            `/api/backtests/${encodeURIComponent(state.experiment.id)}/alignments/${encodeURIComponent(row.id)}`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                start_seconds: Number(start.value),
                end_seconds: Number(end.value),
              }),
            },
          );
          renderExperiment(payload.experiment);
        } catch (error) {
          confidence.textContent = error.message;
        } finally {
          setBusy(line, false);
        }
      });
      container.append(line);
    });
}

function sourceDescription(source) {
  if (source.role === "holdout") {
    if (state.experiment.holdout.predictions_frozen) {
      return "CreatorCut has frozen these recommendations. You can now reveal the organization-selected Shorts without changing the saved ranking.";
    }
    return "This video stays held out. Upload only the long video after both reference sets finish.";
  }
  return "Upload the long video and every corresponding published Short. CreatorCut will locate each Short inside the source and connect its public outcome.";
}

function renderPredictions(card) {
  const snapshot = state.experiment.prediction_snapshot;
  if (!snapshot) return;
  const list = document.createElement("div");
  list.className = "prediction-list";
  snapshot.recommendations.forEach((clip) => {
    const item = document.createElement("article");
    const rank = document.createElement("strong");
    rank.textContent = `#${clip.rank}`;
    const interval = document.createElement("span");
    interval.textContent = `${formatTime(clip.start_seconds)}–${formatTime(clip.end_seconds)} · ESTIMATED FIT ${Math.round(clip.estimated_relative_performance)}/100`;
    item.append(rank, interval);
    list.append(item);
  });
  card.append(list);

  if (!state.experiment.evaluation) {
    const form = document.createElement("form");
    form.className = "backtest-form holdout-clips-form";
    const label = document.createElement("label");
    label.textContent = "Published test Shorts";
    const input = document.createElement("input");
    input.name = "shorts";
    input.type = "file";
    input.multiple = true;
    input.required = true;
    input.accept = "video/mp4,video/quicktime,video/webm,.m4v";
    label.append(input);
    const button = document.createElement("button");
    button.type = "submit";
    button.textContent = "REVEAL SHORTS AND SCORE TEST";
    const message = document.createElement("p");
    message.className = "form-message error";
    message.hidden = true;
    form.append(label, button, message);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      message.hidden = true;
      setBusy(form, true, "ALIGNING AND SCORING…");
      try {
        const payload = await request(
          `/api/backtests/${encodeURIComponent(state.experiment.id)}/holdout-clips`,
          { method: "POST", body: new FormData(form) },
        );
        renderExperiment(payload.experiment);
      } catch (error) {
        message.textContent = error.message;
        message.hidden = false;
      } finally {
        setBusy(form, false);
      }
    });
    card.append(form);
    const expected = document.createElement("p");
    expected.className = "small-print";
    expected.textContent = `Expected: ${state.experiment.holdout.actual_clips.map((row) => row.content_id).join(", ")}`;
    card.append(expected);
  }
}

function renderSource(source) {
  const card = elements.sourceTemplate.content.firstElementChild.cloneNode(true);
  card.querySelector(".source-role").textContent =
    source.role === "holdout" ? "03 / HELD-OUT TEST VIDEO" : "02 / REFERENCE VIDEO";
  card.querySelector(".source-key").textContent = source.source_key;
  card.querySelector(".source-status").textContent = statusText(source.status);
  card.querySelector(".source-copy").textContent = sourceDescription(source);
  const form = card.querySelector(".source-form");
  form.elements.role.value = source.role;
  form.elements.source_key.value = source.source_key;
  const shortLabel = card.querySelector(".short-input");
  if (source.role === "holdout") {
    shortLabel.hidden = true;
    form.elements.shorts.required = false;
  }
  const shouldShowUpload = source.status === "awaiting_upload";
  form.hidden = !shouldShowUpload;
  if (shouldShowUpload && source.role === "holdout" && !state.experiment.can_upload_holdout) {
    form.querySelectorAll("input, button").forEach((control) => { control.disabled = true; });
  }
  if (shouldShowUpload) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = form.querySelector(".form-message");
      message.hidden = true;
      setBusy(form, true, "UPLOADING…");
      try {
        const payload = await request(
          `/api/backtests/${encodeURIComponent(state.experiment.id)}/sources`,
          { method: "POST", body: new FormData(form) },
        );
        renderExperiment(payload.experiment);
        schedulePoll();
      } catch (error) {
        message.textContent = error.message;
        message.hidden = false;
      } finally {
        setBusy(form, false);
      }
    });
  }
  const rows = expectedRows(source);
  const expected = card.querySelector(".expected-files");
  if (source.role === "reference") {
    expected.textContent = `Expected Short filenames: ${rows.map((row) => `${row.content_id}.mp4`).join(", ")}`;
  } else {
    expected.textContent = `${state.experiment.holdout.expected_clip_count} published test Shorts remain sealed until recommendations freeze.`;
  }
  renderAlignments(card.querySelector(".alignment-list"), rows);
  if (source.role === "holdout" && state.experiment.holdout.predictions_frozen) {
    form.hidden = true;
    renderPredictions(card);
  }
  return card;
}

function renderEvaluation(evaluation) {
  elements.results.hidden = !evaluation;
  if (!evaluation) return;
  const metrics = [
    [evaluation.prediction_count, "recommendations"],
    [evaluation.aligned_organization_clip_count, "published Shorts aligned"],
    [evaluation.top_k_recall_at_iou_50 === null ? "—" : `${Math.round(evaluation.top_k_recall_at_iou_50 * 100)}%`, "published moments recovered"],
    [evaluation.performance_rank_correlation === null ? "—" : evaluation.performance_rank_correlation.toFixed(2), "performance rank correlation"],
  ];
  elements.metricGrid.replaceChildren();
  metrics.forEach(([value, label]) => {
    const card = document.createElement("article");
    const strong = document.createElement("strong");
    strong.textContent = value;
    const span = document.createElement("span");
    span.textContent = label;
    card.append(strong, span);
    elements.metricGrid.append(card);
  });
  elements.matchTable.replaceChildren();
  evaluation.matches.forEach((match) => {
    const row = document.createElement("tr");
    const values = [
      `#${match.prediction_rank}`,
      `${formatTime(match.prediction_start_seconds)}–${formatTime(match.prediction_end_seconds)}`,
      match.actual_content_id || "No close match",
      `${Math.round(match.temporal_iou * 100)}%${match.recovered ? " recovered" : ""}`,
      `${Math.round(match.estimated_relative_performance)}/100`,
      match.actual_performance_score === null
        ? "—"
        : `${Math.round(match.actual_performance_score)}/100 · ${Number(match.views).toLocaleString()} views`,
    ];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    elements.matchTable.append(row);
  });
  elements.results.scrollIntoView({ block: "start", behavior: "smooth" });
}

function renderExperiment(experiment) {
  state.experiment = experiment;
  localStorage.setItem("creatorcut_backtest_id", experiment.id);
  elements.setup.hidden = true;
  elements.workspace.hidden = false;
  elements.experimentName.textContent = experiment.name;
  elements.experimentStatus.textContent = `${statusText(experiment.status)} · ${experiment.aligned_reference_clip_count}/${experiment.reference_clip_count} reference Shorts aligned`;
  elements.steps.replaceChildren();
  experiment.sources.forEach((source) => elements.steps.append(renderSource(source)));
  renderEvaluation(experiment.evaluation);
  if (experiment.sources.some((source) => source.status === "processing")) schedulePoll();
}

async function refreshExperiment() {
  if (!state.experiment) return;
  try {
    const payload = await request(`/api/backtests/${encodeURIComponent(state.experiment.id)}`);
    renderExperiment(payload.experiment);
  } catch (error) {
    elements.experimentStatus.textContent = error.message;
  }
}

function schedulePoll() {
  window.clearTimeout(state.pollTimer);
  state.pollTimer = window.setTimeout(async () => {
    await refreshExperiment();
  }, 3000);
}

elements.setupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  elements.setupMessage.hidden = true;
  setBusy(elements.setupForm, true, "CREATING…");
  try {
    const payload = await request("/api/backtests", {
      method: "POST",
      body: new FormData(elements.setupForm),
    });
    renderExperiment(payload.experiment);
  } catch (error) {
    elements.setupMessage.textContent = error.message;
    elements.setupMessage.hidden = false;
  } finally {
    setBusy(elements.setupForm, false);
  }
});

elements.newTest.addEventListener("click", () => {
  window.clearTimeout(state.pollTimer);
  state.experiment = null;
  localStorage.removeItem("creatorcut_backtest_id");
  elements.workspace.hidden = true;
  elements.results.hidden = true;
  elements.setup.hidden = false;
});

async function loadRecent() {
  const payload = await request("/api/account/backtests");
  const list = elements.recent.querySelector("ul");
  list.replaceChildren();
  payload.experiments.forEach((experiment) => {
    const item = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = `${experiment.name} — ${statusText(experiment.status)}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "plain-button";
    button.textContent = "OPEN";
    button.addEventListener("click", async () => {
      const result = await request(`/api/backtests/${encodeURIComponent(experiment.id)}`);
      renderExperiment(result.experiment);
    });
    item.append(name, button);
    list.append(item);
  });
  elements.recent.hidden = payload.experiments.length === 0;
  return payload.experiments;
}

async function initialize() {
  try {
    const session = await request("/api/auth/session");
    if (!session.authenticated) {
      elements.signin.hidden = false;
      return;
    }
    state.csrfToken = session.csrf_token;
    elements.setup.hidden = false;
    const experiments = await loadRecent();
    const savedId = localStorage.getItem("creatorcut_backtest_id");
    const candidate = experiments.find((item) => item.id === savedId);
    if (candidate) {
      const payload = await request(`/api/backtests/${encodeURIComponent(candidate.id)}`);
      renderExperiment(payload.experiment);
    }
  } catch (error) {
    elements.signin.hidden = false;
  }
}

initialize();
