const SCORE_FIELDS = ["hook", "completeness", "payoff", "clarity"];

const state = {
  tasks: [],
  index: 0,
  scores: {},
  stats: { total: 0, completed: 0, remaining: 0 },
  savedInBatch: 0,
};

const elements = {
  loading: document.querySelector("#loading-state"),
  error: document.querySelector("#error-state"),
  fatalError: document.querySelector("#fatal-error"),
  workspace: document.querySelector("#workspace"),
  complete: document.querySelector("#complete-state"),
  completeMessage: document.querySelector("#complete-message"),
  overallCount: document.querySelector("#overall-count"),
  overallProgress: document.querySelector("#overall-progress-bar"),
  batchIndex: document.querySelector("#batch-index"),
  batchTotal: document.querySelector("#batch-total"),
  videoId: document.querySelector("#video-id"),
  duration: document.querySelector("#clip-duration"),
  range: document.querySelector("#clip-range"),
  video: document.querySelector("#clip-video"),
  videoLoading: document.querySelector("#video-loading"),
  transcript: document.querySelector("#transcript-text"),
  form: document.querySelector("#review-form"),
  notes: document.querySelector("#notes"),
  technicalIssue: document.querySelector("#technical-issue"),
  formError: document.querySelector("#form-error"),
  save: document.querySelector("#save-button"),
  skip: document.querySelector("#skip-button"),
  replay: document.querySelector("#replay-button"),
  retry: document.querySelector("#retry-button"),
  nextBatch: document.querySelector("#next-batch-button"),
};

function createScoreButtons() {
  document.querySelectorAll(".score-field").forEach((fieldElement) => {
    const field = fieldElement.dataset.field;
    const group = fieldElement.querySelector(".score-group");
    for (let score = 1; score <= 5; score += 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "score-button";
      button.dataset.field = field;
      button.dataset.score = String(score);
      button.setAttribute("role", "radio");
      button.setAttribute("aria-checked", "false");
      button.setAttribute("aria-label", `${field} score ${score} out of 5`);
      button.textContent = String(score);
      button.addEventListener("click", () => selectScore(field, score));
      group.append(button);
    }
  });
}

function selectScore(field, score) {
  state.scores[field] = score;
  document.querySelectorAll(`.score-button[data-field="${field}"]`).forEach((button) => {
    const selected = Number(button.dataset.score) === score;
    button.setAttribute("aria-checked", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  clearFormError();
}

function resetForm() {
  state.scores = {};
  elements.notes.value = "";
  elements.technicalIssue.checked = false;
  document.querySelectorAll(".score-button").forEach((button) => {
    button.setAttribute("aria-checked", "false");
    button.tabIndex = 0;
  });
  clearFormError();
}

function formatTime(seconds) {
  const rounded = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(rounded / 60);
  const remainder = rounded % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function currentTask() {
  return state.tasks[state.index];
}

function updateOverallProgress() {
  const { completed, total } = state.stats;
  const percentage = total ? (completed / total) * 100 : 0;
  elements.overallCount.textContent = `${completed} of ${total} reviewed`;
  elements.overallProgress.style.width = `${percentage}%`;
}

function renderTask() {
  const task = currentTask();
  if (!task) {
    showBatchComplete();
    return;
  }

  resetForm();
  elements.batchIndex.textContent = String(state.index + 1);
  elements.batchTotal.textContent = String(state.tasks.length);
  elements.videoId.textContent = task.video_id.toUpperCase();
  elements.duration.textContent = formatTime(task.duration_seconds);
  elements.range.textContent = `${formatTime(task.start_seconds)} → ${formatTime(task.end_seconds)}`;
  elements.transcript.textContent = task.transcript_text || "No transcript text is available.";
  elements.videoLoading.hidden = false;
  elements.video.src = `/api/video/${encodeURIComponent(task.video_id)}`;
  elements.video.load();
}

async function loadBatch() {
  showState("loading");
  try {
    const response = await fetch("/api/tasks?limit=10", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`The server returned ${response.status}.`);
    }
    const payload = await response.json();
    state.tasks = payload.tasks;
    state.stats = payload.stats;
    state.index = 0;
    state.savedInBatch = 0;
    updateOverallProgress();
    if (!state.tasks.length) {
      showAllComplete();
      return;
    }
    showState("workspace");
    renderTask();
  } catch (error) {
    elements.fatalError.textContent = error.message || "The local server could not be reached.";
    showState("error");
  }
}

function showState(name) {
  elements.loading.hidden = name !== "loading";
  elements.error.hidden = name !== "error";
  elements.workspace.hidden = name !== "workspace";
  elements.complete.hidden = name !== "complete";
}

function showBatchComplete() {
  const skipped = state.tasks.length - state.savedInBatch;
  const skippedCopy = skipped
    ? ` ${skipped} skipped clip${skipped === 1 ? "" : "s"} will return in a future batch.`
    : "";
  elements.completeMessage.textContent =
    `${state.savedInBatch} review${state.savedInBatch === 1 ? "" : "s"} saved locally.` + skippedCopy;
  elements.nextBatch.hidden = state.stats.remaining === 0;
  showState("complete");
}

function showAllComplete() {
  elements.completeMessage.textContent =
    "Every candidate in the current queue has a saved human review.";
  elements.nextBatch.hidden = true;
  showState("complete");
}

function showFormError(message) {
  elements.formError.textContent = message;
  elements.formError.hidden = false;
}

function clearFormError() {
  elements.formError.textContent = "";
  elements.formError.hidden = true;
}

async function saveReview(event) {
  event.preventDefault();
  const missing = SCORE_FIELDS.filter((field) => !state.scores[field]);
  if (missing.length) {
    showFormError(`Please score ${missing.join(", ")} before saving.`);
    return;
  }

  const task = currentTask();
  const payload = {
    ...state.scores,
    technically_exportable: !elements.technicalIssue.checked,
    notes: elements.notes.value,
  };
  elements.save.disabled = true;
  elements.save.textContent = "Saving…";
  try {
    const response = await fetch(`/api/reviews/${encodeURIComponent(task.annotation_id)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "The review could not be saved.");
    }
    state.savedInBatch += 1;
    state.stats.completed += 1;
    state.stats.remaining -= 1;
    state.index += 1;
    updateOverallProgress();
    renderTask();
  } catch (error) {
    showFormError(error.message || "The review could not be saved.");
  } finally {
    elements.save.disabled = false;
    elements.save.innerHTML = 'Save & next <span aria-hidden="true">→</span>';
  }
}

function skipTask() {
  state.index += 1;
  renderTask();
}

function replayClip() {
  const task = currentTask();
  if (!task) return;
  elements.video.currentTime = task.start_seconds;
  elements.video.play().catch(() => {});
}

elements.video.addEventListener("loadedmetadata", () => {
  const task = currentTask();
  if (!task) return;
  elements.video.currentTime = task.start_seconds;
});

elements.video.addEventListener("seeked", () => {
  elements.videoLoading.hidden = true;
});

elements.video.addEventListener("timeupdate", () => {
  const task = currentTask();
  if (task && elements.video.currentTime >= task.end_seconds) {
    elements.video.pause();
    elements.video.currentTime = task.end_seconds;
  }
});

elements.video.addEventListener("error", () => {
  elements.videoLoading.textContent = "This local video could not be loaded.";
  elements.videoLoading.hidden = false;
});

document.addEventListener("keydown", (event) => {
  const tag = document.activeElement?.tagName;
  if (tag === "TEXTAREA" || tag === "INPUT") return;
  if (event.code === "Space") {
    event.preventDefault();
    if (elements.video.paused) {
      elements.video.play().catch(() => {});
    } else {
      elements.video.pause();
    }
  }
  if (event.key.toLowerCase() === "r") {
    replayClip();
  }
});

elements.form.addEventListener("submit", saveReview);
elements.skip.addEventListener("click", skipTask);
elements.replay.addEventListener("click", replayClip);
elements.retry.addEventListener("click", loadBatch);
elements.nextBatch.addEventListener("click", loadBatch);

createScoreButtons();
loadBatch();
