const SCORE_FIELDS = ["hook", "completeness", "payoff", "clarity"];
const MAXIMUM_BOUNDARY_ADJUSTMENT_SECONDS = 30;
const MINIMUM_CLIP_DURATION_SECONDS = 0.1;

const state = {
  tasks: [],
  index: 0,
  scores: { original: {}, edited: {} },
  stats: { total: 0, completed: 0, remaining: 0 },
  savedInBatch: 0,
  boundary: null,
  playbackInterval: null,
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
  enableBoundaryEdit: document.querySelector("#enable-boundary-edit"),
  boundaryEditor: document.querySelector("#boundary-editor"),
  startHandle: document.querySelector("#start-handle"),
  endHandle: document.querySelector("#end-handle"),
  timelineSelection: document.querySelector("#timeline-selection"),
  windowStart: document.querySelector("#window-start"),
  windowEnd: document.querySelector("#window-end"),
  adjustedStart: document.querySelector("#adjusted-start-time"),
  adjustedEnd: document.querySelector("#adjusted-end-time"),
  adjustedDuration: document.querySelector("#adjusted-duration"),
  startDelta: document.querySelector("#start-delta"),
  endDelta: document.querySelector("#end-delta"),
  previewPosition: document.querySelector("#preview-position"),
  previewAdjusted: document.querySelector("#preview-adjusted"),
  resetBoundaries: document.querySelector("#reset-boundaries"),
};

function roundToTenth(value) {
  return Math.round(Number(value) * 10) / 10;
}

function createScoreButtons() {
  document.querySelectorAll(".score-field").forEach((fieldElement) => {
    const scoreSet = fieldElement.dataset.scoreSet;
    const field = fieldElement.dataset.field;
    const group = fieldElement.querySelector(".score-group");
    for (let score = 1; score <= 5; score += 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "score-button";
      button.dataset.scoreSet = scoreSet;
      button.dataset.field = field;
      button.dataset.score = String(score);
      button.setAttribute("role", "radio");
      button.setAttribute("aria-checked", "false");
      button.setAttribute("aria-label", `${scoreSet} ${field} score ${score} out of 5`);
      button.textContent = String(score);
      button.addEventListener("click", () => selectScore(scoreSet, field, score));
      group.append(button);
    }
  });
}

function selectScore(scoreSet, field, score) {
  state.scores[scoreSet][field] = score;
  document
    .querySelectorAll(
      `.score-button[data-score-set="${scoreSet}"][data-field="${field}"]`,
    )
    .forEach((button) => {
      const selected = Number(button.dataset.score) === score;
      button.setAttribute("aria-checked", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
  clearFormError();
}

function clearScoreSet(scoreSet) {
  state.scores[scoreSet] = {};
  document
    .querySelectorAll(`.score-button[data-score-set="${scoreSet}"]`)
    .forEach((button) => {
      button.setAttribute("aria-checked", "false");
      button.tabIndex = 0;
    });
}

function resetForm() {
  clearScoreSet("original");
  clearScoreSet("edited");
  elements.notes.value = "";
  elements.technicalIssue.checked = false;
  elements.enableBoundaryEdit.checked = false;
  elements.boundaryEditor.hidden = true;
  state.boundary = null;
  state.playbackInterval = null;
  clearFormError();
}

function formatTime(seconds) {
  const tenths = Math.max(0, Math.round(Number(seconds) * 10));
  const minutes = Math.floor(tenths / 600);
  const wholeSeconds = Math.floor((tenths % 600) / 10);
  const decimal = tenths % 10;
  return `${String(minutes).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}.${decimal}`;
}

function formatDelta(seconds) {
  const rounded = roundToTenth(seconds);
  if (Math.abs(rounded) < 0.05) return "no change";
  return `${rounded > 0 ? "+" : ""}${rounded.toFixed(1)}s`;
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

function configureBoundary(task, mediaDuration = null, preserveSelection = false) {
  const maximumFromTask = roundToTenth(
    task.end_seconds + MAXIMUM_BOUNDARY_ADJUSTMENT_SECONDS,
  );
  const maximum = Number.isFinite(mediaDuration)
    ? Math.min(maximumFromTask, roundToTenth(mediaDuration))
    : maximumFromTask;
  const minimum = roundToTenth(
    Math.max(0, task.start_seconds - MAXIMUM_BOUNDARY_ADJUSTMENT_SECONDS),
  );
  const previous = preserveSelection ? state.boundary : null;
  const start = roundToTenth(
    Math.max(minimum, Math.min(previous?.start ?? task.start_seconds, maximum)),
  );
  const end = roundToTenth(
    Math.max(
      start + MINIMUM_CLIP_DURATION_SECONDS,
      Math.min(previous?.end ?? task.end_seconds, maximum),
    ),
  );

  state.boundary = {
    annotationId: task.annotation_id,
    originalStart: roundToTenth(task.start_seconds),
    originalEnd: roundToTenth(task.end_seconds),
    start,
    end,
    minimum,
    maximum,
  };
  state.playbackInterval = {
    start: state.boundary.originalStart,
    end: state.boundary.originalEnd,
  };

  for (const handle of [elements.startHandle, elements.endHandle]) {
    handle.min = String(minimum);
    handle.max = String(maximum);
    handle.step = "0.1";
  }
  updateBoundaryUI();
}

function updateBoundaryUI() {
  const boundary = state.boundary;
  if (!boundary) return;
  elements.startHandle.value = String(boundary.start);
  elements.endHandle.value = String(boundary.end);
  elements.windowStart.textContent = formatTime(boundary.minimum);
  elements.windowEnd.textContent = formatTime(boundary.maximum);
  elements.adjustedStart.textContent = formatTime(boundary.start);
  elements.adjustedEnd.textContent = formatTime(boundary.end);
  elements.adjustedDuration.textContent = formatTime(boundary.end - boundary.start);
  elements.startDelta.textContent = `${formatDelta(boundary.start - boundary.originalStart)} from original`;
  elements.endDelta.textContent = `${formatDelta(boundary.end - boundary.originalEnd)} from original`;

  const windowDuration = boundary.maximum - boundary.minimum;
  const left = windowDuration ? ((boundary.start - boundary.minimum) / windowDuration) * 100 : 0;
  const right = windowDuration ? ((boundary.end - boundary.minimum) / windowDuration) * 100 : 100;
  elements.timelineSelection.style.left = `${left}%`;
  elements.timelineSelection.style.width = `${Math.max(0, right - left)}%`;
}

function setBoundary(kind, rawValue, seekAfterChange = false) {
  const boundary = state.boundary;
  if (!boundary) return;
  const value = roundToTenth(rawValue);
  if (kind === "start") {
    boundary.start = Math.max(
      boundary.minimum,
      Math.min(value, roundToTenth(boundary.end - MINIMUM_CLIP_DURATION_SECONDS)),
    );
  } else {
    boundary.end = Math.min(
      boundary.maximum,
      Math.max(value, roundToTenth(boundary.start + MINIMUM_CLIP_DURATION_SECONDS)),
    );
  }
  updateBoundaryUI();
  clearFormError();

  if (seekAfterChange) {
    const seekTarget =
      kind === "start" ? boundary.start : Math.max(boundary.start, boundary.end - 1.5);
    elements.video.currentTime = seekTarget;
    elements.previewPosition.textContent =
      kind === "start" ? "Showing the adjusted opening" : "Showing the adjusted ending";
  }
}

function resetBoundarySelection() {
  const boundary = state.boundary;
  if (!boundary) return;
  boundary.start = boundary.originalStart;
  boundary.end = boundary.originalEnd;
  clearScoreSet("edited");
  state.playbackInterval = { start: boundary.originalStart, end: boundary.originalEnd };
  elements.video.pause();
  elements.video.currentTime = boundary.originalStart;
  elements.previewPosition.textContent = "Boundaries reset";
  updateBoundaryUI();
  clearFormError();
}

function renderTask() {
  const task = currentTask();
  if (!task) {
    showBatchComplete();
    return;
  }

  resetForm();
  configureBoundary(task);
  elements.batchIndex.textContent = String(state.index + 1);
  elements.batchTotal.textContent = String(state.tasks.length);
  elements.videoId.textContent = task.video_id.toUpperCase();
  elements.duration.textContent = formatTime(task.duration_seconds);
  elements.range.textContent = `${formatTime(task.start_seconds)} → ${formatTime(task.end_seconds)}`;
  elements.transcript.textContent = task.transcript_text || "No transcript text is available.";
  elements.videoLoading.textContent = "Loading local video…";
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

function hasBoundaryChange() {
  const boundary = state.boundary;
  return (
    boundary &&
    (Math.abs(boundary.start - boundary.originalStart) >= 0.05 ||
      Math.abs(boundary.end - boundary.originalEnd) >= 0.05)
  );
}

async function saveReview(event) {
  event.preventDefault();
  const missingOriginal = SCORE_FIELDS.filter((field) => !state.scores.original[field]);
  if (missingOriginal.length) {
    showFormError(`Please score the original ${missingOriginal.join(", ")} before saving.`);
    return;
  }

  let boundaryEdit = null;
  if (elements.enableBoundaryEdit.checked) {
    if (!hasBoundaryChange()) {
      showFormError("Change at least one boundary, or turn off the optional boundary edit.");
      return;
    }
    const missingEdited = SCORE_FIELDS.filter((field) => !state.scores.edited[field]);
    if (missingEdited.length) {
      showFormError(`Please score the edited ${missingEdited.join(", ")} before saving.`);
      return;
    }
    boundaryEdit = {
      start_seconds: state.boundary.start,
      end_seconds: state.boundary.end,
      scores: { ...state.scores.edited },
    };
  }

  const task = currentTask();
  const payload = {
    ...state.scores.original,
    technically_exportable: !elements.technicalIssue.checked,
    notes: elements.notes.value,
    boundary_edit: boundaryEdit,
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

function playInterval(start, end, message) {
  state.playbackInterval = { start, end };
  elements.video.currentTime = start;
  elements.previewPosition.textContent = message;
  elements.video.play().catch(() => {});
}

function replayOriginal() {
  const task = currentTask();
  if (!task) return;
  playInterval(task.start_seconds, task.end_seconds, "Playing original interval");
}

function previewAdjusted() {
  const boundary = state.boundary;
  if (!boundary) return;
  playInterval(boundary.start, boundary.end, "Playing edited interval");
}

function toggleBoundaryEditor() {
  const enabled = elements.enableBoundaryEdit.checked;
  elements.boundaryEditor.hidden = !enabled;
  if (enabled) {
    elements.previewPosition.textContent = "Ready to preview";
  } else {
    resetBoundarySelection();
  }
  clearFormError();
}

elements.video.addEventListener("loadedmetadata", () => {
  const task = currentTask();
  if (!task) return;
  const preserveSelection = state.boundary?.annotationId === task.annotation_id;
  configureBoundary(task, elements.video.duration, preserveSelection);
  elements.video.currentTime = task.start_seconds;
});

elements.video.addEventListener("seeked", () => {
  elements.videoLoading.hidden = true;
});

elements.video.addEventListener("timeupdate", () => {
  const interval = state.playbackInterval;
  if (interval && elements.video.currentTime >= interval.end) {
    elements.video.pause();
    elements.video.currentTime = interval.end;
  }
});

elements.video.addEventListener("error", () => {
  elements.videoLoading.textContent = "This local video could not be loaded.";
  elements.videoLoading.hidden = false;
});

elements.enableBoundaryEdit.addEventListener("change", toggleBoundaryEditor);
elements.startHandle.addEventListener("input", (event) => setBoundary("start", event.target.value));
elements.startHandle.addEventListener("change", (event) =>
  setBoundary("start", event.target.value, true),
);
elements.endHandle.addEventListener("input", (event) => setBoundary("end", event.target.value));
elements.endHandle.addEventListener("change", (event) =>
  setBoundary("end", event.target.value, true),
);
document.querySelectorAll("[data-nudge]").forEach((button) => {
  button.addEventListener("click", () => {
    const kind = button.dataset.nudge;
    const nextValue = state.boundary[kind] + Number(button.dataset.delta);
    setBoundary(kind, nextValue, true);
  });
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
    replayOriginal();
  }
});

elements.form.addEventListener("submit", saveReview);
elements.skip.addEventListener("click", skipTask);
elements.replay.addEventListener("click", replayOriginal);
elements.previewAdjusted.addEventListener("click", previewAdjusted);
elements.resetBoundaries.addEventListener("click", resetBoundarySelection);
elements.retry.addEventListener("click", loadBatch);
elements.nextBatch.addEventListener("click", loadBatch);

createScoreButtons();
loadBatch();
