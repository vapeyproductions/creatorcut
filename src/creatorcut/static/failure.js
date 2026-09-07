const reasonLabels = {
  missing_start_context: "Missing start context",
  incomplete_ending: "Incomplete ending",
  advertisement_or_sponsor: "Advertisement or sponsor",
  music_or_intro: "Music or intro",
  too_long: "Too long",
  low_energy: "Low energy / boring",
  weak_hook: "Weak hook",
  weak_payoff: "Weak payoff",
  unclear_without_context: "Unclear without context",
  other: "Other",
};

const state = {
  currentCase: null,
  intervals: {},
  boundary: null,
  reviewMode: false,
  reviewCases: [],
  reviewIndex: 0,
};
const workspace = document.querySelector("#workspace");
const complete = document.querySelector("#complete");
const form = document.querySelector("#diagnosis-form");
const error = document.querySelector("#error");
const saveButton = document.querySelector("#save");
const startHandle = document.querySelector("#start-handle");
const endHandle = document.querySelector("#end-handle");

function roundTenth(value) {
  return Math.round((value + Number.EPSILON) * 10) / 10;
}

function formatTime(seconds) {
  const rounded = Math.max(0, roundTenth(seconds));
  const minutes = Math.floor(rounded / 60);
  const remainder = (rounded - minutes * 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${remainder}`;
}

function formatDelta(seconds) {
  const rounded = roundTenth(seconds);
  const prefix = rounded > 0 ? "+" : "";
  return `${prefix}${rounded.toFixed(1)}s`;
}

function setupReasons() {
  const container = document.querySelector("#reasons");
  for (const [value, label] of Object.entries(reasonLabels)) {
    const option = document.createElement("label");
    option.className = "reason-option";
    option.innerHTML = `<input type="checkbox" name="reason" value="${value}" /> ${label}`;
    container.append(option);
  }
}

function configureVideo(elementId, videoId, start, end) {
  const video = document.querySelector(`#${elementId}`);
  state.intervals[elementId] = { start, end };
  video.src = `/api/video/${encodeURIComponent(videoId)}#t=${start},${end}`;
  video.onloadedmetadata = () => {
    if (video.currentTime < start || video.currentTime > end) video.currentTime = start;
  };
  video.ontimeupdate = () => {
    const interval = state.intervals[elementId];
    if (!interval) return;
    if (elementId === "model-video") {
      document.querySelector("#preview-position").textContent =
        `${formatTime(video.currentTime)} / ${formatTime(interval.end)}`;
    }
    if (video.currentTime >= interval.end) video.pause();
  };
}

function renderScores(containerId, clip) {
  const scores = { quality: clip.quality_score, ...clip.component_scores };
  document.querySelector(`#${containerId}`).innerHTML = Object.entries(scores)
    .map(([label, value]) => `<span class="score">${label} ${Number(value).toFixed(2)}</span>`)
    .join("");
}

function renderClip(prefix, clip, videoId) {
  configureVideo(`${prefix}-video`, videoId, clip.start_seconds, clip.end_seconds);
  document.querySelector(`#${prefix}-meta`).textContent =
    `${formatTime(clip.start_seconds)}–${formatTime(clip.end_seconds)} · ${clip.duration_seconds.toFixed(1)} seconds`;
  document.querySelector(`#${prefix}-transcript`).textContent = clip.transcript_text;
  renderScores(`${prefix}-scores`, clip);
}

function resetForm() {
  form.reset();
  state.boundary = null;
  error.textContent = "";
  saveButton.disabled = false;
}

function prefillExistingReview(review) {
  if (!review) return;
  for (const reason of review.reasons || []) {
    const input = document.querySelector(`input[name="reason"][value="${reason}"]`);
    if (input) input.checked = true;
  }
  const boundary = document.querySelector(
    `input[name="boundary-fixable"][value="${String(review.boundary_fixable)}"]`,
  );
  if (boundary) boundary.checked = true;
  if (review.preferred_clip) {
    const preferred = document.querySelector(
      `input[name="preferred-clip"][value="${review.preferred_clip}"]`,
    );
    if (preferred) preferred.checked = true;
  }
  document.querySelector("#notes").value = review.notes || "";
}

function syncBoundaryEditor() {
  const boundary = state.boundary;
  if (!boundary) return;
  startHandle.value = boundary.start.toFixed(1);
  endHandle.value = boundary.end.toFixed(1);
  const width = boundary.windowEnd - boundary.windowStart;
  const leftPercent = ((boundary.start - boundary.windowStart) / width) * 100;
  const rightPercent = ((boundary.end - boundary.windowStart) / width) * 100;
  const selection = document.querySelector("#timeline-selection");
  selection.style.left = `${leftPercent}%`;
  selection.style.width = `${rightPercent - leftPercent}%`;

  const startDelta = roundTenth(boundary.start - boundary.originalStart);
  const endDelta = roundTenth(boundary.end - boundary.originalEnd);
  document.querySelector("#adjusted-start-time").textContent = formatTime(boundary.start);
  document.querySelector("#adjusted-end-time").textContent = formatTime(boundary.end);
  document.querySelector("#adjusted-duration").textContent =
    `${roundTenth(boundary.end - boundary.start).toFixed(1)}s`;
  document.querySelector("#start-delta").textContent = `${formatDelta(startDelta)} from original`;
  document.querySelector("#end-delta").textContent = `${formatDelta(endDelta)} from original`;
  document.querySelector("#start-adjustment").value =
    boundary.startTouched ? startDelta.toFixed(1) : "";
  document.querySelector("#end-adjustment").value =
    boundary.endTouched ? endDelta.toFixed(1) : "";
  state.intervals["model-video"] = { start: boundary.start, end: boundary.end };
}

function configureBoundaryEditor(clip, review) {
  const originalStart = Number(clip.start_seconds);
  const originalEnd = Number(clip.end_seconds);
  const startAdjustment = Number.isFinite(review?.start_adjustment_seconds)
    ? Number(review.start_adjustment_seconds)
    : 0;
  const endAdjustment = Number.isFinite(review?.end_adjustment_seconds)
    ? Number(review.end_adjustment_seconds)
    : 0;
  state.boundary = {
    originalStart,
    originalEnd,
    windowStart: Math.max(0, originalStart - 30),
    windowEnd: originalEnd + 30,
    start: originalStart + startAdjustment,
    end: originalEnd + endAdjustment,
    startTouched: Number.isFinite(review?.start_adjustment_seconds),
    endTouched: Number.isFinite(review?.end_adjustment_seconds),
  };
  for (const handle of [startHandle, endHandle]) {
    handle.min = state.boundary.windowStart.toFixed(1);
    handle.max = state.boundary.windowEnd.toFixed(1);
    handle.step = "0.1";
  }
  document.querySelector("#window-start").textContent = formatTime(state.boundary.windowStart);
  document.querySelector("#window-end").textContent = formatTime(state.boundary.windowEnd);
  document.querySelector("#preview-position").textContent = "Ready to preview";
  syncBoundaryEditor();
}

function setBoundary(which, requestedValue, { touched = true, seek = false } = {}) {
  const boundary = state.boundary;
  if (!boundary || !Number.isFinite(requestedValue)) return;
  if (which === "start") {
    boundary.start = Math.min(
      Math.max(roundTenth(requestedValue), boundary.windowStart),
      roundTenth(boundary.end - 0.1),
    );
    boundary.startTouched = touched;
  } else {
    boundary.end = Math.max(
      Math.min(roundTenth(requestedValue), boundary.windowEnd),
      roundTenth(boundary.start + 0.1),
    );
    boundary.endTouched = touched;
  }
  syncBoundaryEditor();
  if (seek) {
    const video = document.querySelector("#model-video");
    video.pause();
    video.currentTime =
      which === "start" ? boundary.start : Math.max(boundary.start, boundary.end - 0.1);
  }
}

function renderCase(value, stats) {
  state.currentCase = value;
  workspace.classList.remove("hidden");
  complete.classList.add("hidden");
  document.querySelector("#video-id").textContent = value.video_id;
  document.querySelector("#regret").textContent = `${value.regret.toFixed(2)} rating-point regret`;
  if (state.reviewMode) {
    document.querySelector("#progress-count").textContent =
      `Editing ${state.reviewIndex + 1} of ${state.reviewCases.length}`;
    document.querySelector("#progress-detail").textContent = "Saved review";
    saveButton.textContent =
      state.reviewIndex + 1 === state.reviewCases.length
        ? "Save changes and finish"
        : "Save changes and continue";
  } else {
    document.querySelector("#progress-count").textContent =
      `${stats.completed} of ${stats.total} diagnosed`;
    document.querySelector("#progress-detail").textContent = `${stats.remaining} remaining`;
    saveButton.textContent = "Save diagnosis and continue";
  }
  resetForm();
  renderClip("model", value.model_selected, value.video_id);
  renderClip("best", value.human_best, value.video_id);
  prefillExistingReview(value.existing_review);
  configureBoundaryEditor(value.model_selected, value.existing_review);
}

function showComplete(stats) {
  state.currentCase = null;
  workspace.classList.add("hidden");
  complete.classList.remove("hidden");
  document.querySelector("#progress-count").textContent =
    `${stats.completed} of ${stats.total} diagnosed`;
  document.querySelector("#progress-detail").textContent = "Complete";
}

async function loadNext() {
  state.reviewMode = false;
  const response = await fetch("/api/cases?limit=1", { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Unable to load failure cases");
  if (!payload.cases.length) {
    showComplete(payload.stats);
    return;
  }
  renderCase(payload.cases[0], payload.stats);
}

async function editCompletedReviews() {
  const response = await fetch("/api/cases?limit=20&include_completed=true", {
    cache: "no-store",
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Unable to load saved reviews");
  state.reviewMode = true;
  state.reviewCases = payload.cases.filter((value) => value.existing_review);
  state.reviewIndex = 0;
  if (!state.reviewCases.length) {
    throw new Error("No saved reviews are available to edit.");
  }
  renderCase(state.reviewCases[0], payload.stats);
}

function optionalNumber(selector) {
  const value = document.querySelector(selector).value;
  return value === "" ? null : Number(value);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.textContent = "";
  const reasons = [...document.querySelectorAll('input[name="reason"]:checked')]
    .map((input) => input.value);
  const boundary = document.querySelector('input[name="boundary-fixable"]:checked');
  const preferred = document.querySelector('input[name="preferred-clip"]:checked');
  if (!reasons.length) {
    error.textContent = "Select at least one reason.";
    return;
  }
  if (!boundary) {
    error.textContent = "Choose whether a boundary adjustment could rescue the clip.";
    return;
  }
  if (!preferred) {
    error.textContent = "Choose which clip you would actually use.";
    return;
  }
  const payload = {
    reasons,
    preferred_clip: preferred.value,
    boundary_fixable: boundary.value === "true",
    start_adjustment_seconds: optionalNumber("#start-adjustment"),
    end_adjustment_seconds: optionalNumber("#end-adjustment"),
    notes: document.querySelector("#notes").value,
  };
  saveButton.disabled = true;
  try {
    const response = await fetch(
      `/api/failure-reviews/${encodeURIComponent(state.currentCase.analysis_id)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
    );
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to save diagnosis");
    if (state.reviewMode) {
      state.reviewIndex += 1;
      if (state.reviewIndex < state.reviewCases.length) {
        renderCase(state.reviewCases[state.reviewIndex], {
          total: state.reviewCases.length,
          completed: state.reviewCases.length,
          remaining: 0,
        });
      } else {
        state.reviewMode = false;
        showComplete({
          total: state.reviewCases.length,
          completed: state.reviewCases.length,
        });
      }
    } else {
      await loadNext();
    }
  } catch (requestError) {
    error.textContent = requestError.message;
    saveButton.disabled = false;
  }
});

document.querySelectorAll(".replay").forEach((button) => {
  button.addEventListener("click", () => {
    const video = document.querySelector(`#${button.dataset.video}`);
    video.currentTime = state.intervals[button.dataset.video].start;
    video.play();
  });
});

for (const [handle, which] of [[startHandle, "start"], [endHandle, "end"]]) {
  handle.addEventListener("input", () => setBoundary(which, Number(handle.value)));
  handle.addEventListener("change", () =>
    setBoundary(which, Number(handle.value), { seek: true }),
  );
}

for (const [selector, which] of [["#start-adjustment", "start"], ["#end-adjustment", "end"]]) {
  document.querySelector(selector).addEventListener("change", (event) => {
    const boundary = state.boundary;
    if (!boundary) return;
    const original = which === "start" ? boundary.originalStart : boundary.originalEnd;
    if (event.target.value === "") {
      setBoundary(which, original, { touched: false, seek: true });
      return;
    }
    setBoundary(which, original + Number(event.target.value), { seek: true });
  });
}

document.querySelectorAll("[data-nudge]").forEach((button) => {
  button.addEventListener("click", () => {
    const which = button.dataset.nudge;
    const current = which === "start" ? state.boundary?.start : state.boundary?.end;
    setBoundary(which, current + Number(button.dataset.delta), { seek: true });
  });
});

document.querySelector("#preview-adjusted").addEventListener("click", () => {
  if (!state.boundary) return;
  const video = document.querySelector("#model-video");
  video.currentTime = state.boundary.start;
  video.play();
});

document.querySelector("#reset-boundaries").addEventListener("click", () => {
  if (!state.currentCase) return;
  configureBoundaryEditor(state.currentCase.model_selected, null);
  const video = document.querySelector("#model-video");
  video.pause();
  video.currentTime = state.boundary.start;
});

document.querySelector("#edit-completed").addEventListener("click", () => {
  editCompletedReviews().catch((loadError) => {
    error.textContent = loadError.message;
    workspace.classList.remove("hidden");
  });
});

setupReasons();
loadNext().catch((loadError) => {
  error.textContent = loadError.message;
  workspace.classList.remove("hidden");
});
