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

const state = { currentCase: null, intervals: {} };
const workspace = document.querySelector("#workspace");
const complete = document.querySelector("#complete");
const form = document.querySelector("#diagnosis-form");
const error = document.querySelector("#error");
const saveButton = document.querySelector("#save");

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = (seconds % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${remainder}`;
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
  video.addEventListener("loadedmetadata", () => {
    if (video.currentTime < start || video.currentTime > end) video.currentTime = start;
  }, { once: true });
  video.ontimeupdate = () => {
    if (video.currentTime >= end) video.pause();
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
  document.querySelector("#start-adjustment").value = review.start_adjustment_seconds ?? "";
  document.querySelector("#end-adjustment").value = review.end_adjustment_seconds ?? "";
  document.querySelector("#notes").value = review.notes || "";
}

function renderCase(value, stats) {
  state.currentCase = value;
  workspace.classList.remove("hidden");
  complete.classList.add("hidden");
  document.querySelector("#video-id").textContent = value.video_id;
  document.querySelector("#regret").textContent = `${value.regret.toFixed(2)} rating-point regret`;
  document.querySelector("#progress-count").textContent = `${stats.completed} of ${stats.total} diagnosed`;
  document.querySelector("#progress-detail").textContent = `${stats.remaining} remaining`;
  resetForm();
  renderClip("model", value.model_selected, value.video_id);
  renderClip("best", value.human_best, value.video_id);
  prefillExistingReview(value.existing_review);
}

async function loadNext() {
  const response = await fetch("/api/cases?limit=1", { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Unable to load failure cases");
  if (!payload.cases.length) {
    state.currentCase = null;
    workspace.classList.add("hidden");
    complete.classList.remove("hidden");
    document.querySelector("#progress-count").textContent = `${payload.stats.completed} of ${payload.stats.total} diagnosed`;
    document.querySelector("#progress-detail").textContent = "Complete";
    return;
  }
  renderCase(payload.cases[0], payload.stats);
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
    const response = await fetch(`/api/failure-reviews/${encodeURIComponent(state.currentCase.analysis_id)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to save diagnosis");
    await loadNext();
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

setupReasons();
loadNext().catch((loadError) => {
  error.textContent = loadError.message;
  workspace.classList.remove("hidden");
});
