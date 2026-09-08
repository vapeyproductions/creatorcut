const state = {
  creatorId: localStorage.getItem("creatorcut_creator_id"),
  video: null,
  pollTimer: null,
};

const STATUS_COPY = {
  queued: [5, "Waiting for the local processing worker."],
  validating: [10, "Checking the video and audio streams."],
  transcribing: [35, "Transcribing speech and locating sentence boundaries."],
  generating_candidates: [60, "Generating possible short clips."],
  ranking_candidates: [80, "Scoring candidates with the frozen ranker."],
  ready: [100, "Three recommendations are ready."],
  failed: [100, "Processing stopped."],
};

const elements = {
  uploadSection: document.querySelector("#upload-section"),
  uploadForm: document.querySelector("#upload-form"),
  uploadButton: document.querySelector("#upload-button"),
  uploadError: document.querySelector("#upload-error"),
  creatorName: document.querySelector("#creator-name"),
  processingSection: document.querySelector("#processing-section"),
  processingFile: document.querySelector("#processing-file"),
  processingMessage: document.querySelector("#processing-message"),
  processingProgress: document.querySelector("#processing-progress"),
  resultsSection: document.querySelector("#results-section"),
  clips: document.querySelector("#clips"),
  personalizationStatus: document.querySelector("#personalization-status"),
  newVideo: document.querySelector("#new-video-button"),
  clipTemplate: document.querySelector("#clip-template"),
  recentSection: document.querySelector("#recent-section"),
  recentVideos: document.querySelector("#recent-videos"),
};

const savedName = localStorage.getItem("creatorcut_creator_name");
if (savedName) elements.creatorName.value = savedName;

function showError(element, message) {
  element.textContent = message;
  element.hidden = !message;
}

async function request(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status}).`);
  return value;
}

function setView(name) {
  elements.uploadSection.hidden = name !== "upload";
  elements.processingSection.hidden = name !== "processing";
  elements.resultsSection.hidden = name !== "results";
}

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError(elements.uploadError, "");
  elements.uploadButton.disabled = true;
  elements.uploadButton.textContent = "UPLOADING…";
  try {
    const form = new FormData(elements.uploadForm);
    if (state.creatorId) form.append("creator_id", state.creatorId);
    const payload = await request("/api/uploads", { method: "POST", body: form });
    state.creatorId = payload.creator.id;
    state.video = payload.video;
    localStorage.setItem("creatorcut_creator_id", payload.creator.id);
    localStorage.setItem("creatorcut_creator_name", payload.creator.display_name);
    localStorage.setItem("creatorcut_last_video_id", payload.video.id);
    elements.processingFile.textContent = payload.video.original_filename;
    setView("processing");
    updateProcessing(payload.video);
    schedulePoll();
  } catch (error) {
    showError(elements.uploadError, error.message || "The upload failed.");
  } finally {
    elements.uploadButton.disabled = false;
    elements.uploadButton.textContent = "PROCESS VIDEO";
  }
});

function schedulePoll() {
  window.clearTimeout(state.pollTimer);
  state.pollTimer = window.setTimeout(pollVideo, 1600);
}

async function pollVideo() {
  if (!state.video) return;
  try {
    const video = await request(`/api/videos/${encodeURIComponent(state.video.id)}`);
    state.video = video;
    updateProcessing(video);
    if (video.status === "ready") {
      renderResults(video);
      return;
    }
    if (video.status === "failed") {
      elements.processingMessage.textContent = video.error_message || "The video could not be processed.";
      return;
    }
    schedulePoll();
  } catch (error) {
    elements.processingMessage.textContent = `${error.message} Retrying…`;
    schedulePoll();
  }
}

function updateProcessing(video) {
  const [progress, copy] = STATUS_COPY[video.status] || [5, "Working."];
  elements.processingProgress.value = progress;
  elements.processingMessage.textContent = copy;
}

function formatTime(seconds) {
  const tenths = Math.round(Number(seconds) * 10);
  const minutes = Math.floor(tenths / 600);
  const wholeSeconds = Math.floor((tenths % 600) / 10);
  return `${minutes}:${String(wholeSeconds).padStart(2, "0")}.${tenths % 10}`;
}

function renderResults(video) {
  setView("results");
  localStorage.setItem("creatorcut_last_video_id", video.id);
  elements.clips.replaceChildren();
  const summary = video.creator_summary;
  elements.personalizationStatus.textContent = summary.personalization_active
    ? `PERSONALIZATION ON — using ${summary.decision_count} prior clip decisions for this creator.`
    : `PERSONALIZATION WARM-UP — ${summary.decision_count}/3 decisions recorded. Global ranking is currently primary.`;

  for (const clip of video.clips) {
    const fragment = elements.clipTemplate.content.cloneNode(true);
    const article = fragment.querySelector(".clip-result");
    article.dataset.clipId = clip.id;
    article.querySelector(".clip-rank strong").textContent = clip.rank;
    article.querySelector(".clip-time").textContent =
      `${formatTime(clip.start_seconds)} — ${formatTime(clip.end_seconds)} / ${formatTime(clip.duration_seconds)}`;
    article.querySelector(".clip-explanation").textContent = clip.explanation;
    article.querySelector(".clip-transcript").textContent = `“${clip.transcript_text}”`;

    const videoElement = article.querySelector(".clip-video");
    videoElement.src = `/api/videos/${encodeURIComponent(video.id)}/source`;
    videoElement.dataset.start = clip.start_seconds;
    videoElement.dataset.end = clip.end_seconds;
    videoElement.addEventListener("loadedmetadata", () => {
      videoElement.currentTime = clip.start_seconds;
    });
    videoElement.addEventListener("timeupdate", () => {
      if (videoElement.currentTime >= Number(videoElement.dataset.end)) {
        videoElement.pause();
        videoElement.currentTime = Number(videoElement.dataset.end);
      }
    });

    const startInput = article.querySelector(".start-input");
    const endInput = article.querySelector(".end-input");
    startInput.value = Number(clip.start_seconds).toFixed(1);
    endInput.value = Number(clip.end_seconds).toFixed(1);
    startInput.min = Math.max(0, clip.start_seconds - 15).toFixed(1);
    startInput.max = (clip.start_seconds + 15).toFixed(1);
    endInput.min = Math.max(0, clip.end_seconds - 15).toFixed(1);
    endInput.max = Math.min(video.duration_seconds, clip.end_seconds + 15).toFixed(1);

    const refreshSelection = () => {
      videoElement.dataset.start = startInput.value;
      videoElement.dataset.end = endInput.value;
      article.querySelector(".clip-time").textContent =
        `${formatTime(startInput.value)} — ${formatTime(endInput.value)} / current selection`;
    };
    startInput.addEventListener("change", refreshSelection);
    endInput.addEventListener("change", refreshSelection);
    article.querySelector(".preview-button").addEventListener("click", () => {
      refreshSelection();
      videoElement.currentTime = Number(startInput.value);
      videoElement.play().catch(() => {});
    });
    article.querySelector(".download-button").addEventListener("click", (event) =>
      downloadClip(article, clip, event.currentTarget),
    );
    article.querySelector(".reject-button").addEventListener("click", (event) =>
      rejectClip(article, clip, event.currentTarget),
    );
    article.querySelector(".performance-form").addEventListener("submit", (event) =>
      savePerformance(article, clip, event),
    );
    elements.clips.append(fragment);
  }
}

async function downloadClip(article, clip, button) {
  const start = Number(article.querySelector(".start-input").value);
  const end = Number(article.querySelector(".end-input").value);
  const message = article.querySelector(".clip-message");
  button.disabled = true;
  button.textContent = "PREPARING DOWNLOAD…";
  message.textContent = "Frame-accurate export is being created.";
  try {
    const result = await request(`/api/clips/${encodeURIComponent(clip.id)}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start_seconds: start, end_seconds: end }),
    });
    const anchor = document.createElement("a");
    anchor.href = result.download_url;
    anchor.download = "";
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    message.textContent = result.edited
      ? "Edited clip downloaded. Timestamp changes were saved."
      : "Clip downloaded. Your selection was saved.";
  } catch (error) {
    message.textContent = error.message || "The clip could not be exported.";
  } finally {
    button.disabled = false;
    button.textContent = "DOWNLOAD CLIP";
  }
}

async function rejectClip(article, clip, button) {
  const message = article.querySelector(".clip-message");
  button.disabled = true;
  try {
    await request(`/api/clips/${encodeURIComponent(clip.id)}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "reject" }),
    });
    article.classList.add("rejected");
    article.querySelector(".download-button").disabled = true;
    message.textContent = "Rejected. This decision was saved.";
  } catch (error) {
    button.disabled = false;
    message.textContent = error.message || "The decision could not be saved.";
  }
}

function optionalInteger(form, name) {
  const value = form.elements[name].value;
  return value === "" ? null : Number.parseInt(value, 10);
}

async function savePerformance(article, clip, event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button[type='submit']");
  const message = article.querySelector(".clip-message");
  button.disabled = true;
  const averageValue = form.elements.average_view_percentage.value;
  const payload = {
    platform: form.elements.platform.value,
    views: optionalInteger(form, "views"),
    likes: optionalInteger(form, "likes"),
    comments: optionalInteger(form, "comments"),
    shares: optionalInteger(form, "shares"),
    average_view_percentage: averageValue === "" ? null : Number(averageValue),
  };
  try {
    await request(`/api/clips/${encodeURIComponent(clip.id)}/performance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    message.textContent = "Performance saved separately from your clip decision.";
    form.reset();
  } catch (error) {
    message.textContent = error.message || "Performance could not be saved.";
  } finally {
    button.disabled = false;
  }
}

elements.newVideo.addEventListener("click", () => {
  window.clearTimeout(state.pollTimer);
  state.video = null;
  elements.uploadForm.reset();
  elements.creatorName.value = localStorage.getItem("creatorcut_creator_name") || "My channel";
  setView("upload");
  loadRecentVideos();
});

async function openRecentVideo(videoId) {
  try {
    const video = await request(`/api/videos/${encodeURIComponent(videoId)}`);
    state.video = video;
    elements.processingFile.textContent = video.original_filename;
    if (video.status === "ready") {
      renderResults(video);
    } else {
      setView("processing");
      updateProcessing(video);
      if (video.status !== "failed") schedulePoll();
    }
  } catch (error) {
    showError(elements.uploadError, error.message || "The saved run could not be opened.");
  }
}

async function loadRecentVideos() {
  if (!state.creatorId) return;
  try {
    const payload = await request(
      `/api/creators/${encodeURIComponent(state.creatorId)}/videos`,
    );
    elements.recentVideos.replaceChildren();
    for (const video of payload.videos) {
      const item = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = video.original_filename;
      const status = document.createElement("span");
      status.textContent = video.status.replaceAll("_", " ").toUpperCase();
      const button = document.createElement("button");
      button.type = "button";
      button.className = "plain-button";
      button.textContent = "OPEN";
      button.addEventListener("click", () => openRecentVideo(video.id));
      item.append(name, status, button);
      elements.recentVideos.append(item);
    }
    elements.recentSection.hidden = payload.videos.length === 0;
  } catch (_error) {
    elements.recentSection.hidden = true;
  }
}

loadRecentVideos();
