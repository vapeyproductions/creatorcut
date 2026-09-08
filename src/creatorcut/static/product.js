const state = {
  creatorId: localStorage.getItem("creatorcut_creator_id"),
  video: null,
  pollTimer: null,
  customPreviewEnd: null,
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
  modelReportSection: document.querySelector("#model-report-section"),
  modelReportButton: document.querySelector("#model-report-button"),
  closeModelReport: document.querySelector("#close-model-report"),
  modelReportIntro: document.querySelector("#model-report-intro"),
  modelReportError: document.querySelector("#model-report-error"),
  reportVideoCount: document.querySelector("#report-video-count"),
  reportPresentedCount: document.querySelector("#report-presented-count"),
  reportSelectedCount: document.querySelector("#report-selected-count"),
  reportEditCount: document.querySelector("#report-edit-count"),
  reportCustomCount: document.querySelector("#report-custom-count"),
  reportAnalyticsCount: document.querySelector("#report-analytics-count"),
  adaptationRows: document.querySelector("#adaptation-rows"),
  operationsList: document.querySelector("#operations-list"),
  lineageBody: document.querySelector("#lineage-body"),
  adjustmentList: document.querySelector("#adjustment-list"),
  eventBody: document.querySelector("#event-body"),
  clips: document.querySelector("#clips"),
  resultsTitle: document.querySelector("#results-title"),
  personalizationStatus: document.querySelector("#personalization-status"),
  semanticTrends: document.querySelector("#semantic-trends"),
  newVideo: document.querySelector("#new-video-button"),
  clipTemplate: document.querySelector("#clip-template"),
  recentSection: document.querySelector("#recent-section"),
  recentVideos: document.querySelector("#recent-videos"),
  sourceAnalyticsForm: document.querySelector("#source-analytics-form"),
  sourceAnalyticsStatus: document.querySelector("#source-analytics-status"),
  sourceAnalyticsMessage: document.querySelector("#source-analytics-message"),
  customSourceVideo: document.querySelector("#custom-source-video"),
  customClipForm: document.querySelector("#custom-clip-form"),
  customClipMessage: document.querySelector("#custom-clip-message"),
  useCurrentStart: document.querySelector("#use-current-start"),
  useCurrentEnd: document.querySelector("#use-current-end"),
  previewCustom: document.querySelector("#preview-custom"),
  finishReview: document.querySelector("#finish-review"),
  finishReviewMessage: document.querySelector("#finish-review-message"),
};

const savedName = localStorage.getItem("creatorcut_creator_name");
if (savedName) elements.creatorName.value = savedName;
elements.modelReportButton.disabled = !state.creatorId;

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
  elements.modelReportSection.hidden = name !== "model-report";
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
    elements.modelReportButton.disabled = false;
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

function formatAdjustment(value) {
  const number = Number(value || 0);
  return `${number >= 0 ? "+" : ""}${number.toFixed(3)}`;
}

function renderSourceAnalytics(summary) {
  if (!summary) {
    elements.sourceAnalyticsStatus.textContent =
      "No source report imported. A report with timestamped audience retention can add a bounded signal.";
    return;
  }
  const views = summary.totals.engaged_views ?? summary.totals.views;
  const viewCopy = views === undefined ? "unknown exposure" : `${views.toLocaleString()} views`;
  elements.sourceAnalyticsStatus.textContent =
    `${summary.original_filename}: ${viewCopy}, ${summary.retention_point_count} retention points. ${summary.learning_status}`;
}

function renderResults(video) {
  setView("results");
  localStorage.setItem("creatorcut_last_video_id", video.id);
  elements.clips.replaceChildren();
  const customClipCount = video.clips.filter((clip) => clip.origin === "creator").length;
  elements.resultsTitle.textContent = customClipCount
    ? `Three ranked clips + ${customClipCount} custom clip${customClipCount === 1 ? "" : "s"}`
    : "Three clips, ranked";
  elements.customSourceVideo.src = `/api/videos/${encodeURIComponent(video.id)}/source`;
  const customStart = elements.customClipForm.elements.start_seconds;
  const customEnd = elements.customClipForm.elements.end_seconds;
  customStart.max = Math.max(0, video.duration_seconds - 5).toFixed(1);
  customEnd.max = Number(video.duration_seconds).toFixed(1);
  if (!customStart.value || Number(customStart.value) >= video.duration_seconds) {
    customStart.value = "0.0";
  }
  if (!customEnd.value || Number(customEnd.value) > video.duration_seconds) {
    customEnd.value = Math.min(30, video.duration_seconds).toFixed(1);
  }
  const summary = video.creator_summary;
  const editorial = summary.personalization_active
    ? `Editorial preference: active from ${summary.decision_count} prior clip decisions.`
    : `Editorial preference: ${summary.decision_count}/3 decisions recorded.`;
  const outcomes = summary.performance_personalization_active
    ? `Audience performance: active from ${summary.performance_eligible_clip_count} comparable Shorts.`
    : `Audience performance: ${summary.performance_eligible_clip_count}/${summary.performance_minimum_clip_count} comparable Shorts eligible.`;
  const semantics = summary.semantic_performance_active
    ? `Semantic performance: active from ${summary.semantic_performance_example_count} transcript embeddings.`
    : `Semantic performance: ${summary.semantic_performance_example_count}/${summary.performance_minimum_clip_count} eligible clips have embeddings.`;
  elements.personalizationStatus.textContent = `${editorial}\n${outcomes}\n${semantics}`;
  renderSemanticTrends(summary);
  renderSourceAnalytics(video.source_analytics);

  for (const clip of video.clips) {
    const fragment = elements.clipTemplate.content.cloneNode(true);
    const article = fragment.querySelector(".clip-result");
    article.dataset.clipId = clip.id;
    const rankLabel = article.querySelector(".clip-rank");
    const rankValue = rankLabel.querySelector("strong");
    if (clip.origin === "creator") {
      article.classList.add("creator-clip");
      rankLabel.childNodes[0].nodeValue = "CUSTOM CLIP ";
      rankValue.textContent = "";
    } else {
      rankValue.textContent = clip.rank;
    }
    article.querySelector(".clip-time").textContent =
      `${formatTime(clip.start_seconds)} — ${formatTime(clip.end_seconds)} / ${formatTime(clip.duration_seconds)}`;
    article.querySelector(".clip-explanation").textContent = clip.explanation;
    article.querySelector(".clip-transcript").textContent = `“${clip.transcript_text}”`;
    article.querySelector(".global-score").textContent = Number(clip.global_score).toFixed(3);
    article.querySelector(".global-rank").textContent =
      clip.origin === "creator"
        ? "Creator supplied"
        : clip.global_rank
          ? `#${clip.global_rank} of all candidates`
          : "Not recorded";
    article.querySelector(".model-version").textContent =
      clip.ranking_model_version || "Legacy local run";
    article.querySelector(".editorial-adjustment").textContent = formatAdjustment(
      clip.editorial_adjustment,
    );
    article.querySelector(".performance-adjustment").textContent = formatAdjustment(
      clip.performance_adjustment,
    );
    article.querySelector(".semantic-adjustment").textContent = formatAdjustment(
      clip.semantic_performance_adjustment,
    );
    article.querySelector(".retention-adjustment").textContent = formatAdjustment(
      clip.source_retention_adjustment,
    );
    const publishability = clip.publishability || {};
    const publishabilityReasons = (publishability.reasons || []).map(
      (reason) => reason.code.replaceAll("_", " "),
    );
    article.querySelector(".publishability-check").textContent = publishability.rule_version
      ? `${publishability.eligible === false ? "BLOCK" : "PASS"} · ` +
        `${Math.round(Number(publishability.score) * 100)}/100` +
        `${publishabilityReasons.length ? ` · ${publishabilityReasons.join(", ")}` : " · no flagged risks"}`
      : "Not recorded for this legacy recommendation";
    article.querySelector(".publishability-adjustment").textContent = formatAdjustment(
      clip.publishability_adjustment,
    );
    const targets = article.querySelector(".target-scores");
    for (const field of ["hook", "completeness", "payoff", "clarity"]) {
      const item = document.createElement("span");
      item.textContent = `${field.toUpperCase()} ${Number(clip.predicted_targets[field]).toFixed(2)}`;
      targets.append(item);
    }

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
    article.querySelector(".clip-analytics-form").addEventListener("submit", (event) =>
      importClipAnalytics(article, clip, event),
    );
    elements.clips.append(fragment);
  }
}

elements.customSourceVideo.addEventListener("timeupdate", () => {
  if (
    state.customPreviewEnd !== null &&
    elements.customSourceVideo.currentTime >= state.customPreviewEnd
  ) {
    elements.customSourceVideo.pause();
    state.customPreviewEnd = null;
  }
});

elements.useCurrentStart.addEventListener("click", () => {
  elements.customClipForm.elements.start_seconds.value =
    elements.customSourceVideo.currentTime.toFixed(1);
});

elements.useCurrentEnd.addEventListener("click", () => {
  elements.customClipForm.elements.end_seconds.value =
    elements.customSourceVideo.currentTime.toFixed(1);
});

elements.previewCustom.addEventListener("click", () => {
  const start = Number(elements.customClipForm.elements.start_seconds.value);
  const end = Number(elements.customClipForm.elements.end_seconds.value);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    elements.customClipMessage.textContent = "Enter a valid start and end first.";
    return;
  }
  state.customPreviewEnd = end;
  elements.customSourceVideo.currentTime = start;
  elements.customSourceVideo.play().catch(() => {});
});

elements.customClipForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.video) return;
  const form = event.currentTarget;
  const button = form.querySelector("button[type='submit']");
  button.disabled = true;
  elements.customClipMessage.textContent = "Scoring and saving your interval…";
  try {
    const payload = await request(
      `/api/videos/${encodeURIComponent(state.video.id)}/custom-clips`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_seconds: Number(form.elements.start_seconds.value),
          end_seconds: Number(form.elements.end_seconds.value),
        }),
      },
    );
    state.video = payload.video;
    renderResults(payload.video);
    elements.customClipMessage.textContent =
      "Custom clip saved as creator preference evidence.";
  } catch (error) {
    elements.customClipMessage.textContent =
      error.message || "The custom clip could not be saved.";
  } finally {
    button.disabled = false;
  }
});

elements.finishReview.addEventListener("click", async () => {
  if (!state.video) return;
  elements.finishReview.disabled = true;
  elements.finishReviewMessage.textContent = "Saving the completed review…";
  try {
    const payload = await request(
      `/api/videos/${encodeURIComponent(state.video.id)}/review-complete`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ complete: true }),
      },
    );
    state.video = payload.video;
    renderResults(payload.video);
    elements.finishReviewMessage.textContent = payload.unselected_count
      ? `${payload.unselected_count} unselected recommendation(s) saved as weak preference evidence.`
      : "This review was already complete; no duplicate events were added.";
  } catch (error) {
    elements.finishReviewMessage.textContent =
      error.message || "The completed review could not be saved.";
  } finally {
    elements.finishReview.disabled = false;
  }
});

function renderSemanticTrends(summary) {
  elements.semanticTrends.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = "AUDIENCE PATTERN SUMMARY";
  elements.semanticTrends.append(heading);
  if (!summary.semantic_performance_active) {
    const copy = document.createElement("p");
    copy.textContent =
      "Trends appear after five sufficiently viewed Shorts have both outcomes and saved transcript embeddings.";
    elements.semanticTrends.append(copy);
    return;
  }
  const insight = document.createElement("p");
  insight.textContent =
    summary.performance_insight_summary ||
    "Performance adaptation is active, but no stable summary is available yet.";
  elements.semanticTrends.append(insight);
  const positive = summary.positive_semantic_trends || [];
  const negative = summary.negative_semantic_trends || [];
  if (!positive.length && !negative.length) {
    const copy = document.createElement("p");
    copy.textContent =
      "Semantic matching is active, but no recurring word or phrase has a stable positive association yet.";
    elements.semanticTrends.append(copy);
    return;
  }
  for (const [label, trends] of [
    ["STRONGER CLIP TERMS", positive],
    ["WEAKER CLIP TERMS", negative],
  ]) {
    if (!trends.length) continue;
    const subheading = document.createElement("h4");
    subheading.textContent = label;
    const list = document.createElement("ul");
    for (const trend of trends) {
      const item = document.createElement("li");
      item.textContent = `${trend.term} (${trend.support} clips)`;
      list.append(item);
    }
    elements.semanticTrends.append(subheading, list);
  }
}

async function downloadClip(article, clip, button) {
  const start = Number(article.querySelector(".start-input").value);
  const end = Number(article.querySelector(".end-input").value);
  const message = article.querySelector(".clip-message");
  const exportFormat = article.querySelector("[name='export_format']").value;
  button.disabled = true;
  button.textContent = "PREPARING DOWNLOAD…";
  message.textContent = "Frame-accurate export is being created.";
  try {
    const result = await request(`/api/clips/${encodeURIComponent(clip.id)}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start_seconds: start,
        end_seconds: end,
        export_format: exportFormat,
      }),
    });
    const anchor = document.createElement("a");
    anchor.href = result.download_url;
    anchor.download = "";
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    const formatCopy = exportFormat === "original" ? "clip" : "vertical clip";
    message.textContent = result.edited
      ? `Edited ${formatCopy} downloaded. Timestamp changes were saved.`
      : `${formatCopy[0].toUpperCase()}${formatCopy.slice(1)} downloaded. Your selection was saved.`;
  } catch (error) {
    message.textContent = error.message || "The clip could not be exported.";
  } finally {
    button.disabled = false;
    button.textContent = "DOWNLOAD CLIP";
  }
}

async function importClipAnalytics(article, clip, event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button[type='submit']");
  const message = article.querySelector(".clip-message");
  button.disabled = true;
  try {
    const payload = await request(`/api/clips/${encodeURIComponent(clip.id)}/analytics`, {
      method: "POST",
      body: new FormData(form),
    });
    const imported = payload.import;
    const views = imported.totals.engaged_views ?? imported.totals.views;
    const viewCopy = views === undefined ? "No view total found." : `${views.toLocaleString()} views found.`;
    message.textContent = `${viewCopy} ${imported.learning_status}`;
    form.reset();
  } catch (error) {
    message.textContent = error.message || "The analytics export could not be imported.";
  } finally {
    button.disabled = false;
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

function appendDefinition(list, term, description) {
  const row = document.createElement("div");
  const name = document.createElement("dt");
  const value = document.createElement("dd");
  name.textContent = term;
  value.textContent = description;
  row.append(name, value);
  list.append(row);
}

function renderAdaptationRow(name, statusText, evidence, cap) {
  const row = document.createElement("div");
  row.className = "adaptation-row";
  const title = document.createElement("strong");
  const status = document.createElement("span");
  const detail = document.createElement("span");
  title.textContent = name;
  status.textContent = statusText;
  detail.textContent = `${evidence} ${cap}`;
  row.append(title, status, detail);
  elements.adaptationRows.append(row);
}

function formatCounts(counts) {
  const entries = Object.entries(counts || {});
  if (entries.length === 0) return "none recorded";
  return entries
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, count]) => `${name.replaceAll("_", " ")}: ${count}`)
    .join(" · ");
}

function formatRecordedAt(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
}

function renderModelReport(report) {
  const feedback = report.feedback;
  const adaptation = report.adaptation;
  const selectionRate = feedback.model_clip_selection_rate;
  elements.modelReportIntro.textContent =
    `${report.creator.display_name} · report generated ${formatRecordedAt(report.generated_at)} · ` +
    `${selectionRate === null ? "selection rate not available" : `${Math.round(selectionRate * 100)}% model-clip selection rate`}`;
  elements.reportVideoCount.textContent = report.serving.video_count;
  elements.reportPresentedCount.textContent = feedback.presented_model_clip_count;
  elements.reportSelectedCount.textContent = feedback.selected_model_clip_count;
  elements.reportEditCount.textContent = feedback.edited_download_count;
  elements.reportCustomCount.textContent = feedback.custom_clip_count;
  elements.reportAnalyticsCount.textContent = feedback.analytics_import_count;

  elements.adaptationRows.replaceChildren();
  renderAdaptationRow(
    "Global ranker",
    "FROZEN",
    "Frozen; promotion requires evaluation on unseen source videos.",
    "No online updates.",
  );
  renderAdaptationRow(
    "Publishability",
    "ACTIVE",
    `${report.publishability.candidate_count} candidates assessed across ` +
      `${report.publishability.assessed_video_count} videos; ` +
      `${report.publishability.blocked_candidate_count} blocked.`,
    "Deterministic penalty capped at −0.35.",
  );
  renderAdaptationRow(
    "Editorial",
    adaptation.personalization_active ? "ACTIVE" : "WAITING",
    `${adaptation.decision_count} explicit decisions; activates at 3.`,
    "Maximum adjustment ±0.35.",
  );
  renderAdaptationRow(
    "Audience",
    adaptation.performance_personalization_active ? "ACTIVE" : "WAITING",
    `${adaptation.performance_eligible_clip_count}/${adaptation.performance_minimum_clip_count} comparable Shorts.`,
    "Combined maximum ±0.25.",
  );
  renderAdaptationRow(
    "Semantics",
    adaptation.semantic_performance_active ? "ACTIVE" : "WAITING",
    `${adaptation.semantic_performance_example_count}/${adaptation.performance_minimum_clip_count} outcome-linked embeddings.`,
    "Maximum component ±0.15.",
  );

  elements.operationsList.replaceChildren();
  appendDefinition(
    elements.operationsList,
    "Video states",
    formatCounts(report.serving.video_status_counts),
  );
  appendDefinition(
    elements.operationsList,
    "Persistent job states",
    formatCounts(report.serving.job_status_counts),
  );
  appendDefinition(
    elements.operationsList,
    "Retry policy",
    "Expiring worker leases, heartbeat renewal, exponential retry delay, 3 attempts maximum.",
  );
  appendDefinition(
    elements.operationsList,
    "Stored outcome records",
    `${feedback.performance_report_count} reports across ${feedback.analytics_import_count} analytics imports.`,
  );
  appendDefinition(
    elements.operationsList,
    "Median boundary correction",
    feedback.median_total_boundary_change_seconds === null
      ? "No edited downloads yet."
      : `${feedback.median_total_boundary_change_seconds.toFixed(1)} total seconds per edited selection.`,
  );

  elements.lineageBody.replaceChildren();
  if (report.lineage.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.textContent = "No ranked recommendations have been stored for this creator yet.";
    row.append(cell);
    elements.lineageBody.append(row);
  } else {
    for (const lineage of report.lineage) {
      const row = document.createElement("tr");
      for (const value of [
        lineage.model_version,
        lineage.video_count,
        lineage.clip_count,
        formatRecordedAt(lineage.last_seen),
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      elements.lineageBody.append(row);
    }
  }

  const adjustmentLabels = [
    ["Publishability gate", "publishability"],
    ["Editorial preference", "editorial"],
    ["Structured performance", "performance"],
    ["Semantic performance", "semantic"],
    ["Source retention", "retention"],
  ];
  elements.adjustmentList.replaceChildren();
  for (const [label, key] of adjustmentLabels) {
    appendDefinition(
      elements.adjustmentList,
      label,
      `mean absolute ${report.adjustments[`${key}_mean_absolute`].toFixed(3)} · ` +
        `maximum absolute ${report.adjustments[`${key}_max_absolute`].toFixed(3)}`,
    );
  }

  elements.eventBody.replaceChildren();
  if (report.recent_events.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = "No feedback events have been recorded yet.";
    row.append(cell);
    elements.eventBody.append(row);
  } else {
    for (const event of report.recent_events) {
      const row = document.createElement("tr");
      const interval =
        event.start_seconds === null || event.end_seconds === null
          ? "—"
          : `${formatTime(event.start_seconds)}–${formatTime(event.end_seconds)}`;
      for (const value of [
        event.event_type.replaceAll("_", " "),
        event.original_filename,
        `${event.origin} #${event.rank}`,
        interval,
        formatRecordedAt(event.created_at),
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      elements.eventBody.append(row);
    }
  }
}

async function loadModelReport() {
  if (!state.creatorId) return;
  showError(elements.modelReportError, "");
  elements.modelReportIntro.textContent = "Loading model lineage and feedback evidence…";
  setView("model-report");
  try {
    const report = await request(
      `/api/creators/${encodeURIComponent(state.creatorId)}/model-report`,
    );
    renderModelReport(report);
  } catch (error) {
    showError(elements.modelReportError, error.message || "The model report could not be loaded.");
  }
}

elements.modelReportButton.addEventListener("click", loadModelReport);
elements.closeModelReport.addEventListener("click", () => {
  if (state.video?.status === "ready") renderResults(state.video);
  else setView("upload");
});

elements.sourceAnalyticsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.video) return;
  const form = event.currentTarget;
  const button = form.querySelector("button[type='submit']");
  button.disabled = true;
  elements.sourceAnalyticsMessage.textContent = "Reading the YouTube report…";
  try {
    const payload = await request(
      `/api/videos/${encodeURIComponent(state.video.id)}/analytics`,
      { method: "POST", body: new FormData(form) },
    );
    state.video.source_analytics = payload.import;
    renderSourceAnalytics(payload.import);
    elements.sourceAnalyticsMessage.textContent = payload.import.learning_status;
    form.reset();
  } catch (error) {
    elements.sourceAnalyticsMessage.textContent =
      error.message || "The source analytics export could not be imported.";
  } finally {
    button.disabled = false;
  }
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
