const state = {
  account: null,
  creatorId: null,
  csrfToken: null,
  video: null,
  pollTimer: null,
  customPreviewEnd: null,
  repurposingPacks: new Map(),
};

const STATUS_COPY = {
  queued: [5, "Waiting for the local processing worker."],
  validating: [10, "Checking the video and audio streams."],
  transcribing: [35, "Transcribing speech and locating sentence boundaries."],
  generating_candidates: [60, "Generating possible short clips."],
  ranking_candidates: [80, "Scoring candidates with the frozen ranker."],
  ready: [100, "Your platform recommendations are ready."],
  failed: [100, "Processing stopped."],
};

const PLATFORM_LABELS = {
  youtube: "YouTube Shorts",
  instagram: "Instagram Reels",
  tiktok: "TikTok",
};

const elements = {
  authSection: document.querySelector("#auth-section"),
  loginForm: document.querySelector("#login-form"),
  registerForm: document.querySelector("#register-form"),
  accountLabel: document.querySelector("#account-label"),
  logoutButton: document.querySelector("#logout-button"),
  uploadSection: document.querySelector("#upload-section"),
  uploadForm: document.querySelector("#upload-form"),
  uploadButton: document.querySelector("#upload-button"),
  uploadError: document.querySelector("#upload-error"),
  contributionForm: document.querySelector("#contribution-form"),
  contributionStatus: document.querySelector("#contribution-status"),
  processingSection: document.querySelector("#processing-section"),
  processingFile: document.querySelector("#processing-file"),
  processingMessage: document.querySelector("#processing-message"),
  processingProgress: document.querySelector("#processing-progress"),
  resultsSection: document.querySelector("#results-section"),
  modelReportSection: document.querySelector("#model-report-section"),
  adminLink: document.querySelector("#admin-link"),
  modelReportButton: document.querySelector("#model-report-button"),
  closeModelReport: document.querySelector("#close-model-report"),
  modelReportIntro: document.querySelector("#model-report-intro"),
  modelReportError: document.querySelector("#model-report-error"),
  downloadFeedback: document.querySelector("#download-feedback"),
  reportVideoCount: document.querySelector("#report-video-count"),
  reportPresentedCount: document.querySelector("#report-presented-count"),
  reportSelectedCount: document.querySelector("#report-selected-count"),
  reportEditCount: document.querySelector("#report-edit-count"),
  reportCustomCount: document.querySelector("#report-custom-count"),
  reportAnalyticsCount: document.querySelector("#report-analytics-count"),
  reportPostPackCount: document.querySelector("#report-post-pack-count"),
  reportPostFeedbackCount: document.querySelector("#report-post-feedback-count"),
  adaptationRows: document.querySelector("#adaptation-rows"),
  operationsList: document.querySelector("#operations-list"),
  releaseList: document.querySelector("#release-list"),
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

elements.modelReportButton.disabled = true;

function syncPlatformPlanControls() {
  for (const platform of Object.keys(PLATFORM_LABELS)) {
    const checkbox = elements.uploadForm.elements[`platform_${platform}`];
    const count = elements.uploadForm.elements[`count_${platform}`];
    count.disabled = !checkbox.checked;
    if (!checkbox.checked) count.value = "";
  }
}

for (const platform of Object.keys(PLATFORM_LABELS)) {
  const checkbox = elements.uploadForm.elements[`platform_${platform}`];
  checkbox.addEventListener("change", syncPlatformPlanControls);
}
syncPlatformPlanControls();

function showError(element, message) {
  element.textContent = message;
  element.hidden = !message;
}

async function request(url, options = {}) {
  const requestOptions = { cache: "no-store", ...options };
  const method = (requestOptions.method || "GET").toUpperCase();
  const headers = new Headers(requestOptions.headers || {});
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && state.csrfToken) {
    headers.set("X-CSRF-Token", state.csrfToken);
  }
  requestOptions.headers = headers;
  const response = await fetch(url, requestOptions);
  const value = await response.json();
  if (response.status === 401 && !url.startsWith("/api/auth/")) showSignedOut();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status}).`);
  return value;
}

async function loadConfiguration() {
  try {
    const configuration = await request("/api/config");
    elements.adminLink.hidden = !configuration.admin_dashboard_enabled;
    elements.modelReportButton.hidden = !configuration.admin_dashboard_enabled;
    elements.modelReportButton.disabled = !configuration.admin_dashboard_enabled;
  } catch (_error) {
    elements.adminLink.hidden = true;
    elements.modelReportButton.hidden = true;
    elements.modelReportButton.disabled = true;
  }
}

async function loadContributionSettings() {
  if (!state.creatorId) return;
  try {
    const settings = await request("/api/account/contribution-settings");
    elements.contributionForm.elements.performance_enabled.checked =
      settings.performance_enabled;
    elements.contributionStatus.textContent = settings.updated_at
      ? "Saved setting loaded."
      : "Audience-result sharing is off.";
  } catch (error) {
    elements.contributionStatus.textContent =
      error.message || "Contribution settings could not be loaded.";
  }
}

function setView(name) {
  elements.authSection.hidden = name !== "auth";
  elements.uploadSection.hidden = name !== "upload";
  elements.processingSection.hidden = name !== "processing";
  elements.resultsSection.hidden = name !== "results";
  elements.modelReportSection.hidden = name !== "model-report";
}

function showSignedOut() {
  window.clearTimeout(state.pollTimer);
  state.account = null;
  state.creatorId = null;
  state.csrfToken = null;
  state.video = null;
  state.repurposingPacks.clear();
  elements.accountLabel.hidden = true;
  elements.logoutButton.hidden = true;
  elements.adminLink.hidden = true;
  elements.modelReportButton.hidden = true;
  elements.modelReportButton.disabled = true;
  setView("auth");
}

async function activateSession(payload) {
  state.account = payload.account;
  state.creatorId = payload.account.creator_id;
  state.csrfToken = payload.csrf_token;
  elements.accountLabel.textContent = payload.account.display_name;
  elements.accountLabel.hidden = false;
  elements.logoutButton.hidden = false;
  elements.modelReportButton.hidden = true;
  elements.modelReportButton.disabled = true;
  setView("upload");
  await Promise.all([
    loadConfiguration(),
    loadRecentVideos(),
    loadContributionSettings(),
  ]);
}

async function submitCredentials(form, endpoint) {
  const errorElement = form.querySelector(".error");
  const button = form.querySelector("button[type='submit']");
  showError(errorElement, "");
  button.disabled = true;
  try {
    const fields = new FormData(form);
    const payload = Object.fromEntries(fields.entries());
    const session = await request(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    form.reset();
    await activateSession(session);
  } catch (error) {
    showError(errorElement, error.message || "Account access failed.");
  } finally {
    button.disabled = false;
  }
}

elements.loginForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitCredentials(event.currentTarget, "/api/auth/login");
});

elements.registerForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitCredentials(event.currentTarget, "/api/auth/register");
});

elements.logoutButton.addEventListener("click", async () => {
  elements.logoutButton.disabled = true;
  try {
    await request("/api/auth/logout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ logout: true }),
    });
  } catch (_error) {
    // The local session is cleared even if it already expired server-side.
  } finally {
    elements.logoutButton.disabled = false;
    showSignedOut();
  }
});

elements.contributionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type='submit']");
  button.disabled = true;
  elements.contributionStatus.textContent = "Saving permissions…";
  try {
    const settings = await request("/api/account/contribution-settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        performance_enabled:
          event.currentTarget.elements.performance_enabled.checked,
      }),
    });
    elements.contributionStatus.textContent = settings.performance_enabled
      ? "Audience-result sharing is on."
      : "Audience-result sharing is off.";
  } catch (error) {
    elements.contributionStatus.textContent =
      error.message || "Contribution settings could not be saved.";
  } finally {
    button.disabled = false;
  }
});

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError(elements.uploadError, "");
  elements.uploadButton.disabled = true;
  elements.uploadButton.textContent = "UPLOADING…";
  try {
    const form = new FormData(elements.uploadForm);
    const payload = await request("/api/uploads", { method: "POST", body: form });
    state.video = payload.video;
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
  elements.clips.replaceChildren();
  const customClipCount = video.clips.filter((clip) => clip.origin === "creator").length;
  const modelClipCount = video.clips.length - customClipCount;
  const platformNames = [...new Set(video.clips.map((clip) => clip.platform || "youtube"))];
  elements.resultsTitle.textContent =
    `${modelClipCount} ranked clip${modelClipCount === 1 ? "" : "s"} for ` +
    `${platformNames.length} platform${platformNames.length === 1 ? "" : "s"}` +
    (customClipCount ? ` + ${customClipCount} custom` : "");
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
  if (state.account?.is_admin) {
    const editorial = summary.personalization_active
      ? `Editorial preference: active from ${summary.decision_count} prior clip decisions.`
      : `Editorial preference: ${summary.decision_count}/${summary.editorial_minimum_decision_count} decisions recorded.`;
    const platformLearning = Object.entries(summary.platform_performance || {})
      .map(([platform, performance]) => {
        const status = performance.active ? "active" : "waiting";
        return `${PLATFORM_LABELS[platform]} audience model: ${status} ` +
          `(${performance.eligible_clip_count}/${performance.minimum_clip_count} eligible clips).`;
      })
      .join("\n");
    elements.personalizationStatus.textContent = `${editorial}\n${platformLearning}`;
  } else {
    elements.personalizationStatus.textContent =
      "CreatorCut adapts as you choose clips, adjust boundaries, and add audience results.";
  }
  renderSemanticTrends(summary);
  renderSourceAnalytics(video.source_analytics);

  const platformContainers = new Map();
  for (const platform of platformNames) {
    const section = document.createElement("section");
    section.className = "platform-group";
    const header = document.createElement("header");
    const heading = document.createElement("h3");
    heading.textContent = PLATFORM_LABELS[platform] || platform;
    const detail = document.createElement("p");
    const plan = video.clip_plan?.platforms?.[platform];
    detail.textContent = plan
      ? `${plan.delivered_count} delivered. Suggested range ${plan.suggested_range[0]}–${plan.suggested_range[1]}; ` +
        `${plan.available_non_overlapping_count} distinct usable moments found. ` +
        `${plan.profile.aspect_ratio}, ${plan.profile.generation_range_seconds[0]}–${plan.profile.generation_range_seconds[1]} second generation band.`
      : "Creator-defined clips for this platform.";
    header.append(heading, detail);
    section.append(header);
    elements.clips.append(section);
    platformContainers.set(platform, section);
  }

  for (const clip of video.clips) {
    const fragment = elements.clipTemplate.content.cloneNode(true);
    const article = fragment.querySelector(".clip-result");
    article.querySelector(".model-evidence").hidden = !state.account?.is_admin;
    article.dataset.clipId = clip.id;
    const rankLabel = article.querySelector(".clip-rank");
    const rankValue = rankLabel.querySelector("strong");
    article.querySelector(".clip-platform").textContent =
      (PLATFORM_LABELS[clip.platform] || clip.platform || "YouTube Shorts").toUpperCase();
    if (clip.origin === "creator") {
      article.classList.add("creator-clip");
      rankLabel.childNodes[0].nodeValue = "CUSTOM CLIP ";
      rankValue.textContent = "";
    } else {
      rankValue.textContent = clip.platform_rank || clip.rank;
    }
    article.querySelector(".clip-time").textContent =
      `${formatTime(clip.start_seconds)} — ${formatTime(clip.end_seconds)} / ${formatTime(clip.duration_seconds)}`;
    article.querySelector(".clip-explanation").textContent = clip.explanation;
    article.querySelector(".clip-transcript").textContent = `“${clip.transcript_text}”`;
    if (state.account?.is_admin) {
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
      article.querySelector(".platform-adjustment").textContent = formatAdjustment(
        clip.platform_adjustment,
      );
      article.querySelector(".platform-score").textContent = Number(
        clip.platform_score ?? clip.personalized_score,
      ).toFixed(3);
      const delivery = clip.multimodal_features || {};
      const featureStatus = delivery.status === "unavailable" ? "unavailable" : null;
      article.querySelector(".audio-urgency").textContent = featureStatus ||
        `${Math.round(Number(delivery.audio_urgency_score ?? 0.5) * 100)}/100`;
      article.querySelector(".visual-excitement").textContent = featureStatus ||
        `${Math.round(Number(delivery.visual_excitement_score ?? 0.5) * 100)}/100`;
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
    article.querySelector("[name='export_format']").value = "vertical_captions";
    article.querySelector(".clip-analytics-form [name='platform']").value =
      clip.platform || "youtube";
    article.querySelector(".performance-form [name='platform']").value =
      clip.platform || "youtube";

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
    article.querySelector(".generate-posts-button").addEventListener("click", (event) =>
      generatePlatformPosts(article, clip, event.currentTarget),
    );
    platformContainers.get(clip.platform || "youtube").append(fragment);
  }
}

function buildPlatformPosts(article, clip, pack) {
  const output = article.querySelector(".repurposing-output");
  output.replaceChildren();
  for (const [platform, value] of Object.entries(pack.platforms)) {
    const card = document.createElement("section");
    card.className = "platform-post";
    const heading = document.createElement("h4");
    heading.textContent = value.label;
    const textarea = document.createElement("textarea");
    textarea.rows = 8;
    textarea.value = value.text;
    textarea.dataset.generatedText = value.text;
    textarea.setAttribute("aria-label", `${value.label} post text`);
    const actions = document.createElement("div");
    actions.className = "platform-post-actions";
    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.textContent = "COPY & SAVE CHOICE";
    copyButton.addEventListener("click", () =>
      copyPlatformPost(article, clip, platform, textarea, copyButton),
    );
    const rejectButton = document.createElement("button");
    rejectButton.type = "button";
    rejectButton.className = "plain-button";
    rejectButton.textContent = "REJECT COPY";
    rejectButton.addEventListener("click", () =>
      rejectPlatformPost(article, clip, platform, textarea, rejectButton),
    );
    actions.append(copyButton, rejectButton);
    card.append(heading, textarea, actions);
    output.append(card);
  }
  output.hidden = false;
}

async function generatePlatformPosts(article, clip, button) {
  const note = article.querySelector(".repurposing-note");
  button.disabled = true;
  note.textContent = "Finding transcript topics and preparing platform copy…";
  try {
    const payload = await request(`/api/clips/${encodeURIComponent(clip.id)}/repurpose`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    state.repurposingPacks.set(clip.id, payload.pack);
    buildPlatformPosts(article, clip, payload.pack);
    const aligned = payload.pack.evidence.audience_aligned_topics || [];
    note.textContent = aligned.length
      ? `Personalized using positive audience terms: ${aligned.join(", ")}. ${payload.pack.note}`
      : `No eligible audience-topic boost was applied. ${payload.pack.note}`;
    button.textContent = "REGENERATE FROM TRANSCRIPT";
  } catch (error) {
    note.textContent = error.message || "Platform posts could not be generated.";
  } finally {
    button.disabled = false;
  }
}

async function writeToClipboard(text, textarea) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  textarea.focus();
  textarea.select();
  if (!document.execCommand("copy")) throw new Error("Your browser blocked clipboard access.");
}

async function savePlatformPostFeedback(clip, platform, action, generatedText, finalText) {
  const pack = state.repurposingPacks.get(clip.id);
  if (!pack) throw new Error("Generate the platform posts again before saving feedback.");
  await request(`/api/clips/${encodeURIComponent(clip.id)}/repurpose-feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      pack_id: pack.id,
      platform,
      action,
      generated_text: generatedText,
      final_text: finalText,
    }),
  });
}

async function copyPlatformPost(article, clip, platform, textarea, button) {
  const note = article.querySelector(".repurposing-note");
  const finalText = textarea.value;
  if (!finalText.trim()) {
    note.textContent = "Add some post text before copying it.";
    return;
  }
  button.disabled = true;
  try {
    await writeToClipboard(finalText, textarea);
    const generatedText = textarea.dataset.generatedText;
    const action = finalText === generatedText ? "copied_original" : "copied_edited";
    await savePlatformPostFeedback(
      clip,
      platform,
      action,
      generatedText,
      finalText,
    );
    note.textContent = action === "copied_edited"
      ? "Edited copy placed on the clipboard; the exact edit was saved."
      : "Generated copy placed on the clipboard; the choice was saved.";
  } catch (error) {
    note.textContent = error.message || "The post choice could not be saved.";
  } finally {
    button.disabled = false;
  }
}

async function rejectPlatformPost(article, clip, platform, textarea, button) {
  const note = article.querySelector(".repurposing-note");
  button.disabled = true;
  try {
    await savePlatformPostFeedback(
      clip,
      platform,
      "rejected",
      textarea.dataset.generatedText,
      null,
    );
    note.textContent = `${platform} copy rejected; the decision was saved.`;
  } catch (error) {
    note.textContent = error.message || "The rejection could not be saved.";
    button.disabled = false;
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
          platform: form.elements.platform.value,
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
  heading.textContent = "PLATFORM AUDIENCE PATTERNS";
  elements.semanticTrends.append(heading);
  for (const [platform, performance] of Object.entries(summary.platform_performance || {})) {
    const subheading = document.createElement("h4");
    subheading.textContent = (PLATFORM_LABELS[platform] || platform).toUpperCase();
    elements.semanticTrends.append(subheading);
    if (!performance.semantic_active) {
      const copy = document.createElement("p");
      copy.textContent = state.account?.is_admin
        ? `Waiting for ${performance.minimum_clip_count} sufficiently viewed clips with outcomes and transcript embeddings.`
        : "Insights will appear after enough published clips have audience results.";
      elements.semanticTrends.append(copy);
      continue;
    }
    const insight = document.createElement("p");
    insight.textContent =
      performance.insight_summary ||
      "Performance adaptation is active, but no stable summary is available yet.";
    elements.semanticTrends.append(insight);
    for (const [label, trends] of [
      ["STRONGER TERMS", performance.positive_semantic_trends || []],
      ["WEAKER TERMS", performance.negative_semantic_trends || []],
    ]) {
      if (!trends.length) continue;
      const termHeading = document.createElement("h4");
      termHeading.textContent = label;
      const list = document.createElement("ul");
      for (const trend of trends) {
        const item = document.createElement("li");
        item.textContent = state.account?.is_admin
          ? `${trend.term} (${trend.support} clips)`
          : trend.term;
        list.append(item);
      }
      elements.semanticTrends.append(termHeading, list);
    }
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
    const trackingCopy =
      result.reframing?.mode === "subject_aware_face_tracking"
        ? " Face tracking kept the detected subject in frame."
        : exportFormat !== "original"
          ? " No reliable face was detected, so center-crop fallback was used."
          : "";
    message.textContent = (result.edited
      ? `Edited ${formatCopy} downloaded. Timestamp changes were saved.`
      : `${formatCopy[0].toUpperCase()}${formatCopy.slice(1)} downloaded. Your selection was saved.`) +
      trackingCopy;
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
  const completionValue = form.elements.completion_rate_percentage.value;
  const payload = {
    platform: form.elements.platform.value,
    views: optionalInteger(form, "views"),
    likes: optionalInteger(form, "likes"),
    comments: optionalInteger(form, "comments"),
    shares: optionalInteger(form, "shares"),
    saves: optionalInteger(form, "saves"),
    follows: optionalInteger(form, "follows"),
    average_view_percentage: averageValue === "" ? null : Number(averageValue),
    completion_rate_percentage:
      completionValue === "" ? null : Number(completionValue),
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
  syncPlatformPlanControls();
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
  elements.reportPostPackCount.textContent = feedback.repurposing_pack_count;
  elements.reportPostFeedbackCount.textContent = Object.values(
    feedback.repurposing_feedback_counts || {},
  ).reduce((total, count) => total + count, 0);

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
    `${adaptation.decision_count} explicit decisions; activates at ${adaptation.editorial_minimum_decision_count}.`,
    "Maximum adjustment ±0.35.",
  );
  for (const [platform, performance] of Object.entries(
    adaptation.platform_performance || {},
  )) {
    const label = PLATFORM_LABELS[platform] || platform;
    renderAdaptationRow(
      `${label} audience`,
      performance.active ? "ACTIVE" : "WAITING",
      `${performance.eligible_clip_count}/${performance.minimum_clip_count} comparable published clips.`,
      "Combined maximum ±0.25.",
    );
    renderAdaptationRow(
      `${label} semantics`,
      performance.semantic_active ? "ACTIVE" : "WAITING",
      `${performance.semantic_example_count}/${performance.minimum_clip_count} outcome-linked embeddings.`,
      "Maximum component ±0.15.",
    );
  }

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
    "Repurposing feedback",
    `${feedback.repurposing_pack_count} generated packs · ` +
      `${formatCounts(feedback.repurposing_feedback_counts)}.`,
  );
  appendDefinition(
    elements.operationsList,
    "Median boundary correction",
    feedback.median_total_boundary_change_seconds === null
      ? "No edited downloads yet."
      : `${feedback.median_total_boundary_change_seconds.toFixed(1)} total seconds per edited selection.`,
  );

  const release = report.serving_release || { status: "unavailable" };
  elements.releaseList.replaceChildren();
  appendDefinition(elements.releaseList, "Release", release.release_id || "Not configured");
  appendDefinition(elements.releaseList, "Status", release.status.toUpperCase());
  appendDefinition(
    elements.releaseList,
    "Global ranker",
    release.global_ranker_schema || "Unavailable",
  );
  appendDefinition(
    elements.releaseList,
    "Artifact SHA-256",
    release.global_ranker_sha256 || "Unavailable",
  );
  appendDefinition(
    elements.releaseList,
    "Frozen components",
    Object.entries(release.components || {})
      .map(([name, version]) => `${name}: ${version}`)
      .join(" · ") || "Unavailable",
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
    ["Platform prior", "platform"],
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
        `${PLATFORM_LABELS[event.platform] || event.platform} · ${event.origin} #${event.platform_rank || event.rank}`,
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
      "/api/account/model-report",
    );
    renderModelReport(report);
  } catch (error) {
    showError(elements.modelReportError, error.message || "The model report could not be loaded.");
  }
}

elements.modelReportButton.addEventListener("click", loadModelReport);
elements.downloadFeedback.addEventListener("click", async () => {
  if (!state.creatorId) return;
  elements.downloadFeedback.disabled = true;
  elements.downloadFeedback.textContent = "PREPARING SNAPSHOT…";
  try {
    const response = await fetch(
      "/api/account/feedback-export",
      { cache: "no-store" },
    );
    if (!response.ok) {
      const value = await response.json();
      throw new Error(value.error || "The snapshot could not be created.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "creatorcut-feedback-snapshot.json";
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    elements.modelReportIntro.textContent =
      "Feedback snapshot downloaded with model lineage and an integrity hash.";
  } catch (error) {
    showError(
      elements.modelReportError,
      error.message || "The feedback snapshot could not be downloaded.",
    );
  } finally {
    elements.downloadFeedback.disabled = false;
    elements.downloadFeedback.textContent = "DOWNLOAD ML SNAPSHOT";
  }
});
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
      "/api/account/videos",
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

async function loadSession() {
  try {
    const session = await request("/api/auth/session");
    if (session.authenticated) await activateSession(session);
    else showSignedOut();
  } catch (_error) {
    showSignedOut();
  }
}

loadSession();
