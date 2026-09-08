const elements = {
  refresh: document.querySelector("#refresh"),
  status: document.querySelector("#report-status"),
  error: document.querySelector("#report-error"),
  offlinePairwise: document.querySelector("#offline-pairwise"),
  offlineTopOne: document.querySelector("#offline-top-one"),
  offlineRegret: document.querySelector("#offline-regret"),
  offlineTopThree: document.querySelector("#offline-top-three"),
  releaseDetails: document.querySelector("#release-details"),
  accountCount: document.querySelector("#account-count"),
  videoCount: document.querySelector("#video-count"),
  clipCount: document.querySelector("#clip-count"),
  presentedCount: document.querySelector("#presented-count"),
  selectedCount: document.querySelector("#selected-count"),
  selectionRate: document.querySelector("#selection-rate"),
  editCount: document.querySelector("#edit-count"),
  analyticsCount: document.querySelector("#analytics-count"),
  rankBody: document.querySelector("#rank-body"),
  eventCounts: document.querySelector("#event-counts"),
  sourceTypes: document.querySelector("#source-types"),
  clipOrigins: document.querySelector("#clip-origins"),
  clipPlatforms: document.querySelector("#clip-platforms"),
  sourceDuration: document.querySelector("#source-duration"),
  clipDuration: document.querySelector("#clip-duration"),
  selectedDuration: document.querySelector("#selected-duration"),
  exportFormats: document.querySelector("#export-formats"),
  videoStatuses: document.querySelector("#video-statuses"),
  startEdits: document.querySelector("#start-edits"),
  endEdits: document.querySelector("#end-edits"),
  totalEdits: document.querySelector("#total-edits"),
  analyticsRoles: document.querySelector("#analytics-roles"),
  analyticsPlatforms: document.querySelector("#analytics-platforms"),
  performancePlatforms: document.querySelector("#performance-platforms"),
  analyticsTypes: document.querySelector("#analytics-types"),
  analyticsFiles: document.querySelector("#analytics-files"),
  analyticsMetrics: document.querySelector("#analytics-metrics"),
  performanceMetrics: document.querySelector("#performance-metrics"),
  retentionPoints: document.querySelector("#retention-points"),
  outcomeDistributions: document.querySelector("#outcome-distributions"),
  communityEditorialCreators: document.querySelector("#community-editorial-creators"),
  communityEditorialDecisions: document.querySelector("#community-editorial-decisions"),
  communityPerformanceOptIns: document.querySelector("#community-performance-opt-ins"),
  communityConsentEvents: document.querySelector("#community-consent-events"),
  communityFlowBody: document.querySelector("#community-flow-body"),
  communityPlatformBody: document.querySelector("#community-platform-body"),
  communityPolicy: document.querySelector("#community-policy"),
  communityEditorialPolicy: document.querySelector("#community-editorial-policy"),
  communityPerformancePolicy: document.querySelector("#community-performance-policy"),
  communityEditorialAdjustments: document.querySelector("#community-editorial-adjustments"),
  communityPerformanceAdjustments: document.querySelector("#community-performance-adjustments"),
  communitySemanticAdjustments: document.querySelector("#community-semantic-adjustments"),
  accountBody: document.querySelector("#account-body"),
  jobStatuses: document.querySelector("#job-statuses"),
  jobAttempts: document.querySelector("#job-attempts"),
  postActions: document.querySelector("#post-actions"),
  postPlatforms: document.querySelector("#post-platforms"),
};

function percentage(value) {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

function number(value, digits = 1) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

function addDefinition(list, term, description) {
  const row = document.createElement("div");
  const name = document.createElement("dt");
  const value = document.createElement("dd");
  name.textContent = term;
  value.textContent = description;
  row.append(name, value);
  list.append(row);
}

function renderCounts(list, counts) {
  list.replaceChildren();
  const entries = Object.entries(counts || {});
  if (!entries.length) {
    addDefinition(list, "No records", "—");
    return;
  }
  for (const [key, value] of entries.sort(([left], [right]) => left.localeCompare(right))) {
    addDefinition(list, key.replaceAll("_", " "), Number(value).toLocaleString());
  }
}

function renderDistribution(list, summary, unit = "") {
  list.replaceChildren();
  const suffix = unit ? ` ${unit}` : "";
  addDefinition(list, "Observations", Number(summary.count || 0).toLocaleString());
  for (const [label, field] of [
    ["Minimum", "minimum"],
    ["25th percentile", "p25"],
    ["Median", "median"],
    ["Mean", "mean"],
    ["75th percentile", "p75"],
    ["Maximum", "maximum"],
  ]) {
    addDefinition(list, label, summary[field] === null ? "—" : `${number(summary[field])}${suffix}`);
  }
}

function renderReport(report) {
  const holdout = report.serving_release?.evaluation_evidence?.external_holdout_v1 || {};
  elements.offlinePairwise.textContent = percentage(holdout.pairwise_accuracy);
  elements.offlineTopOne.textContent = percentage(holdout.top_1_hit_rate);
  elements.offlineRegret.textContent = number(holdout.mean_top_1_regret, 3);
  elements.offlineTopThree.textContent = percentage(holdout.top_3_hit_rate);
  elements.releaseDetails.replaceChildren();
  addDefinition(elements.releaseDetails, "Release", report.serving_release?.release_id || "—");
  addDefinition(elements.releaseDetails, "Status", holdout.status?.replaceAll("_", " ") || "—");
  addDefinition(elements.releaseDetails, "Evaluation set", `${holdout.video_count || 0} videos / ${holdout.clip_count || 0} clips`);
  addDefinition(elements.releaseDetails, "Model digest", holdout.frozen_model_sha256 || "—");

  elements.accountCount.textContent = report.scope.authenticated_account_count;
  elements.videoCount.textContent = report.scope.source_video_count;
  elements.clipCount.textContent = report.scope.clip_count;
  elements.presentedCount.textContent = report.model_behavior.presented_model_clip_count;
  elements.selectedCount.textContent = report.model_behavior.selected_model_clip_count;
  elements.selectionRate.textContent = percentage(report.model_behavior.selection_rate);
  elements.editCount.textContent = report.editing.edited_download_count;
  elements.analyticsCount.textContent = report.analytics.import_count;

  elements.rankBody.replaceChildren();
  const ranks = report.model_behavior.selection_by_display_rank;
  if (!ranks.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.textContent = "No presented recommendations yet.";
    row.append(cell);
    elements.rankBody.append(row);
  } else {
    for (const rank of ranks) {
      const row = document.createElement("tr");
      for (const value of [rank.rank, rank.presented, rank.selected, percentage(rank.selection_rate)]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      elements.rankBody.append(row);
    }
  }

  renderCounts(elements.eventCounts, report.model_behavior.event_counts);
  renderCounts(elements.sourceTypes, report.media.source_file_types);
  renderCounts(elements.clipOrigins, report.media.clip_origins);
  renderCounts(elements.clipPlatforms, report.media.clip_platforms);
  renderDistribution(elements.sourceDuration, report.media.source_duration_seconds, "seconds");
  renderDistribution(elements.clipDuration, report.media.clip_duration_seconds, "seconds");
  renderDistribution(elements.selectedDuration, report.media.selected_duration_seconds, "seconds");
  renderCounts(elements.exportFormats, report.media.export_format_counts);
  renderCounts(elements.videoStatuses, report.media.video_status_counts);
  renderDistribution(elements.startEdits, report.editing.absolute_start_change_seconds, "seconds");
  renderDistribution(elements.endEdits, report.editing.absolute_end_change_seconds, "seconds");
  renderDistribution(elements.totalEdits, report.editing.total_boundary_change_seconds, "seconds");
  renderCounts(elements.analyticsRoles, report.analytics.report_roles);
  renderCounts(elements.analyticsPlatforms, report.analytics.platforms);
  renderCounts(elements.performancePlatforms, report.analytics.performance_platforms);
  renderCounts(elements.analyticsTypes, report.analytics.report_types);
  renderCounts(elements.analyticsFiles, report.analytics.file_types);
  renderCounts(elements.analyticsMetrics, report.analytics.metric_coverage);
  renderCounts(elements.performanceMetrics, report.analytics.performance_metric_coverage);
  renderDistribution(elements.retentionPoints, report.analytics.retention_points_per_import, "points");
  elements.outcomeDistributions.replaceChildren();
  const outcomeEntries = Object.entries(
    report.analytics.performance_metric_distributions || {},
  );
  if (!outcomeEntries.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.textContent = "No clip outcome metrics have been uploaded yet.";
    row.append(cell);
    elements.outcomeDistributions.append(row);
  } else {
    for (const [metric, summary] of outcomeEntries) {
      const row = document.createElement("tr");
      for (const value of [
        metric.replaceAll("_", " "),
        summary.count,
        number(summary.minimum),
        number(summary.median),
        number(summary.mean),
        number(summary.maximum),
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      elements.outcomeDistributions.append(row);
    }
  }
  renderCounts(elements.jobStatuses, report.operations.job_status_counts);
  renderDistribution(elements.jobAttempts, report.operations.job_attempt_distribution, "attempts");
  renderCounts(elements.postActions, report.repurposing.action_counts);
  renderCounts(elements.postPlatforms, report.repurposing.platform_counts);

  const community = report.community_learning;
  elements.communityEditorialCreators.textContent = community.editorial.contributor_count;
  elements.communityEditorialDecisions.textContent = community.editorial.decision_count;
  elements.communityPerformanceOptIns.textContent =
    community.performance.opted_in_account_count;
  elements.communityConsentEvents.textContent = community.performance.audit_event_count;
  elements.communityFlowBody.replaceChildren();
  for (const flow of community.flows) {
    const row = document.createElement("tr");
    for (const value of [flow.source, flow.destination, flow.permission]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    elements.communityFlowBody.append(row);
  }
  elements.communityPlatformBody.replaceChildren();
  for (const [platform, model] of Object.entries(
    community.performance.platform_models,
  )) {
    const row = document.createElement("tr");
    for (const value of [
      platform,
      model.active ? "active" : "waiting for evidence",
      model.contributor_count,
      model.eligible_clip_count,
      model.strategy.replaceAll("_", " "),
      (model.comparable_signals || []).join(", ") || "global pool",
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    elements.communityPlatformBody.append(row);
  }
  elements.communityPolicy.textContent =
    `${community.policy_version} · editorial collection automatic · audience analytics explicit opt-in · ` +
    `global model auto-training ${community.global_promotion.automatic_retraining ? "enabled" : "disabled"}.`;
  elements.communityEditorialPolicy.replaceChildren();
  addDefinition(
    elements.communityEditorialPolicy,
    "Status",
    community.editorial.active ? "active" : "waiting for evidence",
  );
  addDefinition(
    elements.communityEditorialPolicy,
    "Activation gate",
    `${community.editorial.minimum_contributor_count} creators / ${community.editorial.minimum_decision_count} decisions`,
  );
  addDefinition(
    elements.communityEditorialPolicy,
    "Maximum adjustment",
    `±${community.editorial.maximum_adjustment}`,
  );
  addDefinition(
    elements.communityEditorialPolicy,
    "Feature weights",
    Object.entries(community.editorial.feature_weights)
      .map(([name, value]) => `${name.replaceAll("_", " ")} ${Number(value).toFixed(3)}`)
      .join(", "),
  );
  elements.communityPerformancePolicy.replaceChildren();
  const firstPlatformModel = Object.values(community.performance.platform_models)[0];
  addDefinition(
    elements.communityPerformancePolicy,
    "Activation gate",
    `${firstPlatformModel.minimum_contributor_count} creators / ${firstPlatformModel.minimum_clip_count} clips / ${firstPlatformModel.minimum_clips_per_contributor} per creator`,
  );
  addDefinition(
    elements.communityPerformancePolicy,
    "Maximum adjustment",
    `±${firstPlatformModel.maximum_adjustment}`,
  );
  for (const [platform, model] of Object.entries(
    community.performance.platform_models,
  )) {
    addDefinition(
      elements.communityPerformancePolicy,
      `${platform} weights`,
      Object.entries(model.feature_weights)
        .map(([name, value]) => `${name.replaceAll("_", " ")} ${Number(value).toFixed(3)}`)
        .join(", "),
    );
  }
  renderDistribution(
    elements.communityEditorialAdjustments,
    community.applied_adjustments.editorial,
  );
  renderDistribution(
    elements.communityPerformanceAdjustments,
    community.applied_adjustments.performance,
  );
  renderDistribution(
    elements.communitySemanticAdjustments,
    community.applied_adjustments.semantic,
  );

  elements.accountBody.replaceChildren();
  if (!report.accounts.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 12;
    cell.textContent = "No creator accounts have been created yet.";
    row.append(cell);
    elements.accountBody.append(row);
  } else {
    for (const account of report.accounts) {
      const row = document.createElement("tr");
      const accountLabel = account.email
        ? `${account.display_name}\n${account.email}`
        : `${account.display_name}\nlegacy local profile`;
      const accessLabel = `${account.role}\n${account.account_status.replaceAll("_", " ")}`;
      for (const value of [
        accountLabel,
        accessLabel,
        account.video_count,
        account.model_clip_count,
        account.selected_model_clip_count,
        percentage(account.selection_rate),
        account.custom_clip_count,
        account.edited_download_count,
        account.median_total_boundary_change_seconds === null
          ? "—"
          : `${number(account.median_total_boundary_change_seconds)}s`,
        `${account.analytics_import_count}\n${formatCounts(account.analytics_platform_counts)}`,
        `${account.performance_report_count}\n${formatCounts(account.performance_platform_counts)}`,
        account.performance_contribution_enabled ? "OPTED IN" : "PRIVATE ONLY",
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      elements.accountBody.append(row);
    }
  }
  elements.status.textContent =
    `Updated ${new Date(report.generated_at).toLocaleString()} · ` +
    `${report.analytics.recognized_row_count.toLocaleString()} analytics rows parsed · ` +
    `${report.repurposing.feedback_count.toLocaleString()} post-copy decisions.`;
}

async function loadReport() {
  elements.refresh.disabled = true;
  elements.error.hidden = true;
  try {
    const response = await fetch("/api/admin/model-report", { cache: "no-store" });
    const report = await response.json();
    if (!response.ok) throw new Error(report.error || "The observatory could not be loaded.");
    renderReport(report);
  } catch (error) {
    elements.error.textContent = error.message || "The observatory could not be loaded.";
    elements.error.hidden = false;
    elements.status.textContent = "No report is available.";
  } finally {
    elements.refresh.disabled = false;
  }
}

elements.refresh.addEventListener("click", loadReport);
loadReport();
