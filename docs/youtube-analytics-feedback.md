# YouTube analytics feedback design

CreatorCut accepts YouTube Studio `.zip` and `.csv` exports for two distinct jobs:

1. **Source-video analytics** provide a bounded attention prior for moments inside the uploaded
   long-form video.
2. **Published-Short analytics** personalize future recommendations to outcomes observed for one
   creator.

Neither source replaces the human-rated global ranker. This separation prevents a highly promoted
or already popular video from being treated as intrinsically better training data.

## What full YouTube reports can contain

YouTube Studio Advanced Mode exports the currently configured report view. A UI export is limited
to 500 rows; larger or automated histories require the YouTube Analytics or Reporting APIs. The
available reports include:

- performance by date, video, country, subscription status, live/on-demand status, and Shorts vs.
  video content type;
- engaged views, views, watch time, average view duration, average percentage viewed, likes,
  comments, shares, playlist activity, and subscribers gained/lost;
- thumbnail impressions and click-through rate;
- traffic source and playback location;
- device type and operating system;
- demographics, sharing service, and subtitle/caption usage;
- audience-retention points using elapsed-video ratio, audience-watch ratio, and relative
  retention performance.

Sources: [Advanced Mode export](https://support.google.com/youtube/answer/9717005),
[YouTube Analytics metrics](https://developers.google.com/youtube/analytics/metrics),
[channel bulk reports](https://developers.google.com/youtube/reporting/v1/reports/channel_reports),
and [full report list](https://developers.google.com/youtube/reporting/v1/reports/full_report_list).

## Shorts outcome hierarchy

Raw Shorts views are an exposure measure, not CreatorCut's quality target. Since March 31, 2025,
Shorts views count starts and replays with no minimum watch-time requirement. YouTube retained
**engaged views** for viewers who stayed past the initial seconds, excluding loops, and bases Shorts
average view duration and average percentage viewed on engaged views.

CreatorCut therefore prioritizes:

1. **Engaged views** as the rate denominator when available.
2. **Stayed to watch / chose to view** for opening strength.
3. **Average percentage viewed** and **average view duration relative to clip duration** for
   retention.
4. **Shares, comments, likes, and net subscribers per engaged view** for downstream value.
5. Raw views, shown-in-feed counts, impressions, and CTR as exposure or packaging context only.

This follows YouTube's definitions for
[Shorts content analytics](https://support.google.com/youtube/answer/12220281),
[Shorts discovery metrics](https://support.google.com/youtube/answer/12942217), and the
[2025 view-count change](https://support.google.com/youtube/answer/10059070).

## Source-video moment signal

The original video's aggregate views cannot identify a good clip. A timestamped audience-retention
curve can. CreatorCut maps each candidate's start and end timestamps to `elapsedVideoTimeRatio`,
averages retention within that interval, and compares it with the video's own baseline. Relative
retention is preferred when present; otherwise the audience-watch ratio is centered on the video's
median.

The adjustment activates only with at least 100 source views and 10 usable retention points, is
bounded to ±0.20 rating points, and is applied after global ML scoring. YouTube notes that automatic
top-moment, spike, and dip highlighting generally requires at least 100 views, so the same threshold
is used as a conservative product gate. See [audience-retention reporting](https://support.google.com/youtube/answer/9314415)
and [retention dimensions](https://developers.google.com/youtube/analytics/dimensions).

## Creator outcome personalization

Each published clip report is joined back to the exact shown clip, its frozen model outputs, its
transcript embedding, and the final timestamps the creator exported. Metrics are converted to
within-creator percentiles before fitting two complementary preference components. This avoids
comparing a small channel's raw totals with a large channel's totals.

- The structured component learns whether hook, completeness, payoff, clarity, or duration are
  associated with stronger outcomes.
- The semantic component compares a new candidate's frozen transcript embedding with previously
  published clips and transfers the outcomes of the most semantically similar examples. This can
  capture recurring subject matter and phrasing even when the exact words differ.

For transparency, recurring unigrams and two-word phrases with positive outcome associations are
shown in the interface. These terms explain the observed history; candidate scoring itself uses the
full semantic embedding rather than exact keyword matching.

The performance layer requires at least five distinct Shorts, at least 50 views or engaged views
per Short, at least one useful outcome metric, and non-constant outcomes. Both components shrink
toward zero with small samples, and their combined contribution is capped at ±0.25 rating points.
The current layer is an adaptive product feature, not a validated portfolio accuracy claim.

Traffic source, country, device, demographics, and publication context are preserved in the parsed
report for future cohort analysis. They do not directly boost a clip because they are strong
confounders: distribution and audience composition affect outcomes independently of clip quality.

## Parser and product behavior

The import layer is tolerant of Studio labels and Analytics API-style camelCase headers. It stores
recognized raw rows plus normalized totals, daily series, report types, and retention points. It
accepts either a single CSV or a ZIP containing up to 30 CSVs, rejects oversized or unsupported
uploads, and explains when a saved report is not yet eligible to influence ranking.

Users can attach source analytics during upload or after processing and can attach a published
Short report to any returned clip. A newly imported report affects the next applicable ranking run;
it never silently rewrites recommendations already shown.

New recommendations persist their frozen transcript embedding when they are ranked. If analytics
are attached to a recommendation produced by an earlier local build, CreatorCut backfills that
embedding before saving the outcome so legacy recommendations can participate in semantic trends.

## Leakage and feedback-loop controls

- The frozen global model remains unchanged by product analytics.
- Outcome normalization happens only within one creator.
- Every recommendation logs a `presented` event so exposure is distinguishable from rejection.
- Editorial choices and audience outcomes remain separate adjustment layers.
- Only the newest report for a clip contributes, preventing repeated imports from duplicating it.
- Global model promotion still requires versioned training data and evaluation on unseen source
  videos and creators.
- A future production experiment should add a small randomized exploration bucket and log selection
  propensity before estimating causal performance effects.
