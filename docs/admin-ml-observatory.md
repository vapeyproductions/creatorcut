# Cross-account ML observatory

CreatorCut includes a local administrator dashboard at `/admin`. It is disabled by default and
must be started explicitly:

```bash
creatorcut-web --enable-admin-dashboard
```

The first account registered against a new local database in this mode becomes the administrator.
On an existing database, register normally and run
`creatorcut-account promote-admin --email creator@example.com`. Both the dashboard document and
report endpoint verify the server-side session and administrator role.

The observatory is designed to answer two different questions without conflating them.

## 1. Does the frozen model generalize?

This comes from offline evaluation on videos excluded from fitting and feature selection. The
dashboard reads the immutable external-holdout metrics recorded in the serving release:

- pairwise accuracy: whether the model orders two differently rated clips correctly;
- top-1 hit rate: whether its first choice ties a human-best candidate for that source video;
- top-1 regret: the human-quality gap between its first choice and the best reviewed candidate;
- top-3 hit rate: whether one of the three suggestions includes a human-best candidate.

These are model-evaluation metrics. They do not update when a product user clicks a button.

## 2. How is the model behaving in use?

The live sections aggregate operational evidence across creator profiles:

- model clips presented, selected, rejected, and selected by displayed rank;
- proposed clip and source-video duration distributions;
- exact start, end, and total boundary-correction distributions;
- source file types, custom/model clip origins, clip platforms, and export formats;
- YouTube, Instagram, and TikTok analytics imports and outcomes by platform, report type,
  recognized rows, retention-point counts, and metric availability;
- post-copy choices by action and platform;
- automatic editorial-community contributors, labels, feature weights, and adjustment distributions;
- opted-in audience-community contributors, audit events, per-platform eligibility, comparable
  signals, cohort strategy, semantic trends, and applied adjustments;
- persistent processing-job states and retry counts;
- a per-account coverage table.

Authenticated-account and legacy-profile counts are reported separately. Each credentialed row
shows its email and role so the operator can audit which creator owns the associated evidence.

Selection rate is labeled as a product-choice signal, not accuracy. Rank position affects clicks,
an unselected result is not necessarily bad, and audience performance is confounded by packaging,
publication time, audience, and distribution. The dashboard therefore emphasizes denominators and
coverage instead of converting these signals into an unsupported success score.

## Privacy and access boundary

The report omits transcript text, local media paths, and raw analytics rows. It retains local
profile names and account emails because the purpose is account-level operations. The dashboard is
unavailable unless the explicit local admin flag is set, and every request requires an
administrator session. The current credentialed web process refuses non-loopback binding; a public
deployment must replace this local boundary with managed HTTPS and production identity controls.

Regular creator accounts cannot request either the cross-account observatory or the detailed
per-account model report. Model-flow controls, cohort identifiers, weights, activation thresholds,
and adjustment lineage stay inside the administrator boundary. See
[community-learning-governance.md](community-learning-governance.md) for the exact four-flow
contract.
