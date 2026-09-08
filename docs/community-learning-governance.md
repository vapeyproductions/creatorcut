# Community learning and governance

CreatorCut separates product feedback into four explicit learning paths. The creator-facing
experience stays simple; the data lineage, activation gates, feature weights, cohort strategy,
and applied adjustment distributions are available only in the authenticated administrator ML
observatory.

## Data-flow contract

| Signal | Destination | Participation | Serving behavior |
| --- | --- | --- | --- |
| Clip selections, rejections, custom clips, and boundary edits | Creator-specific editorial reranker | Automatic product telemetry | Activates after 3 explicit decisions; capped at ±0.35 |
| The same editorial decisions | Creator-balanced community editorial prior | Automatic product telemetry | Activates after 3 creators and 15 decisions; capped at ±0.10 |
| Uploaded audience analytics | Creator-specific platform performance and semantic models | Private processing | Activates after 5 eligible clips on one platform; combined cap ±0.25 |
| Uploaded audience analytics | Comparable-audience/content community model | Explicit account opt-in | Activates after 3 other opted-in creators and 12 eligible clips; combined cap ±0.12 |

An editorial action is part of the core clip-selection service and therefore has no contribution
toggle. Uploaded audience outcomes can reveal more about an audience, so cross-creator use is off
by default and is controlled by one account setting. Every change to that setting creates an
append-only audit event with the policy version and timestamp. Turning it off immediately removes
the account from future community calculations. Existing frozen release artifacts are immutable.

## Community editorial prior

Only explicit decisions are labels. Merely viewing a recommendation or leaving it untouched is
not a negative label; closing a completed review records untouched suggestions as weak negatives.
The newest editorial event for a clip wins. Edited downloads remain positive, but their target is
discounted as the total boundary change grows because a heavily repaired proposal is weaker
evidence than an unchanged selection.

Each creator first receives an independent feature-preference vector. Creator vectors are then
averaged, which prevents one highly active account from dominating the community prior. The final
weight vector is shrinkage-weighted toward zero and uses hook, completeness, payoff, clarity,
duration, audio urgency, and visual excitement. This prior updates ranking online but cannot
replace the frozen global model.

## Comparable-audience performance model

Performance outcomes are normalized within each creator and platform. Raw reach is not treated as
quality: available retention, completion, engagement, conversion, replay, save, and view-choice
signals are converted into centered within-creator outcome targets. Creators need at least three
eligible clips to contribute.

For the creator receiving recommendations, CreatorCut builds a signature from available audience
outcomes, clip duration, aggregate source-video retention, and a normalized transcript-embedding
centroid. It then selects up to five nearest eligible creators using the overlapping fields. If
the target creator has too little history for a meaningful distance, the model falls back to the
whole eligible opt-in pool. This soft neighborhood is deliberately preferable to a hard cluster
assignment at current data volume: it degrades gracefully with missing platform fields and can be
replaced by learned clustering once offline evidence justifies it.

The selected neighborhood produces two separately bounded signals:

- an interpretable feature association over clip quality, length, audio, and visual activity;
- a semantic nearest-example association using frozen transcript embeddings.

Recurring positive or negative terms are reported only when at least two contributing creators
support them. These summaries are correlations, not causal claims.

## Privacy and access boundary

Regular creator responses do not include community cohort identifiers, weights, adjustment
lineage, thresholds, or contribution counts. Creators see ranked recommendations and concise
content insights. The administrator endpoint and page require both an authenticated administrator
account and the explicit server-side admin-dashboard flag.

The shared audience model does not use account names, emails, source-video bytes, local media
paths, or raw transcript text as transferable records. Semantic embeddings and aggregate recurring
terms may be calculated after audience-analytics opt-in. Community reporting omits raw analytics
rows and transcript content.

## Promotion and monitoring

The administrator observatory audits:

- all four data paths and their participation rules;
- editorial contributor and decision counts;
- audience-analytics opt-in and consent-event counts;
- per-platform eligibility, strategy, comparable fields, and cohort size;
- the distributions of editorial, structured-performance, and semantic adjustments actually
  applied in production;
- frozen offline holdout metrics separately from live product-choice signals.

Production feedback never silently retrains or overwrites the frozen ranker. A future global
release must use a versioned snapshot, group evaluation by source video/creator to prevent
leakage, and pass evaluation on untouched videos before promotion.
