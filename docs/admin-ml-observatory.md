# Cross-account ML observatory

CreatorCut includes a local administrator dashboard at `/admin`. It is disabled by default and
must be started explicitly:

```bash
creatorcut-web --enable-admin-dashboard
```

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
- source file types, custom/model clip origins, and export formats;
- YouTube analytics import roles, report types, recognized rows, retention-point counts, and
  metric availability;
- post-copy choices by action and platform;
- persistent processing-job states and retry counts;
- a per-account coverage table.

Selection rate is labeled as a product-choice signal, not accuracy. Rank position affects clicks,
an unselected result is not necessarily bad, and audience performance is confounded by packaging,
publication time, audience, and distribution. The dashboard therefore emphasizes denominators and
coverage instead of converting these signals into an unsupported success score.

## Privacy and access boundary

The report omits transcript text, local media paths, and raw analytics rows. It retains local
profile names because the purpose is account-level operations. The dashboard has no application
authentication in the current single-user build, so its routes are unavailable unless the explicit
local admin flag is set. CreatorCut also refuses to combine that flag with a non-loopback host. A
public deployment must add authenticated administrator authorization before enabling it.
