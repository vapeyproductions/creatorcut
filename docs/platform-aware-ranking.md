# Platform-aware clip ranking and analytics

CreatorCut can produce separate ranked sets for YouTube Shorts, Instagram Reels, and TikTok from
one long-form upload. This is not three copies of the same list: the frozen quality estimate stays
shared, while platform fit and outcome learning are applied independently.

## Ranking path

```text
long-form video
  -> sentence-aligned candidates
  -> publishability eligibility
  -> frozen MiniLM + ridge quality predictions
  -> compact audio and visual delivery descriptors
  -> creator editorial preference
  -> creator-and-platform outcome preference
  -> bounded platform prior
  -> strict temporal diversity and clip-count planning
```

The global model predicts hook, completeness, payoff, and clarity from the transcript. Its artifact
and MiniLM revision remain frozen. Product activity cannot overwrite it. A future global release
still requires evaluation on untouched source videos.

The production delivery pass decodes audio once and samples video frames once per second. Each
candidate stores:

- active-speech energy, energy variation, opening energy change, silence and clipping ratios;
- transcript words per second;
- colorfulness, saturation, motion, luminance variation, and scene-change rate;
- within-video audio-urgency and visual-excitement percentiles.

Within-video normalization matters: a quiet interview should not be penalized simply because a
different creator publishes loud gameplay. These descriptors are inputs and associations, not
claims that loudness or color causes success.

## Platform profiles

All default exports are vertical 9:16 at 720×1280 with optional burned captions. CreatorCut's
current generation band is 20–60 seconds so the same candidate generator remains precise,
editable, and conservatively inside the supported short-form upload paths.

The initial bounded prior emphasizes different evidence:

- **YouTube Shorts:** completeness, payoff, hook, and fit around a 45-second center;
- **Instagram Reels:** visual activity, hook, audio delivery, and fit around 32 seconds;
- **TikTok:** hook, audio urgency, visual activity, and fit around 25 seconds.

The prior can move a candidate by at most ±0.18 rating points. It is deliberately a transparent
cold-start rule, not a trained claim about platform algorithms. As creator-specific outcome data
arrives, the learned platform layer supplies the personalized evidence.

YouTube currently categorizes square or vertical uploads of up to three minutes as Shorts. TikTok
Studio's web uploader documents MP4/WebM, at least 720×1280, up to 30 minutes, and less than 10 GB.
The current 20–60 second generation band is therefore a CreatorCut quality/space decision rather
than the maximum accepted by either platform.

Official references:

- [YouTube: understand three-minute Shorts](https://support.google.com/youtube/answer/15424877?hl=en)
- [TikTok: Studio tools, analytics, and upload requirements](https://support.tiktok.com/en/using-tiktok/creating-videos/creator-tools-on-tiktok?lang=nl)
- [TikTok: how content is recommended](https://support.tiktok.com/en/using-tiktok/exploring-videos/how-tiktok-recommends-content?ftag=YHF4eb9d17)
- [Instagram: Reels total and average watch time](https://about.fb.com/news/2023/04/instagram-reels-trending-audio-and-gifts-updates/)

## Clip-count recommendation

The creator can enter one to eight clips per selected platform or leave the count on **Auto**.
CreatorCut reports both a recommended value and a suggested range. The planner:

1. removes candidates more than 0.65 points below that platform's strongest option;
2. permits at most 10% interval-overlap between delivered clips;
3. gives the source a duration budget of roughly one clip per five minutes, capped at six for Auto;
4. reports how many distinct usable moments are actually available;
5. caps an explicit request when the source cannot support that many non-repetitive clips.

This is an explainable capacity estimate, not an optimal posting-frequency prediction. It prevents
a requested number from forcing duplicate, weak, or highly overlapping output.

## Analytics ingestion

Published clips accept `.csv`, `.tsv`, `.xlsx`, and `.zip` exports. The XLSX path covers a Google
Sheet downloaded as Microsoft Excel; the product does not currently request live Google-account
access. ZIP archives may contain the multiple CSV tables exported by creator tools.

The parser is header-driven because exported columns vary by platform, report configuration,
account eligibility, locale, and product version. It normalizes common names into one schema:

| Signal family | Examples accepted |
| --- | --- |
| Identity | video ID, post ID, media ID, reel ID, content/caption |
| Exposure | views, plays, initial plays, reach, shown in feed, impressions |
| Attention | engaged views, total watch/play time, average watch time, average viewed %, completion rate |
| Response | likes, comments, shares, saves, total interactions |
| Conversion | follows/new followers, profile visits, subscriber change |
| Repeat viewing | replays |
| Source timing | elapsed-video ratio and audience-retention ratio for YouTube long-form reports |

Unknown columns are preserved only in the uploaded report file, not invented as known metrics.
Missing values remain missing. The admin observatory reports platform, file type, recognized rows,
metric coverage, and outcome distributions so schema drift is visible.

## How outcomes affect future clips

Performance learning is isolated by creator **and** platform. A creator's Instagram history cannot
adjust TikTok clips, and one account cannot affect another account's personalized ranker.

The layer activates only after at least five distinct clips on that platform have 50 or more views
or engaged views and at least one useful outcome signal. Raw views provide an exposure gate; they
are not the target. Eligible outcomes are converted to within-creator, within-platform percentiles
using attention, completion, engagement, saves, follows, reach-normalized response, profile visits,
and replays when available.

Two bounded models then rerank future candidates:

- an interpretable ridge profile over predicted quality, duration, audio urgency, and visual
  excitement;
- a frozen-embedding semantic neighbor model that transfers outcomes from topically similar clips.

The semantic path means recurring subjects and phrases can influence related future clips even when
the wording is not identical. Visible stronger/weaker words and deterministic summaries explain
the evidence history, while the embedding similarity performs the scoring.

YouTube's newer raw Shorts view count includes starts and replays without a minimum watch-time
requirement, so CreatorCut prefers engaged views as an exposure denominator when present. TikTok's
own description of recommendation signals includes watching in full, skipping, likes, shares,
comments, and watch time. Instagram documents total and average watch time for diagnosing where a
stronger hook may be needed. These sources motivate the supported fields; CreatorCut does not claim
to reproduce any platform's private recommendation algorithm.

- [YouTube: Shorts views and engaged views](https://support.google.com/youtube/answer/10059070?hl=en)
- [TikTok: recommendation signals](https://support.tiktok.com/en/using-tiktok/exploring-videos/how-tiktok-recommends-content?ftag=YHF4eb9d17)
- [Instagram: Reels watch-time insights](https://about.fb.com/news/2023/04/instagram-reels-trending-audio-and-gifts-updates/)

## Current limits

- CreatorCut accepts exported files; it does not yet use OAuth to pull live platform analytics.
- Some exports contain account-level rows or multiple posts. Clip learning requires a report
  filtered to one published clip so outcomes are not attributed to the wrong candidate.
- Platform priors are conservative cold-start assumptions. Real personalized learning begins only
  after the evidence gate.
- Audience outcomes are observational and confounded by caption copy, posting time, audience,
  distribution, cover, audio choice, and external promotion.
- The global model remains evaluated on human clip-quality labels. Product outcome data is retained
  as a versioned future-training source, not silently treated as ground truth.
