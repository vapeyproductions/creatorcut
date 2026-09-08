# Personalization and feedback design

## Product loop

CreatorCut's local product surface runs one deliberately narrow workflow:

1. A creator profile uploads a source video.
2. The existing transcription and sentence-aligned candidate pipeline produces possible clips.
3. Frozen ranker v1 scores the candidates.
4. A diversity pass returns three recommendations from different moments in the source.
5. The creator downloads the original interval, changes its boundaries and downloads the edit, or
   explicitly rejects it.
6. If the clip is published, the creator can enter metrics manually or import a YouTube Studio
   ZIP/CSV report.

The current build persists structured metadata in SQLite and video bytes on the local filesystem.
This cleanly maps to a hosted design in which relational records move to D1 or Postgres and video
bytes move to object storage.

## Two levels of learning

The global model and the creator-specific model should answer different questions.

- The global ranker estimates generally useful clip quality: hook, completeness, payoff, and
  clarity. Its supervised releases continue to use versioned datasets and video-grouped holdouts.
- The preference layer estimates what a particular creator tends to choose. It may learn preferred
  duration, pacing, and relative emphasis on the four global outputs. It is a bounded reranker, so
  sparse personal data cannot overwhelm the global model.

The local MVP begins with the global score. After three distinct decisions, a small interpretable
preference adjustment is activated. Its contribution is shrinkage-weighted toward zero and capped
at 0.35 rating points. This is intentionally conservative; it demonstrates online personalization
without pretending that three clicks are enough to train a reliable neural recommender.

## Events and labels

Every displayed recommendation creates a `presented` event with its rank. Capturing exposure is
essential: a missing click is not a rejection if the user never saw the option. Subsequent events
retain the exact interval and decision type:

- `download_original` is positive editorial feedback on the proposed interval.
- `download_edited` is positive feedback on the underlying moment plus an observed boundary
  correction. The start and end deltas are saved as separate learning targets.
- `reject` is explicit negative editorial feedback.

These are implicit preference signals, not interchangeable replacements for the carefully scored
hook, completeness, payoff, and clarity labels. A later global-learning job can construct
within-impression preference pairs, correct for presentation rank, and compare the resulting model
against the frozen benchmark before promotion.

## Audience performance

Post-publication metrics are deliberately stored separately from editorial decisions. Raw views are
not comparable across creators, platforms, account sizes, publication times, or distribution
conditions. They must not be poured directly into the quality target.

The current YouTube performance layer uses engaged views as the preferred denominator, then
within-creator percentiles for stayed-to-watch, average viewed percentage, relative view duration,
engagement rates, and subscriber conversion. It activates only after five distinct clips have at
least 50 views or engaged views and a useful outcome metric. The learned preference contribution is
shrinkage-weighted and capped at 0.25 rating points.

An optional source-video analytics import is handled separately. If a timestamped long-form
audience-retention curve has at least ten points and 100 source views, it can add a within-video
attention adjustment capped at 0.20 points. Aggregate source views never identify clip timestamps.
See [youtube-analytics-feedback.md](youtube-analytics-feedback.md) for the full report mapping,
thresholds, and official metric definitions.

Both layers are product adaptations, not evidence that the global ranker generalizes. Model release
claims still require a separate, untouched video-level evaluation.

## Avoiding a feedback loop

Learning only from top-ranked results would reinforce the current model's beliefs and hide candidates
it undervalues. A production training loop therefore needs a small exploration allocation, explicit
impression logging, propensity or rank-bias correction, and evaluation on creators and source videos
that were not used for fitting. The current store keeps only the latest outcome per clip in the
active preference estimate and caps every adaptive layer. Personal data should be isolated by
creator identity, exportable, and deletable before hosted accounts are introduced.
