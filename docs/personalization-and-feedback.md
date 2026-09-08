# Personalization and feedback design

## Product loop

CreatorCut's local product surface runs one deliberately narrow workflow:

1. A creator profile uploads a source video.
2. The existing transcription and sentence-aligned candidate pipeline produces possible clips.
3. Frozen ranker v1 scores the candidates.
4. A diversity pass returns three recommendations from different moments in the source.
5. The creator downloads the original interval, changes its boundaries and downloads the edit, or
   explicitly rejects it.
6. If the clip is published, the creator can later enter platform performance.

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

A future performance model should use platform-specific outcomes such as average viewed percentage,
shares per view, and comments per view, normalized against the creator's recent baseline. It should
also retain the model version and publication context. The current schema collects the raw inputs
without yet using them to influence ranking.

## Avoiding a feedback loop

Learning only from top-ranked results would reinforce the current model's beliefs and hide candidates
it undervalues. A production training loop therefore needs a small exploration allocation, explicit
impression logging, propensity or rank-bias correction, and evaluation on creators and source videos
that were not used for fitting. Personal data should be isolated by creator identity, exportable,
and deletable before hosted accounts are introduced.
