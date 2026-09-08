# Frozen ranker v1

CreatorCut ranker v1 is frozen for external holdout evaluation. New examples must be evaluated
without changing the model, features, thresholds, or selection rule first.

## Frozen specification

| Item | Value |
| --- | --- |
| Model | Multi-output ridge regression |
| Regularization | `alpha=10` |
| Features | 15 transcript/structural features + 384 normalized semantic dimensions |
| Semantic encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Encoder revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Development data | 96 clips from 16 videos |
| Development fingerprint | `d4056622420d095eb6b22a7939e32910c96853fff5d1eeb7a4d572b07f6ec1c9` |
| Selection protocol | Four deterministic folds grouped by video, seed 42 |
| Frozen status | `frozen_for_external_holdout_evaluation` |

The serialized artifact contains the fitted feature means, scales, intercepts, and standardized
coefficients and is versioned at `models/frozen_model_v1.json` for reproducible local inference.
Private annotations, transcripts, media, embeddings, and clip-level holdout predictions remain
outside Git.

## External holdout contract

1. Use only new video IDs that were not among development videos `video_005` through
   `video_020`.
2. Generate and score candidates with the frozen pipeline before inspecting aggregate test
   performance.
3. Do not fit, tune, select features, adjust thresholds, or choose a new model using holdout
   outcomes.
4. Report pairwise accuracy, top-1 hit rate, top-1 regret, quality MAE, and per-video results.
5. Record product-level failure categories separately from model metrics.
6. Only after the frozen report is saved may those examples be reclassified as development data
   for a clearly versioned model v2 experiment.

This separation prevents the next videos from becoming another informal tuning set and makes the
reported generalization result credible to reviewers.

The completed result is reported in
[`external-holdout-v1-results.md`](external-holdout-v1-results.md). Frozen v1 is retained as the
honest benchmark rather than refit in place.
