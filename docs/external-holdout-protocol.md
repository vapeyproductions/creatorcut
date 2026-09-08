# External holdout protocol

## Purpose

CreatorCut uses a new set of source videos to measure frozen ranker v1 without selecting features,
models, hyperparameters, or thresholds from the new outcomes. Development videos and holdout
artifacts remain physically and logically separated.

## Registered set

Holdout v1 reserves `video_021` through `video_037`. Its public manifest is
`data/holdout_v1/videos.json`; transcripts, candidate records, embeddings, reviews, predictions,
and evaluation details remain ignored under `data/processed/holdout_v1/`.

Each uploaded file is registered with a SHA-256 checksum before predictions are sealed. Source
URLs were intentionally not collected for this holdout. The checksums prove which exact local
files were evaluated, while the missing URLs mean the repository cannot independently verify
source-level non-duplication against the development corpus. This limitation is retained rather
than silently backfilled after evaluation.

## Fixed sequence

1. Transcribe the registered media with the same `small.en`, int8, English, word-timestamp, and
   voice-activity configuration used by the development pipeline.
2. Validate transcription identity, timing, confidence, and repetition checks.
3. Generate the unchanged sentence-aligned 20–60 second candidate pool.
4. Sample six candidates per eligible video with the unchanged seed-42 stratified sampler. Its
   proxy scores stay hidden and are not labels.
5. Encode the sampled transcripts with the exact pinned MiniLM revision specified by frozen v1.
6. Run `creatorcut-score-holdout`. The command has no review input and serializes the frozen
   predictions plus a public SHA-256 commitment.
7. Commit and push `data/holdout_v1/prediction_commitment.json` before human review begins.
8. Review every sampled clip in the Annotation Studio, which does not expose model scores or
   ranks. Score the sealed interval as originally proposed; an optional boundary edit and second
   score set may be recorded separately.
9. Run `creatorcut-evaluate-holdout` only after all sealed prediction identities have matching
   reviews.
10. Publish pairwise accuracy, top-1 hit rate, mean top-1 regret, quality MAE, and per-video error
    analysis before any retraining.

## Original and boundary-adjusted labels

The top-level `hook`, `completeness`, `payoff`, and `clarity` fields always describe the exact
candidate interval that was sealed before review. `creatorcut-evaluate-holdout` reads only these
original-interval labels when reporting frozen v1 performance.

If a candidate would improve with different boundaries, the annotation tool may additionally save
`boundary_edit.start_seconds`, `boundary_edit.end_seconds`, and a nested four-score judgment of the
edited version. Edits are limited to 30 seconds in either direction, never overwrite the original
timestamps or scores, and are excluded from v1 model selection and holdout metrics. They form a
separate dataset for later boundary-optimization experiments after the registered evaluation is
reported.

## Leakage controls

- Frozen scoring rejects any `video_id` found in the development cross-validation folds.
- Embedding model, revision, file, truncation length, dimension, normalization, and pooling must
  match the frozen model metadata exactly.
- The frozen feature order, means, scales, intercepts, and coefficients are loaded from the
  serialized artifact rather than refit.
- Queue and model fingerprints are included in the private prediction artifact and public
  commitment.
- Prediction/review joins require the exact same annotation, candidate, and video identities.
- The holdout cannot become v2 training data until the frozen v1 report is saved. If it is later
  used for development, a third untouched corpus is required for an unbiased v2 evaluation.
