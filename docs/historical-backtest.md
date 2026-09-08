# Historical recommendation test

CreatorCut includes a creator-facing test for comparing its ranked intervals with Shorts that a
channel actually published. The test is designed as a small, honest case study rather than a claim
that two source videos are enough to establish general model accuracy.

## Workflow

1. Create a historical test from a CSV, TSV, or XLSX tracker with `Link`, `Views`, `Likes`,
   `Comments`, and `video ID` columns.
2. Upload two reference long videos and all corresponding published Short files. Filename stems
   must match the tracker IDs, such as `training_vid_1_clip3.mp4`.
3. CreatorCut uses audio fingerprints to map each Short back to its long-video timestamps. Low
   confidence and likely compound edits remain visible for manual timestamp correction.
4. Eligible reference intervals are scored with the same frozen transcript, semantic, audio,
   visual, structural, publishability, and platform feature contracts used in serving.
5. A regularized channel calibration layer learns from the reference Shorts' likes-per-view and
   comments-per-view outcomes. It requires at least five aligned Shorts across both reference
   videos. Raw views gate evidence quality but do not act as the training target.
6. Only after that layer is ready can the creator upload the held-out long video. CreatorCut saves
   its exact recommendations, model release, timestamps, ranks, and relative estimates.
7. The application then accepts the organization's held-out Short files, aligns them, and computes
   the comparison report.

## Leakage boundary

The tracker can contain rows for all three videos, but held-out rows are quarantined in the
experiment tables. They are not copied into the serving performance table and no holdout Short
media is accepted before the prediction snapshot exists. The snapshot explicitly records
`holdout_outcomes_used: false`.

The held-out video bypasses ordinary account-history personalization. Its only creator-specific
adjustment is the experiment layer fitted from aligned reference clips. This prevents unrelated
earlier account activity from changing the result.

## Evaluation

The report uses one-to-one temporal matching between recommendations and organization-selected
Shorts.

- `top_k_recall_at_iou_50`: share of aligned published moments recovered at temporal IoU 0.50 or
  higher.
- `mean_best_iou`: mean overlap across the ranked recommendation set.
- boundary error: signed start and end differences for each matched pair.
- performance rank correlation: Spearman agreement between CreatorCut's reference-calibrated
  relative estimates and the outcomes of matched published Shorts.

For reporting only, the actual outcome index combines within-test views, likes per view, and
comments per view. It is a comparative outcome measure, not a predicted number of views. Raw
metrics remain beside the index so a reviewer can see what drove it.

Audio alignment uses a 50 Hz log-energy envelope, normalized FFT cross-correlation, separated peak
checks, and several windows across each Short. Multiple stable source offsets are flagged as a
possible compound edit instead of being silently treated as one continuous ground-truth interval.

## Operational records

SQLite stores the experiment state, role of each source video, tracker rows, private Short paths,
alignment method and confidence, manual corrections, the immutable prediction snapshot, serving
release, and final evaluation. Every API read and write is restricted to the owning creator
account. The ordinary clip workflow does not expose these details.
