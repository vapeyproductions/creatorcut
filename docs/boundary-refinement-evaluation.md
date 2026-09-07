# Boundary-refinement baseline

## Question

Can local sentence boundaries and acoustic pauses turn a promising model-selected moment into the
edited interval a human reviewer would publish?

This experiment deliberately separates **content selection** from **boundary refinement**. The
ranking model chooses the moment. The refiner is allowed to move its start and end by up to 30
seconds without changing which source video is being considered.

## Fixed baseline

For each of the seven ranking failures, CreatorCut analyzes a 35-second context window on both
sides of the model-selected interval. It uses two pretrained components:

- `faster-whisper tiny.en` supplies word timestamps and punctuation.
- Silero voice-activity detection supplies speech/pause intervals.

Word sequences are split at punctuation, pauses of at least 650 ms, or a 20-second safety limit.
The fixed scoring rule rewards self-contained openings, complete endings, and nearby pauses. It
penalizes context-dependent opening words, intro/advertising phrases, large boundary movement, and
large duration changes. Valid proposals last 8–90 seconds and retain some of the original moment.

The rule and its weights were specified before evaluation. Proposal generation reads only the
failure case and the local audio window. It writes every proposal before loading the manual
corrections, so the labels cannot influence the proposed timestamps.

## Data

The review contains seven conditional clip preferences. Five cases have a numeric start
correction and one has a numeric end correction. This is enough for failure analysis, not model
training or a statistically reliable comparison. The zero-change baseline simply preserves each
original model boundary.

## Results

| Boundary | Labeled edits | Fixed refiner MAE | Zero-change MAE | Within 2 seconds |
|---|---:|---:|---:|---:|
| Start | 5 | 16.36 s | **14.56 s** | 0.0% |
| End | 1 | 24.14 s | **5.00 s** | 0.0% |

The fixed refiner is worse than making no change: start MAE increases by 1.80 seconds. The only
labeled end is missed by 24.14 seconds, so no general conclusion about end performance is
possible.

## Interpretation

The reviewer corrections are usually semantic rather than mechanical. “Include the setup,”
“remove the introductory section,” and “begin where the statement becomes clear” can require
moving across several valid pauses and complete sentences. A system that optimizes only clean
local syntax and silence therefore chooses polished boundaries around the wrong amount of
context.

The run also exposes an important distinction in the earlier analysis: preferring the model's
moment *after editing* does not mean its raw interval was the better clip. The model may identify
valuable content while candidate generation still produces an unusable cut.

## Decision

CreatorCut retains this implementation as the first reproducible boundary baseline and does not
tune it on the same six numeric labels. The next boundary model should score several locally valid
variants using semantic context, then learn from direct comparisons such as “too early,” “too
late,” or “best of these boundaries.” Evaluation must hold out complete source videos so variants
from one video never appear in both training and test data.

## Reproduce locally

```bash
creatorcut-refine-boundaries --media-dir /path/to/downloaded/videos
```

The command caches only the seven context windows, proposals, and aggregate evaluation under
`data/processed/`. Those private artifacts occupy about 223 KB in this run and remain excluded
from Git. The downloaded `tiny.en` model occupies about 74 MB and can be removed after the cached
windows are created.
