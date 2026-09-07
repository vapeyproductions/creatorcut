# Audio-delivery feature ablation

## Question

Do compact audio-delivery measurements improve CreatorCut's ability to rank six candidate clips
from a previously unseen video?

## Experiment

The development set contains 96 human-reviewed clips from 16 videos. Every representation uses
the same deterministic four-fold split grouped by `video_id`, so no clips from a test video appear
in that fold's training set. Every ridge model uses the same fixed `alpha=10`; no audio-specific
hyperparameter search was performed.

The `audio_delivery_v1` extractor decodes each source once at 16 kHz and computes 21 aggregate
descriptors for each reviewed interval:

- speech activity, segment rate, and maximum silence from Silero VAD;
- energy level, variation, trend, opening/closing deltas, and pause statistics;
- pitch level and variation estimated with windowed autocorrelation;
- zero-crossing rate and clipping ratio.

Extraction does not read any human score. Cached rows are joined to labels by unique
`annotation_id` and checked against `video_id`. Feature scaling is fit only on each fold's training
videos by the ridge pipeline.

## Results

All ranking metrics use out-of-fold predictions. Top-1 hit rate is the fraction of videos for
which the model selects a clip tied for the highest human score; regret is the selected clip's
score gap from that video's best clip.

| Representation | Quality MAE | Spearman | Pairwise accuracy | Top-1 hit rate | Mean top-1 regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| Audio only | 1.234 | -0.170 | 46.4% | 12.5% | 1.609 |
| Semantic only | 1.043 | 0.234 | 58.8% | 43.8% | 0.625 |
| Text features + semantic | **1.027** | **0.251** | **62.4%** | **56.2%** | **0.422** |
| Semantic + audio | 1.068 | 0.204 | 55.2% | 37.5% | 0.906 |
| Text features + semantic + audio | 1.049 | 0.236 | 57.9% | 37.5% | 0.938 |

The full multimodal model loses 4.5 percentage points of pairwise accuracy and 18.8 points of
top-1 hit rate relative to text+semantic ridge. Its mean selection regret increases by 0.516
rating points.

A 10,000-iteration paired cluster bootstrap resamples whole videos. For full multimodal minus
text+semantic, the 95% intervals are:

- pairwise-accuracy change: -10.2 to +0.5 percentage points;
- top-1 hit-rate change: -37.5 to 0.0 percentage points;
- regret change: 0.000 to 1.063 rating points;
- quality-MAE change: -0.041 to +0.070 rating points.

Only 3.4% of bootstrap samples improve pairwise accuracy, and none improve top-1 hit rate or
regret. These intervals are still wide because the independent sample size is 16 videos, not 96
clips.

## Decision

The selection rule was declared before running the ablation: between text+semantic ridge and full
multimodal ridge, maximize grouped top-1 hit rate, then minimize regret, then maximize pairwise
accuracy. It selects `text_semantic_ridge`, which is frozen as model v1.

This is evidence that these aggregate audio features do not help this dataset—not that delivery
never matters. The sources are already edited podcasts with relatively consistent production
quality, and aggregate descriptors discard delivery sequence and speaker context. A future audio
experiment should be motivated by errors on a new development cycle rather than tuned against the
external holdout.
