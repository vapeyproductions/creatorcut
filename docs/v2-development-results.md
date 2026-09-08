# V2 development: attainable-quality ranking and boundary learning

## Status

This is a development result, not a second external holdout result. The original v1 model was
evaluated once against its sealed predictions, and that published result remains unchanged. Only
afterward were those 96 reviewed clips promoted into the v2 development corpus. Any claim about
v2 generalization requires another set of entirely unseen videos.

## Why the target changed

The v1 target scored each originally proposed interval. That combines two different errors:

1. whether the underlying moment is worth publishing; and
2. whether the proposed start and end make that moment coherent.

The v2 moment ranker instead uses **attainable quality**. For the 55 clips with a separately scored
boundary edit, it uses the edited-version hook, completeness, payoff, and clarity scores. All other
rows retain their original scores. Boundary choice is trained and evaluated separately. This makes
the intended production system explicit: find a promising moment first, then select clean edges.

## Development corpus and leakage controls

- 192 reviewed clips from 32 source videos
- 96 clips / 16 videos from the original development cohort
- 96 clips / 16 videos promoted only after the frozen v1 evaluation
- exact embedding encoder metadata match and one embedding per annotation ID
- no overlapping video IDs between cohorts
- leave-one-video-out (LOOV) evaluation for moment ranking
- four deterministic folds grouped by source video for the boundary experiment
- no sampler proxy score used as a model feature

LOOV gives every video its own test fold, maximizes the amount of training data available in this
small corpus, and removes the favorable/unfavorable fold assignments observed in repeated four-fold
checks.

## Moment-ranking result

The pointwise model predicts the four human ratings with hybrid transcript features and frozen
MiniLM embeddings. The pairwise model optimizes Bradley–Terry preferences within each source video.
The ensemble is a fixed equal average of their within-video percentile ranks; it has no tuned blend
weight.

| Development estimate | Pairwise accuracy | Top-1 hit | Mean top-1 regret | Top-3 hit | Mean top-3 regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact random expectation | 50.0% | 46.9% | 0.910 | 76.3% | 0.240 |
| Pointwise hybrid ridge | 55.2% | 56.3% | 0.656 | 75.0% | 0.227 |
| Pairwise hybrid logistic | 58.0% | 53.1% | 0.633 | **90.6%** | **0.102** |
| Equal-rank ensemble | **58.6%** | **59.4%** | **0.586** | 81.3% | 0.148 |

The models are complementary: the pointwise objective is better at its first choice, while the
pairwise objective produces a stronger set of three. The equal-rank ensemble has the best pairwise,
top-1, and top-1-regret estimates; pairwise alone has the best top-three set. Product selection can
therefore use the ensemble for rank 1 and preserve pairwise candidates while constructing a diverse
set of three.

### Cohort check

The combined number hides different label distributions, so both cohorts are reported separately.

| Cohort | Random top-1 | Ensemble top-1 | Random regret | Ensemble regret | Ensemble pairwise |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original development | 20.8% | 43.8% | 1.333 | 0.734 | 59.7% |
| Promoted / edited-label cohort | 72.9% | 75.0% | 0.487 | 0.438 | 56.2% |

The promoted cohort has many tied maximum scores after editing, which makes top-1 hit rate much
easier even for random selection. Pairwise accuracy and regret remain necessary alongside hit rate.

## Boundary-learning result

For every reviewed clip, candidate starts and ends were generated from all word-level transcript
boundaries within 35 seconds. Audio energy immediately before and after each option was reduced to
compact numeric features before the source videos were removed.

| Boundary | Material edits | Candidate-oracle MAE | Oracle within 1 s | Edit-gate ROC AUC | No-change MAE | Learned MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Start | 44 | 0.109 s | 100.0% | 0.495 | 6.925 s | 12.771 s |
| End | 25 | 0.212 s | 96.9% | 0.731 | 2.921 s | 4.936 s |

The high oracle coverage proves that transcription resolution and search radius are adequate. The
learned selector nevertheless loses to leaving boundaries unchanged. Local silence, punctuation,
duration, and token cues cannot tell whether a creator wants to include earlier setup or remove a
semantically irrelevant ending. The end-edit gate has useful signal, but it is not enough to deploy
the selector.

## Decision

- Keep frozen model v1 as the current product model until v2 is tested externally.
- Carry the fixed pointwise/pairwise ensemble into the next unseen-video evaluation.
- Keep the boundary candidate generator, but do not deploy the learned boundary selector.
- Add semantic representations of the full clip created by each candidate boundary, rather than
  tuning more pause weights on these same 96 clips.
- Evaluate the next model once on a new video-level holdout before making a portfolio claim.

The public aggregate metrics are in
[`data/v2_development/evaluation_summary.json`](../data/v2_development/evaluation_summary.json).
Private transcripts, labels, candidate-level features, predictions, and media remain excluded from
Git.
