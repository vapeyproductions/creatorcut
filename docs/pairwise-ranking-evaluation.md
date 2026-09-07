# Pairwise learning-to-rank evaluation

## Question

Does training directly on clip preferences improve ranking compared with predicting the four
absolute human ratings?

CreatorCut ultimately needs an ordering, not a calibrated star rating. This experiment changes
only the supervised objective while preserving the representation, labels, grouped folds, and
ranking metrics used by the strongest existing model.

## Preference construction

Each of the 16 reviewed videos has six candidate clips. A clip's quality score is the unweighted
mean of its hook, completeness, payoff, and clarity ratings. CreatorCut compares every pair from
the same video and drops exact quality-score ties, producing 221 preferences from 96 reviewed
clips.

These pairs are derived views of the original ratings; they are not 221 independent human
annotations. Comparisons never cross video boundaries because the intended task is to rank
candidates from one source video.

For clips `i` and `j`, the linear Bradley–Terry model estimates:

```text
P(i preferred to j) = sigmoid(w · (x_i - x_j))
```

Feature differences are oriented so the human-preferred clip is positive. The ranker minimizes
logistic loss with fixed `L2=0.1`. A Newton–Raphson optimizer with backtracking line search
converges in every fold. At inference time, `w · x` becomes each clip's ranking utility.

## Controlled objective ablation

Both models receive the same 399 features:

- 15 timing and transcript-structure features
- 384 frozen MiniLM transcript-embedding dimensions

The comparison is:

1. **Pointwise hybrid ridge:** minimize squared error on the four absolute ratings and average
   the predicted ratings into a quality score.
2. **Pairwise hybrid logistic:** minimize logistic loss on non-tied within-video preferences.

All predictions are out-of-fold. Four deterministic folds are grouped by `video_id`, leaving
four complete videos for testing in each fold. Feature scaling, preference construction, and
model fitting use only the 12 training videos. Technical exportability remains a separate gate.

## Results

| Objective | Pairwise accuracy ↑ | Top-1 hit ↑ | Top-1 regret ↓ |
|---|---:|---:|---:|
| Pointwise hybrid ridge | **0.624** | **0.562** | **0.422** |
| Pairwise hybrid logistic | 0.575 | 0.312 | 0.891 |

The pairwise model orders 127 of 221 comparable held-out pairs correctly, selects a top-rated
clip in 5 of 16 unseen videos, and loses an average of 0.89 rating points when its top selection
is not optimal. The pointwise model selects a top-rated clip in 9 of 16 videos.

## Uncertainty

A paired cluster bootstrap resamples whole videos 10,000 times. Deltas are pairwise minus
pointwise:

| Pairwise minus pointwise | Estimate | 95% bootstrap interval | Probability improved |
|---|---:|---:|---:|
| Pairwise accuracy | -0.050 | [-0.126, 0.032] | 10.7% |
| Top-1 hit rate | -0.250 | [-0.500, -0.062] | 0.0% |
| Top-1 regret | +0.469 | [0.062, 0.969] | 0.0% |

The pairwise-accuracy interval includes zero, but both selection-level intervals favor the
pointwise model. On this dataset, optimizing individual preferences does not improve the final
top-clip decision.

## Interpretation and decision

Pairwise expansion changes the loss but does not create new information: the effective sample
still contains only 16 independent source videos. The unweighted preference target also treats a
0.25-point rating difference like a four-point difference, discarding confidence available in
the original ratings. The pointwise model retains that magnitude information and can learn the
four component judgments separately.

CreatorCut therefore keeps pointwise hybrid ridge as the current ranker. This is a model-selection
result rather than a failed implementation: the optimizer converged, the evaluation remained
leakage-safe, and uncertainty was measured at the video level. A more complex ranker is not
justified by the observed evidence.

## Limitations

- The evaluation covers 96 clips from 16 videos and one reviewer.
- Derived preference pairs are statistically dependent within a video.
- Regularization was fixed before evaluation rather than tuned with nested cross-validation.
- The model uses transcript and timing information but no vocal-delivery or visual features.
- The experiment ranks content quality; it does not replace the technical-exportability gate.

## Reproduce locally

Build the private reviewed table, encode the transcript text, and run the objective ablation:

```bash
creatorcut-build-training-data --from-queue
creatorcut-encode-transcripts
creatorcut-train-pairwise
```

Reviews, training rows, embeddings, fitted model weights, predictions, and evaluation artifacts
remain excluded from Git. The public repository contains the feature construction, optimizer,
grouped cross-validation, metrics, bootstrap procedure, tests, and this aggregate report.
