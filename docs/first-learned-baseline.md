# First learned ranking baseline

## Question

Can inexpensive transcript and timing features learn enough from blind human reviews to rank
clips from previously unseen videos?

This experiment is intentionally small and interpretable. It establishes a supervised baseline
before semantic embeddings, audio features, or visual features are introduced.

## Data and target

The dataset contains 96 non-overlapping candidate clips from 16 videos, with six clips per
video. Each clip was reviewed without displaying the sampler's heuristic score and received
four ratings from 1 to 5:

- hook
- completeness
- payoff
- clarity

The primary ranking target is `quality_score`, the unweighted mean of those four ratings.
Presentation is excluded because it is constant in the current edited-podcast corpus.
Technical exportability is retained as a separate gate rather than folded into content quality.

The earlier 17 hand-selected seed clips are not mixed into this experiment. They were collected
with a different selection process and skew strongly positive, so treating them as ordinary
training examples would introduce selection bias.

The sampler's proxy score is not present in the model-ready rows. This prevents the learned
model from merely imitating the heuristic that selected the annotation sample.

## Features and model

The model uses 15 numeric features available at candidate-generation time:

- duration, word count, sentence count, and speaking rate
- whether the opening appears context-dependent
- whether the ending has terminal punctuation
- lightweight hook cues
- duration, density, and sentence-structure fit
- lexical diversity
- filler-word frequency
- intro or outro phrase detection

A multi-output ridge regression predicts the four human ratings. The predicted
`quality_score` is their mean. Features are standardized from training data only, and the
regularization value is fixed at `alpha=10` rather than tuned against the evaluation folds.

## Leakage-safe evaluation

The evaluation uses four deterministic folds grouped by `video_id`. Every clip from a video is
held out together, so a model cannot benefit from seeing the same podcast, speaker, vocabulary,
or editing style during training and testing. Every reported prediction is out-of-fold.

The comparison model predicts the training-fold mean for every held-out clip. Ranking metrics
are calculated only between clips from the same video:

- **Pairwise accuracy:** how often two clips are placed in the correct order; 50% is chance.
- **Top-1 hit rate:** how often the predicted best clip is tied for the actual best clip.
- **Top-1 regret:** the human-score gap between the selected clip and that video's best clip.

## Results

| Model | Quality MAE ↓ | Quality RMSE ↓ | Spearman ↑ | Pairwise accuracy ↑ | Top-1 hit ↑ | Top-1 regret ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Fold training mean | 1.045 | 1.238 | -0.323 | 0.500 | 0.125 | 1.688 |
| Ridge, handcrafted features | 1.074 | 1.275 | 0.120 | 0.557 | 0.188 | 1.266 |

Per-target ridge results:

| Target | MAE ↓ | RMSE ↓ | Spearman ↑ |
|---|---:|---:|---:|
| Hook | 1.278 | 1.492 | 0.131 |
| Completeness | 1.315 | 1.610 | -0.152 |
| Payoff | 1.289 | 1.571 | 0.239 |
| Clarity | 1.377 | 1.610 | 0.047 |

The ridge model does not improve absolute score error over the mean predictor. It does produce
a modest improvement in within-video ordering and reduces the average cost of choosing one
clip per video. Payoff has the strongest individual rank correlation; completeness has none.

That distinction is informative. Surface-level transcript features can recover a small amount
of content-value signal, but they do not understand whether a thought is self-contained or
whether the message delivers a meaningful conclusion. The result motivates semantic text
representations and boundary-context features while leaving a reproducible baseline they must
beat.

Coefficient magnitudes are exploratory rather than causal. With 96 examples and correlated
features, they should be used for debugging hypotheses, not broad claims about audience
behavior. The labels also come from one reviewer, so they represent a consistent editorial
preference rather than consensus across creators.

## Reproduce locally

The reviews, transcripts, generated training rows, predictions, evaluation report, and model
artifact remain ignored because they are derived from private local media. Recreate them with:

```bash
creatorcut-build-training-data
creatorcut-train-ridge
```

The first command validates and joins reviews, queue entries, and generated candidates. The
second writes grouped out-of-fold predictions, the evaluation report, and a portable JSON model
containing feature statistics and coefficients.
