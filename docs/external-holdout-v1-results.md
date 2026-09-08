# External holdout v1 results

## Outcome

Frozen ranker v1 retains a measurable within-video ordering signal on new videos, but its absolute
scores are poorly calibrated and its single best recommendation is not reliable enough for a
fully automatic product. Returning three diverse recommendations is a defensible product choice,
provided CreatorCut continues to present them as editorial options rather than guaranteed winners.

This report was generated and committed before any retraining on the holdout examples.

## Evaluation integrity

| Item | Value |
| --- | --- |
| Prediction commitment | `b016380db69b77599588b93c6f631a5499986a0eaa33c12b20f6daf902acff7f` |
| Evaluated clips | 96 |
| Evaluated videos | 16 |
| Candidates per video | 6 |
| Human reviews matched | 96 of 96 |
| Technical export failures | 0 |

The set contains `video_021` through `video_037`, excluding `video_033` because its transcript
failed the prereview repetition check. Each remaining video contributes the same number of sampled
candidates. Frozen prediction, candidate, video, and timestamp identities matched every review.

## Registered metrics

| Metric | Development grouped CV | External holdout |
| --- | ---: | ---: |
| Pairwise accuracy | 62.4% | 58.8% |
| Top-1 hit rate | 56.2% (9/16) | 43.8% (7/16) |
| Mean top-1 regret | 0.422 | 0.828 |
| Quality-score MAE | — | 1.292 |
| Quality-score Spearman | — | 0.242 |

The video-cluster bootstrap 95% percentile interval is 18.8%–68.8% for top-1 hit rate and
0.359–1.375 for mean regret. Sixteen independent video groups are enough to reveal the performance
drop but not enough for a narrow estimate.

Against a uniform random choice among the six displayed candidates, the expected top-1 hit rate is
31.2% and expected regret is 1.227. V1 therefore improves the average ordering, but the advantage is
modest rather than production-grade.

## Calibration failure

The mean human quality score is 3.523 while the mean prediction is 2.699. Forty-three of 384
individual target predictions fall outside the valid 1–5 range, spanning -0.404 to 6.008. The
unbounded ridge output is therefore unsuitable as a user-facing probability or rating even when it
provides some ranking value.

Target-level rank correlations are weak: 0.044 for hook, 0.148 for completeness, 0.239 for payoff,
and 0.089 for clarity. Payoff transfers best; hook and clarity transfer least.

## Exploratory top-three product view

The best human-rated candidate appears among the model's top three for 13 of 16 videos (81.2%), and
mean best-of-three regret is 0.219. These were not preregistered primary metrics and are labeled
exploratory because the product was designed to return three choices.

The comparison needs context: ties at the maximum score are common, so three uniformly random
candidates would achieve an expected 71.6% hit rate and 0.320 regret. The model's top three are
better, but not by enough to claim that clip selection is solved.

## Boundary-edit evidence

Fifty-five of 96 candidates received an optional edited interval. Among the clips the reviewer
chose to edit, 54 improved and one remained tied; the mean four-target quality increase is 1.645.
Completeness changes most (+2.891 on average), followed by hook (+2.036), clarity (+1.000), and
payoff (+0.655). Edited starts move 6.338 seconds earlier on average, while ends move 2.779 seconds
later.

This is conditional evidence: the reviewer edited clips that appeared improvable, so the result is
not an unbiased estimate of editing every candidate. It nevertheless gives CreatorCut 55 directly
observed boundary examples and confirms that ranking a promising moment and selecting clean
boundaries should be modeled as separate stages.

## Decision

Ranker v1 remains frozen and its result is retained. It is not promoted as a finished model.
Holdout v1 may now be reclassified as development data for explicitly versioned v2 experiments,
but any v2 selection must be evaluated on a third, untouched set of source videos.

The highest-value v2 work is:

1. Train and evaluate a boundary-adjustment stage using the observed start and end corrections.
2. Replace unbounded absolute regression with a ranking or ordinal objective and evaluate bounded
   calibration separately.
3. Add discourse-boundary, context dependency, advertisement/music, speaker-turn, and semantic
   coherence signals under grouped evaluation rather than tuning examples individually.
4. Retain the three-option product interface while measuring both top-1 and best-of-three utility.

The machine-readable aggregate is
[`data/holdout_v1/evaluation_summary.json`](../data/holdout_v1/evaluation_summary.json). Clip-level
reviews, transcripts, media, embeddings, and predictions remain outside Git.
