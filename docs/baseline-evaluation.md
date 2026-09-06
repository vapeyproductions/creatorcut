# Transcript baseline evaluation

## Purpose

The first ranking experiment tests whether inexpensive surface-level transcript features are
enough to recover the human-selected clips. Its weights were fixed before evaluation and were
not tuned after inspecting the seed results.

## Baselines

The duration-only baseline favors candidates closest to 40 seconds. The transcript heuristic
combines:

- Sentence-aligned start and end indicators
- Opening hook keywords, questions, and numbers
- Duration and speaking-density fit
- Sentence-count structure
- Lexical diversity and filler rate
- Intro and outro phrase penalties

Overlapping recommendations are suppressed using temporal non-maximum suppression at IoU 0.50.

## Evaluation

The seed set contains 17 human annotations from three source videos. Ten are provisionally
considered strong because their mean hook, completeness, payoff, and clarity score is at least
4.0.

| Baseline | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Human-score Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| Duration only | 0.00 | 0.00 | 0.00 | 0.20 | -0.321 |
| Transcript heuristic | 0.00 | 0.00 | 0.00 | 0.00 | -0.414 |

Recall@K asks what fraction of the ten known strong intervals overlap a top-K recommendation
from the same video at IoU 0.50 or greater. Spearman correlation compares human relevance with
the baseline score of each annotation's most-overlapping generated candidate.

## Interpretation

The heuristic is not aligned with human relevance. Sentence-aligned generation makes ending
punctuation nearly constant, hook dictionaries miss semantic hooks, and professionally edited
podcasts maintain similar speaking density across both strong and weak sections. The heuristic
therefore rewards tidy surface form without understanding whether a clip delivers an important
or satisfying message.

This failed baseline establishes the need for semantic representations and supervised ranking.
It should not be tuned on the same 17 examples: three videos are insufficient for a credible
train/test split, and tuning would convert the evaluation set into training data.

## Limitations

The annotations cover selected intervals rather than every possible candidate. Recall of known
positives is measurable, but precision is not: an unlabeled recommendation may still be a good
clip. A larger pairwise-preference dataset is required before training and evaluating the first
learned ranker.
