# Publishability gate

CreatorCut separates two questions that are easy to conflate:

1. **Is this interval safe and coherent enough to propose?**
2. **Among eligible intervals, which three have the strongest content value?**

The first question is handled by a versioned deterministic gate. The second remains the job of the
frozen learned ranker plus bounded creator and audience adjustments.

## Current rules

`creatorcut_publishability_rules_v1` records high-precision block rules for explicit sponsor/promo
phrases and music-only transcript markers. Blocked intervals are removed before semantic embedding
and ranking. Review-level signals cover context-dependent first words, incomplete endings,
intro/outro language, unusually sparse or dense speech, high filler ratio, and near-maximum clip
length. These do not hard-block an interval; they apply a combined deterministic penalty capped at
0.35 and remain visible with the recommendation.

Every proposed or creator-authored clip stores the full versioned assessment, individual reason
codes, underlying signals, and applied adjustment. The same fields appear in the offline feedback
snapshot. This lets later experiments learn a calibrated classifier from explicit user decisions
without reconstructing which rules happened to be live at collection time.

## Why this is not presented as a trained classifier

The first ratings treated “publishable” as technically exportable, so nearly every clip received a
positive value. Seven structured failure diagnoses are useful qualitative evidence but are not a
large or balanced negative class. Training a classifier on that target would produce a misleading
accuracy number and likely learn the collection procedure instead of publishability.

The current rules are therefore labeled as rules, and their score is not described as a
probability. A supervised replacement should be promoted only after collecting explicit reasoned
accept/reject labels across new source videos, evaluating by held-out video, measuring precision and
recall for every blocking category, and comparing it against the frozen rules. Until then, hard
blocks are deliberately narrow and review-level warnings preserve creator control.

## Development shadow check

The rules were run without fitting on all 192 reviewed development clips from 32 source videos. One
interval was hard-blocked: an explicit sponsor read with a human quality score of 1.0. No source
video had every human-best interval blocked. Review signals flagged 74 context-dependent starts, 19
incomplete endings, 28 long intervals, six dense-speech intervals, and one sparse-speech interval.

Those numbers are a regression check, not a recall estimate: the reviewed set was not sampled to
contain every type of ad, music, or context failure. The machine-readable summary is stored at
`data/v2_development/publishability_gate_summary.json`, and
`creatorcut-evaluate-publishability` reproduces it from the private reviewed records.
