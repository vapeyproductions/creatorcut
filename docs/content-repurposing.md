# Personalized content repurposing

CreatorCut can turn any ranked or creator-authored clip into editable copy for YouTube Shorts,
TikTok, Instagram Reels, and LinkedIn. This layer is deliberately transcript-grounded: it reuses
sentences and terms from the selected clip instead of asking a language model to invent claims.

## How generation works

1. Unigrams and bigrams are extracted from the clip transcript.
2. Local TF-IDF compares them with the creator's previously saved clip transcripts.
3. Recurring terms associated with sufficiently exposed, stronger Shorts receive a bounded `1.35`
   multiplier. Terms associated with weaker outcomes receive `0.75`.
4. A complete transcript sentence with the strongest topic overlap becomes the short hook.
5. Platform length limits and transcript-derived hashtags produce four editable outputs.

The stored pack includes the algorithm version, source clip ID, ranking-model version, topic
scores, document frequencies, and any audience-aligned terms. `personalization_active` is false
when no eligible positive audience trend matches the clip.

## Feedback contract

Every generated pack is stored separately from clip-selection labels. For each platform, the
creator may:

- copy the generated text unchanged;
- edit and copy it, preserving both generated and final text; or
- explicitly reject it.

These records are included in the versioned offline feedback export. They are future supervision
for a copy preference model; they do not silently retrain the frozen clip ranker and do not alter
the quality label for the source clip.

## Current limits

This first version optimizes grounding and auditable evidence, not fluent rewriting. Transcript
punctuation, proper nouns, and each platform's tone still need creator review. The semantic trend
multiplier is only available after the existing audience-performance gate has enough comparable
Shorts and saved transcript embeddings. A future generative layer can use these extractive outputs
as citations and be evaluated against the saved creator edits.
