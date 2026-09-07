# Seed annotation schema

`seed_labels.jsonl` contains human judgments for candidate intervals from the source-video
manifest. Times are represented as numeric seconds and all quality scores use a 1–5 scale.

| Field | Meaning |
| --- | --- |
| `annotation_id` | Stable label identifier |
| `video_id` | Source-video identifier from `data/videos.json` |
| `start_seconds` | Human-selected clip start |
| `end_seconds` | Human-selected clip end |
| `hook` | Opening attention strength |
| `completeness` | Ability to stand alone without missing context |
| `payoff` | Value, insight, surprise, emotion, or resolution delivered |
| `clarity` | Understandability and concision |
| `presentation` | Audio-visual suitability |
| `technically_exportable` | Whether the interval can be rendered; not a ranking label |
| `notes` | Concise annotation rationale |

The initial professionally edited podcast videos have constant presentation and technical
exportability values. Those fields are retained for schema continuity but excluded from the
first content-ranking target.

## Sampled review queue

`creatorcut-sample-annotations` creates a local `data/processed/annotation_queue.jsonl` file.
It excludes videos that already have labels or transcript-quality warnings and samples six
candidates from each remaining video: two from each low, medium, and high
transcript-heuristic score band. Temporal spreading, duration targets, and interval-overlap
suppression reduce near-duplicate review tasks. Warned transcripts can be included deliberately
with `--include-warned-transcripts` after manual review or improved transcription.

The proxy score is used only to stratify the sample. It is not a training label, and annotation
interfaces should hide the `sampling` object from reviewers to avoid anchoring their judgments.
The empty `labels` object is filled during review, then converted into the flat annotation schema
above before model training.

`creatorcut-annotate` serves the next unfinished batch in a private local browser interface and
saves completed records to `data/processed/annotation_reviews.jsonl`. Reviewers score hook,
completeness, payoff, and clarity. `presentation` is written as 5 for this professionally edited
corpus, while the technical-export gate defaults to true unless the reviewer marks an issue.
