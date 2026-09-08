# ML deployment and feedback architecture

CreatorCut is intentionally implemented as more than a notebook model. The local application owns
inference, media processing, durable product state, model lineage, feedback collection, adaptive
reranking, and export. This document defines what is operational today and what would change in a
hosted deployment.

## Runtime path

```text
upload
  -> media validation
  -> word-timestamped transcription
  -> sentence-aligned candidate generation
  -> frozen transcript embedding + handcrafted features
  -> frozen global quality predictions
  -> editorial + performance + semantic + source-retention adjustments
  -> diverse top-three selection
  -> preview / edit / export / reject / custom alternative
  -> optional YouTube outcome import
```

Processing runs in a one-worker executor so multiple uploads cannot compete for the local ASR and
ONNX resources. Each video exposes a durable state (`queued`, `validating`, `transcribing`,
`generating_candidates`, `ranking_candidates`, `ready`, or `failed`) through the API. Failures are
truncated and persisted instead of disappearing in a background thread.

## Durable data contracts

SQLite stores creator profiles, source-video jobs, clips, feedback events, audience-performance
reports, and raw normalized analytics imports. Media, transcripts, experiment embeddings, and
exports remain outside Git.

Each persisted clip records:

- source video and exact start/end/duration;
- `model` or `creator` origin;
- transcript and frozen semantic embedding;
- frozen model version;
- predicted hook, completeness, payoff, and clarity;
- global candidate rank and global score;
- editorial, structured-performance, semantic-performance, and source-retention adjustments;
- final personalized score and explanation.

The schema uses additive migrations so a local database from an earlier build gains new lineage
columns without being deleted. Legacy clips receive a frozen embedding when performance is first
attached.

## Feedback semantics

| Event | Meaning | Current preference weight |
| --- | --- | ---: |
| `presented` | The model exposed the option at a known rank | Not a label |
| `download_original` | Creator selected the proposed interval | +1.00 |
| `download_edited` | Creator selected the moment and supplied corrected boundaries | +1.00 |
| `custom_created` | Creator supplied a moment absent from the ranked set | +1.00 |
| `review_closed_unselected` | Explicitly left over after a positive selection | -0.35 |
| `reject` | Creator explicitly rejected the recommendation | -1.00 |

An untouched clip is not automatically negative. The weaker unselected event is created only when
the creator presses **Finish review** after selecting or creating at least one clip. This preserves
the distinction between lack of exposure, indecision, mild preference, and explicit rejection.

Every presentation event includes the scoring components and model version that produced it. This
supports later offline reconstruction, rank-bias analysis, and comparison between model releases.

`creatorcut-export-feedback` materializes the operational records as a versioned JSONL snapshot and
writes a manifest containing record/creator/video counts, label and performance coverage, and a
canonical SHA-256 digest. The snapshot excludes local media paths and can be filtered to one creator.
This creates an inspectable boundary between the serving database and future offline experiments.

## Online adaptation

The global model remains frozen. Product feedback affects only bounded creator-specific layers:

- editorial preference: activates after three explicit clip decisions; cap ±0.35;
- structured audience preference: at least five sufficiently viewed Shorts; component cap ±0.15;
- semantic audience similarity: at least five outcome-linked embeddings; component cap ±0.15;
- combined audience adjustment: cap ±0.25;
- source-video retention: at least 100 views and ten retention points; cap ±0.20.

All audience targets are within-creator percentiles. Raw views are exposure, never the quality
label. The semantic layer uses similarity-weighted outcomes from frozen transcript embeddings; the
visible recurring terms explain the history but do not define semantic similarity.

## Transparent audience insights

Once audience adaptation is eligible, CreatorCut shows:

- a plain-language summary of the strongest hook/completeness/payoff/clarity/duration directions;
- recurring words and two-word phrases associated with stronger clips;
- recurring terms associated with weaker clips;
- the number of eligible published Shorts;
- every adjustment applied to each new recommendation.

The summary is deterministic and evidence-derived. It explicitly describes association rather than
causation, because topic, publication time, distribution, audience mix, and packaging can all
confound performance.

## Model promotion boundary

Creator-specific adjustments can update online because they are capped, isolated by creator, and
shrunk toward zero. The global model follows a separate release process: versioned features and
labels, grouped evaluation on unseen source videos, comparison with the frozen benchmark, and an
explicit promotion decision. Production analytics do not silently fine-tune the frozen ranker.

## Hosted deployment boundary

The current build is a complete local deployment, but it does not pretend to be internet-scale. A
hosted version would replace the in-process executor with a durable job queue, local media with
object storage and lifecycle policies, SQLite with managed Postgres, and the local creator profile
with authenticated tenant isolation. It would also add retryable job leases, structured logs,
latency/error dashboards, data export/deletion controls, and model/data drift alerts.

Those infrastructure substitutions do not change the event or model contracts described above,
which is the reason they are defined separately from the local storage implementation.
