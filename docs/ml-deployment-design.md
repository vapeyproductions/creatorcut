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
  -> versioned publishability gate
  -> frozen transcript embedding + handcrafted features
  -> frozen global quality predictions
  -> editorial + performance + semantic + source-retention adjustments
  -> diverse top-three selection
  -> preview / edit / subject-aware export / reject / custom alternative
  -> transcript-grounded platform posts + exact copy feedback
  -> optional YouTube outcome import
```

Processing runs through a persistent job queue so multiple uploads cannot compete for the local ASR
and ONNX resources. Each video exposes a durable state (`queued`, `validating`, `transcribing`,
`generating_candidates`, `ranking_candidates`, `ready`, or `failed`) through the API. Failures are
truncated and persisted instead of disappearing in a background thread.

Each upload and queue record are committed together before the web request returns. A worker claims
the oldest available job atomically, renews an expiring lease while inference runs, and records an
attempt count. If the process disappears, another worker recovers the expired lease. Failed work is
retried with bounded backoff and becomes a visible terminal failure after three attempts. The normal
local command embeds the same worker loop for convenience; production mode separates the stateless
web server from the resource-heavy ML worker.

The repository includes a two-service container topology with shared persistent volumes, a non-root
runtime image, liveness and readiness endpoints, structured JSON logs, and CI checks for lint, tests,
and container construction. Readiness verifies the serving database and frozen model artifact;
queue counters expose queued, running, succeeded, and failed work without revealing creator data.

## In-product ML operations report

The website exposes a per-creator report backed by live operational records rather than demo
numbers. It combines video and persistent-job states, model-version lineage, presented and selected
clip counts, custom alternatives, explicit rejects, edited-boundary magnitude, analytics coverage,
adaptive-layer gates, observed adjustment magnitudes, and the latest feedback events. This gives a
reviewer a trace from model impression to human decision to audience outcome while keeping media
paths and transcript embeddings private.

The report deliberately labels the global ranker as frozen and shows inactive adaptation layers as
waiting for evidence. A missing denominator stays unavailable; it is not rendered as zero success.
It also exposes the verified serving-release ID, component versions, and model-artifact digest. A
creator can download the same integrity-hashed feedback snapshot used at the offline-training
boundary directly from the report.

An opt-in local administrator observatory aggregates model impressions, choices by displayed rank,
clip/source duration distributions, boundary corrections, file and export formats, analytics-field
coverage, processing jobs, and per-account denominators. It keeps frozen offline evaluation
separate from live behavioral signals and omits transcripts, media paths, and raw analytics rows.
The route is disabled by default because a public version requires administrator authentication.

## Serving release integrity

`models/serving_release_v1.json` is the production release contract. Readiness verifies the exact
global-ranker SHA-256, ranker schema, publishability rules, reframing schema, repurposing algorithm,
personalization evidence thresholds, and adjustment caps. Inference checks the contract again
before using the frozen ranker. Editing a model artifact or policy constant in place therefore
causes an explicit invalid release instead of an untracked behavior change.

## Publishability before quality

Candidate eligibility is separate from learned quality. A narrow deterministic ruleset blocks
high-confidence sponsor/promo reads and music-only intervals before embedding. Review-level context,
boundary, intro/outro, duration, speech-density, and filler signals contribute a visible penalty
capped at 0.35. The ruleset version, reasons, raw signals, and applied adjustment are persisted with
each clip and exported for future supervised evaluation. The score is not described as a learned
probability because the current ratings do not contain a balanced publishability target.

## Reframing lineage

Vertical rendering samples face detections, tracks a stable subject target, and smooths the crop
trajectory. Center crop remains the deterministic fallback. Export feedback stores the actual mode,
detector name, detection coverage, multi-face count, and smoothing configuration so later outcome
analysis does not mix face-tracked and fallback treatments unknowingly.

## Durable data contracts

SQLite stores creator profiles, source-video jobs, clips, feedback events, audience-performance
reports, raw normalized analytics imports, repurposing packs, and exact copy choices. Media,
transcripts, experiment embeddings, and exports remain outside Git.

Each persisted clip records:

- source video and exact start/end/duration;
- `model` or `creator` origin;
- transcript and frozen semantic embedding;
- frozen model version;
- predicted hook, completeness, payoff, and clarity;
- global candidate rank and global score;
- publishability, editorial, structured-performance, semantic-performance, and source-retention adjustments;
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

The current build is deployable on one durable host and does not pretend to be internet-scale. Its
web/worker separation, retryable leases, structured logs, health probes, container image, and CI are
operational today. A multi-host version would replace local media with object storage and lifecycle
policies, SQLite with managed Postgres, and the local creator profile with authenticated tenant
isolation. It would also add centralized latency/error dashboards, data export/deletion controls,
and model/data drift alerts.

Those infrastructure substitutions do not change the event or model contracts described above,
which is the reason they are defined separately from the local storage implementation.
