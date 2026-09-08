# CreatorCut

CreatorCut is a multimodal ML system that turns long-form creator videos into ranked short clips, thumbnail candidates, captions, and platform-specific posts.

Rather than treating clip selection as an opaque generative-AI task, CreatorCut frames it as an explainable multimodal learning-to-rank problem. Candidate clips are evaluated using transcript, audio, visual, and structural signals. Detailed lineage and model diagnostics are reserved for the administrator ML observatory so the creator workflow remains focused.

## Current product workflow

1. Create a private creator account or sign in.
2. Upload a long-form MP4, MOV, M4V, or WebM video.
3. Transcribe it with word timestamps and generate sentence-aligned candidates.
4. Choose YouTube Shorts, Instagram Reels, TikTok, or any combination, then request a clip count
   or let CreatorCut estimate a non-overlapping range.
5. Rank candidates with the frozen global model, production audio/visual descriptors, and a
   separately learned creator-and-platform outcome layer.
6. Preview and adjust the ranked intervals for each platform.
7. Export the source aspect ratio, a subject-aware vertical 9:16 crop, or a vertical clip with
   burned captions.
8. Generate editable, transcript-grounded posts for four platforms.
9. Capture editorial decisions automatically and use optional YouTube, Instagram, or TikTok
   analytics for bounded feature and semantic personalization.

## ML system

- Human clip-quality ratings converted into within-video preference pairs
- Text, audio, visual, and structural feature pipelines
- Heuristic and learning-to-rank baselines
- Leakage-safe evaluation and modality ablations
- Model explanations and systematic failure analysis
- Asynchronous inference served through a web application

## Stack

- Python 3.11, a standard-library HTTP application, PyAV/FFmpeg codecs, and SQLite
- Faster-Whisper word-level speech recognition
- frozen MiniLM transcript embeddings plus audio, visual, and structural features
- grouped ridge and Bradley–Terry ranking experiments
- plain HTML, CSS, and JavaScript with no frontend framework

## Project status

CreatorCut currently includes a 20-video source manifest, media and transcript validation,
word-level timestamped transcription, a pipeline that joins human-scored intervals to
transcript text, and a sentence-aligned candidate generator evaluated against the human
selections. Transparent duration-only and transcript-heuristic rankers establish the first
ranking baselines. A completed blind-review dataset now supports a leakage-safe first learned
ranker evaluated on entirely unseen videos.

## Local development

CreatorCut requires Python 3.11.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the local product website:

```bash
creatorcut-web
```

Enable the separate local cross-account ML observatory when administering the system:

```bash
creatorcut-web --enable-admin-dashboard
```

The first account registered on a new local database in this mode becomes its administrator.
Existing installations can promote a registered account without exposing its password on the
command line:

```bash
creatorcut-account promote-admin --email creator@example.com
```

The product then links to `/admin`, where offline holdout evidence is shown separately from live
selection, rank, clip-length, boundary-edit, input-format, per-platform analytics coverage,
cross-creator learning, consent history, applied adjustments, account, and job-state signals. Both
the page and its data endpoint require an authenticated administrator.
See [docs/admin-ml-observatory.md](docs/admin-ml-observatory.md).

That command starts the web process plus a durable embedded worker for convenient local use. The
upload itself and its processing job are both committed to SQLite before the request returns. Jobs
use expiring leases, heartbeats, bounded retries, and persisted errors, so a stopped worker can
recover unfinished work instead of losing an in-memory task.

Run the local web and ML worker as separate processes:

```bash
creatorcut-web --worker-mode external
creatorcut-worker
```

The standard-library account server deliberately refuses non-loopback binding so credentials are
not exposed over plaintext HTTP. The container image defaults to the inference worker. A public web
deployment requires managed HTTPS, persistent shared storage, and a production identity provider;
that boundary is documented in [docs/account-security.md](docs/account-security.md).

`/api/health/live` reports web-process liveness. `/api/health/ready` verifies the database and frozen
serving release and reports persistent queued/running/succeeded/failed job counts. Both processes
emit one-line structured JSON events suitable for a hosted log drain. GitHub Actions runs lint,
tests, and a clean container build on every push and pull request.

Verify the immutable release contract directly:

```bash
creatorcut-verify-release
```

The release manifest pins the global-model digest, encoder revision, component versions,
personalization gates, and adjustment caps. Production inference refuses to run when that contract
does not match the artifact and runtime.

The product surface accepts an uploaded video, runs durable local transcription and frozen-model ranking,
removes high-confidence ads/music-only intervals through a versioned publishability gate, returns
platform-specific non-duplicative recommendations, and creates frame-accurate MP4 downloads.
Operational score evidence remains available to administrators. A creator may adjust either boundary by up to 15 seconds,
choose source-ratio or face-tracked 9:16 captioned export, reject a recommendation, or define a custom
interval. Finishing a review explicitly records untouched model options as weak unselected evidence;
ordinary missing clicks stay unlabeled. The SQLite store keeps presentations, choices, exact edits,
custom intervals, model lineage, and post-publication outcomes as separate records. YouTube Studio,
Instagram, and TikTok CSV/TSV/XLSX/ZIP exports can be attached to a published clip; YouTube
source-video exports can additionally provide timestamped retention evidence. Insufficient reports
are saved without influencing ranking. After enough comparable clips on one platform, the product learns both
interpretable quality preferences and similarity to semantically related high-performing clips,
then displays an evidence-derived summary and recurring stronger/weaker terms. See
[docs/personalization-and-feedback.md](docs/personalization-and-feedback.md) and
[docs/platform-aware-ranking.md](docs/platform-aware-ranking.md). Cross-creator editorial learning
is automatic and creator-balanced; shared use of uploaded audience analytics requires explicit
opt-in and is fully audited. See
[docs/community-learning-governance.md](docs/community-learning-governance.md). The safety/quality split and
its current evidence limits are documented in
[docs/publishability-gate.md](docs/publishability-gate.md). Runtime data contracts,
event semantics, model lineage, and the local-to-hosted boundary are documented in
[docs/ml-deployment-design.md](docs/ml-deployment-design.md).

Vertical exports use sampled face detection, continuity-aware target choice, temporal smoothing,
and a deterministic center-crop fallback. The exact export mode is recorded with the user event;
see [docs/subject-aware-reframing.md](docs/subject-aware-reframing.md).

Platform post packs use creator-local TF-IDF topics and, when evidence gates are met, bounded
positive audience-topic trends. Generated text, creator edits, and explicit rejections are stored
with algorithm lineage as separate future-training signals. See
[docs/content-repurposing.md](docs/content-repurposing.md).

The administrator-only evaluation harness parses a channel tracker, automatically aligns uploaded
Shorts to two reference long videos, fits a regularized reference-only channel layer, freezes
recommendations for a third video, and reveals the held-out Shorts only afterward. It is excluded
from the creator portal and every route is administrator-gated. Temporal recovery, boundary error,
and performance-rank agreement are persisted as an evaluation record. See
[docs/historical-backtest.md](docs/historical-backtest.md).

Export a versioned, integrity-hashed offline snapshot of the local feedback data:

```bash
creatorcut-export-feedback
```

The ignored JSONL snapshot includes impressions, explicit choices, custom clips, timestamp deltas,
model lineage, semantic representations, repurposing edits/rejections, and the latest audience
outcome without exposing media paths.

The administrator-only **ML observatory** makes the system visible without opening the database. It
reports serving/job states, model versions, presentation and choice counts, selection rate,
timestamp-correction magnitude, audience-data coverage, adaptive and community gates,
score-adjustment distributions, the verified serving release, and consent audit counts. Empty or
insufficient evidence is shown as such rather than converted into a success metric.

Validate the source manifest and annotations:

```bash
creatorcut-validate
```

Inspect downloaded media before inference:

```bash
creatorcut-inspect-media data/raw/video_001.mp4
```

Transcribe every local source video with word-level timestamps:

```bash
creatorcut-transcribe
```

CPU batching can accelerate corpus transcription. Voice-activity detection remains enabled by
default; for a source where VAD incorrectly removes real speech, use the explicit sequential
fallback and preserve that choice in the transcript metadata:

```bash
creatorcut-transcribe --batch-size 4
creatorcut-transcribe --video-id video_010 --force --disable-vad
creatorcut-validate-transcripts
```

Join transcript text to the human-labeled intervals:

```bash
creatorcut-build-dataset
```

Generate sentence-aligned 20–60 second candidate clips and evaluate interval recall:

```bash
creatorcut-generate-candidates
creatorcut-evaluate-candidates
creatorcut-sample-annotations
```

The current local corpus contains approximately 5.36 hours of media and 56,534 timestamped
words. Candidate generation produces 13,934 sentence-aligned intervals across 20 videos and
recovers all 17 human-selected seed intervals at temporal IoU ≥ 0.70. Generated media,
transcripts, candidate records, and evaluation artifacts remain local.

The annotation sampler excludes already labeled videos and transcript-quality warnings. The
current clean corpus creates a deterministic 96-clip review queue balanced across low, medium,
and high heuristic-score bands. Proxy scores are hidden during human review and are never
treated as ground-truth labels.

Launch the private Annotation Studio and review the next ten clips:

```bash
creatorcut-annotate --media-dir /path/to/downloaded/videos --batch-size 10
```

The local interface streams only manifest-declared videos, pauses playback at the sampled clip
boundary, hides all proxy-score metadata, and resumes after restarts. Reviews are saved atomically
to `data/processed/annotation_reviews.jsonl`, which remains excluded from Git.

Build model-ready rows from the completed reviews, then train the first supervised baseline:

```bash
creatorcut-build-training-data
creatorcut-train-ridge
```

An immutable annotation queue can also reconstruct the reviewed feature table without the full
candidate cache. This derives features only from the queued transcript and timing fields and
still excludes every sampler field:

```bash
creatorcut-build-training-data --from-queue
```

The training pipeline joins 96 blind reviews from 16 videos to versioned transcript features.
It excludes the sampler's proxy score and evaluates ridge regression with four folds grouped by
video. The initial learned model improves within-video pairwise accuracy from 50.0% to 55.7%,
but does not beat the fold-mean predictor on absolute error. This provides an honest, reproducible
threshold for semantic models to beat. See
[docs/first-learned-baseline.md](docs/first-learned-baseline.md) for the full experiment.

Encode the same candidate transcripts with a pinned MiniLM model and run a controlled
representation ablation:

```bash
python -m pip install -e '.[semantic]'
creatorcut-encode-transcripts
creatorcut-evaluate-semantic
```

The frozen 384-dimensional sentence embeddings improve pairwise accuracy to 58.8%. Combining
them with the handcrafted features reaches 62.4%, selects a top-rated clip in 9 of 16 unseen
videos, and reduces average top-selection regret from 1.69 to 0.42 rating points. A video-level
paired bootstrap reports uncertainty rather than treating the 96 correlated clips as independent.
See [docs/semantic-embedding-evaluation.md](docs/semantic-embedding-evaluation.md) for the ablation,
per-target results, and limitations.

Extract label-independent audio-delivery descriptors and run the final representation ablation:

```bash
creatorcut-extract-audio
creatorcut-evaluate-audio
```

The audio pipeline measures energy dynamics, pauses, voice activity, zero-crossing rate, pitch
variation, and clipping. Audio alone performs below the semantic models on this small edited-
podcast corpus, and adding it reduces top-1 hit rate from 56.2% to 37.5%. The predeclared selection
rule therefore keeps and freezes the text+semantic ridge model for evaluation on new videos. See
[docs/audio-feature-ablation.md](docs/audio-feature-ablation.md) for the controlled experiment and
[docs/frozen-model-v1.md](docs/frozen-model-v1.md) for the immutable test contract.

Evaluate the frozen ranker on a new, separately registered corpus:

```bash
creatorcut-encode-transcripts \
  --input data/processed/holdout_v1/annotation_queue.jsonl \
  --output data/processed/holdout_v1/semantic_embeddings.json
creatorcut-score-holdout
# Complete the blind reviews only after predictions have been sealed.
creatorcut-evaluate-holdout
```

The scorer accepts no review or label input. It verifies unseen video IDs, the exact frozen feature
and encoder schemas, and serialized v1 parameters before writing private predictions. A public
SHA-256 commitment can then prove those predictions existed before the labels. See
[docs/external-holdout-protocol.md](docs/external-holdout-protocol.md).

The Annotation Studio always records required scores for the original sealed interval. When a
candidate is nearly usable, its optional boundary editor can also capture adjusted timestamps and
a separate score set for the edited version. These secondary labels support future boundary-model
research without changing frozen v1's holdout target or metrics.

External holdout v1 is now complete: 96 blind reviews across 16 new videos were matched to the
committed predictions before evaluation. Frozen v1 reaches 58.8% pairwise accuracy, a 43.8% top-1
hit rate, and 0.828 mean top-selection regret. Its best candidate appears in the top three for 13 of
16 videos, but this exploratory result benefits from frequent ties and does not erase weak absolute
calibration. The full preregistered and exploratory analysis is reported in
[docs/external-holdout-v1-results.md](docs/external-holdout-v1-results.md).

After that frozen result was recorded, the reviewed cohort was promoted into v2 development data.
The v2 target separates a moment's attainable content quality from boundary quality: when a clip
has a separately scored edit, the moment ranker uses that edited-version score, while start/end
selection is modeled independently. Build the 192-clip, 32-video corpus and its public-safe report
with:

```bash
creatorcut-build-v2-development \
  --original-records data/processed/training_clips.jsonl \
  --completed-queue data/processed/holdout_v1/annotation_queue.jsonl \
  --completed-reviews data/processed/holdout_v1/annotation_reviews.jsonl \
  --original-embeddings data/processed/semantic_embeddings.json \
  --completed-embeddings data/processed/holdout_v1/semantic_embeddings.json \
  --output-records data/processed/v2/training_clips.jsonl \
  --output-embeddings data/processed/v2/semantic_embeddings.json \
  --output-summary data/processed/v2/development_summary.json
```

Leave-one-video-out evaluation gives a fixed equal-rank pointwise/pairwise ensemble 58.6%
pairwise accuracy, a 59.4% top-1 hit rate, and 0.586 mean top-1 regret, versus exact random
expectations of 50.0%, 46.9%, and 0.910. These are development estimates, not a replacement for
the frozen external result. See
[docs/v2-development-results.md](docs/v2-development-results.md) for the target definition,
cohort check, boundary experiment, and next-test decision.

The same completed reviews support a separate multimodal boundary-choice experiment:

```bash
creatorcut-train-boundaries \
  --queue data/processed/holdout_v1/annotation_queue.jsonl \
  --reviews data/processed/holdout_v1/annotation_reviews.jsonl \
  --transcripts-dir data/processed/holdout_v1/transcripts \
  --media-dir /path/to/downloaded/videos \
  --artifact data/processed/v2/boundary_candidates_multimodal_v1.json \
  --evaluation data/processed/v2/boundary_evaluation_v1.json \
  --model data/processed/v2/boundary_model_v1.json
```

Nearby transcript choices achieve 0.109-second start and 0.212-second end oracle MAE, but the
learned local-cue selector does not beat leaving the boundaries unchanged. It is therefore not
deployed; the result motivates semantic full-clip features for each candidate edge.

Evaluate that full-clip semantic boundary hypothesis without changing the deployed model:

```bash
creatorcut-evaluate-semantic-boundaries
```

The script re-encodes every candidate-edited interval with the exact frozen semantic model and
evaluates its boundary choices in folds grouped by source video. Its result remains a development
experiment until a boundary model beats the no-change baseline and is then confirmed on new videos.

Test whether a ranking-specific objective improves the same hybrid representation:

```bash
creatorcut-train-pairwise
```

The pairwise logistic model learns from 221 non-tied within-video preferences, but it does not
beat pointwise hybrid ridge on unseen videos: pairwise accuracy falls from 62.4% to 57.5%, top-1
hit rate falls from 56.2% to 31.2%, and mean selection regret rises from 0.42 to 0.89. CreatorCut
therefore retains the simpler pointwise model. See
[docs/pairwise-ranking-evaluation.md](docs/pairwise-ranking-evaluation.md) for the objective,
grouped evaluation, uncertainty analysis, and decision.

Generate and review the current model's out-of-fold top-selection mistakes:

```bash
creatorcut-analyze-failures
creatorcut-review-failures
```

The automatic pass isolates seven misses without exposing transcripts. Across those misses, the
human-best clip improves hook by 1.29 points on average, completeness by 1.00, clarity by 0.86,
and payoff by 0.71. Four model selections begin with a context-dependent first word, while all
seven appear complete under the current punctuation-only ending rule. The local comparison UI
collects a conditional editorial preference—which option wins after the proposed boundary edits
are applied to the model-selected clip—plus structured boundary, ad/music, duration, delivery,
hook, and payoff diagnoses in a separate ignored file without changing the original ratings.
Its dual-handle timeline supports 0.1-second nudges, immediate adjusted-interval playback, and
reopening completed reviews so timestamp labels are observed rather than guessed.

In the first manual review, the model-selected moment was preferred in six of seven failures only
after proposed edits; five of those six included an explicit timestamp correction. The remaining
case favored the comparison clip. These are therefore counterfactual editing labels, not evidence
of 6/7 raw ranking accuracy. The result motivates a two-stage design: rank promising moments, then
refine their start and end boundaries before the final editorial comparison.

Run the fixed transcript-and-pause boundary baseline:

```bash
creatorcut-refine-boundaries --media-dir /path/to/downloaded/videos
```

On the five labeled starts, the refiner reaches 16.36-second MAE versus 14.56 seconds for leaving
the original boundary unchanged; it also misses the sole labeled end. The negative result shows
that nearby punctuation and silence cannot recover semantic edits such as adding setup or removing
an intro. The implementation is retained without tuning on these six labels. See
[docs/boundary-refinement-evaluation.md](docs/boundary-refinement-evaluation.md) for the leakage
controls, results, limitations, and next model decision.

Score and evaluate the transparent ranking baselines:

```bash
creatorcut-score-baselines
creatorcut-evaluate-baselines
```

The initial transcript heuristic does not recover a known strong clip in its top 10 and has
negative rank correlation with the human scores. This is retained as an honest experimental
baseline: surface-level punctuation, keyword, and speaking-rate features do not capture the
semantic value and payoff represented in the annotations. See
[docs/baseline-evaluation.md](docs/baseline-evaluation.md) for methodology and limitations.

Run the automated checks:

```bash
pytest -q
ruff check src tests
```

Raw videos, model weights, embeddings, and generated transcripts are intentionally excluded from
Git. Reproducible experiment definitions, tests, results, and model-selection decisions are
versioned in the repository.
