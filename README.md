# CreatorCut

CreatorCut is a multimodal ML system that turns long-form creator videos into ranked short clips, thumbnail candidates, captions, and platform-specific posts.

Rather than treating clip selection as an opaque generative-AI task, CreatorCut frames it as an explainable multimodal learning-to-rank problem. Candidate clips are evaluated using transcript, audio, visual, and structural signals, and the interface shows why each recommendation was made.

## Current product workflow

1. Upload a long-form MP4, MOV, M4V, or WebM video.
2. Transcribe it with word timestamps and generate sentence-aligned candidates.
3. Generate candidate short-form clips.
4. Rank candidates using an explainable ML model.
5. Preview and adjust the three recommended intervals.
6. Export the source aspect ratio, a vertical 9:16 crop, or a vertical clip with burned captions.
7. Capture editorial decisions and optional YouTube analytics for bounded feature and semantic
   personalization.

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

The product surface accepts an uploaded video, runs local transcription and frozen-model ranking,
returns three non-duplicative recommendations, exposes global and adaptive score evidence, and
creates frame-accurate MP4 downloads. A creator may adjust either boundary by up to 15 seconds,
choose source-ratio or 9:16 captioned export, reject a recommendation, or define a completely custom
interval. Finishing a review explicitly records untouched model options as weak unselected evidence;
ordinary missing clicks stay unlabeled. The SQLite store keeps presentations, choices, exact edits,
custom intervals, model lineage, and post-publication outcomes as separate records. YouTube Studio
ZIP/CSV exports can be attached to the source video or a published Short; insufficient reports are
saved without influencing ranking. After enough comparable Shorts, the product learns both
interpretable quality preferences and similarity to semantically related high-performing clips,
then displays an evidence-derived summary and recurring stronger/weaker terms. See
[docs/personalization-and-feedback.md](docs/personalization-and-feedback.md) and
[docs/youtube-analytics-feedback.md](docs/youtube-analytics-feedback.md). Runtime data contracts,
event semantics, model lineage, and the local-to-hosted boundary are documented in
[docs/ml-deployment-design.md](docs/ml-deployment-design.md).

Export a versioned, integrity-hashed offline snapshot of the local feedback data:

```bash
creatorcut-export-feedback
```

The ignored JSONL snapshot includes impressions, explicit choices, custom clips, timestamp deltas,
model lineage, semantic representations, and the latest audience outcome without exposing media
paths.

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
