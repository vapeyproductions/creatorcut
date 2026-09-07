# CreatorCut

CreatorCut is a multimodal ML system that turns long-form creator videos into ranked short clips, thumbnail candidates, captions, and platform-specific posts.

Rather than treating clip selection as an opaque generative-AI task, CreatorCut frames it as an explainable multimodal learning-to-rank problem. Candidate clips are evaluated using transcript, audio, visual, and structural signals, and the interface shows why each recommendation was made.

## Planned MVP

1. Upload a short MP4 video.
2. Transcribe it and identify topic boundaries.
3. Generate candidate short-form clips.
4. Rank candidates using an explainable ML model.
5. Recommend thumbnail frames.
6. Generate grounded, platform-specific publishing copy.
7. Preview and export the resulting content package.

## ML system

- Human-annotated clip-quality and pairwise-preference data
- Text, audio, visual, and structural feature pipelines
- Heuristic and learning-to-rank baselines
- Leakage-safe evaluation and modality ablations
- Model explanations and systematic failure analysis
- Asynchronous inference served through a web application

## Proposed stack

- Python, FastAPI, FFmpeg
- Whisper-compatible speech recognition
- Sentence and vision-language embeddings
- LightGBM ranking model
- Next.js web interface
- SQLite initially, with object storage added when deployment requires it

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

Raw videos, model weights, and generated transcripts are intentionally excluded from Git.
