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

- Human clip-quality ratings converted into within-video preference pairs
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
