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

CreatorCut currently includes a validated seed annotation set, timestamped transcription,
and a pipeline that joins human-scored clip intervals to their transcript text.

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

Transcribe every local source video with word-level timestamps:

```bash
creatorcut-transcribe
```

Join transcript text to the human-labeled intervals:

```bash
creatorcut-build-dataset
```

Run the automated checks:

```bash
pytest -q
ruff check src tests
```

Raw videos, model weights, and generated transcripts are intentionally excluded from Git.
