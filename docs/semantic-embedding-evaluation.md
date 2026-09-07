# Semantic transcript embedding evaluation

## Question

Do pretrained semantic transcript representations improve clip ranking beyond the first
handcrafted-feature model?

The experiment compares three representations while holding the model class, labels, folds,
and evaluation metrics constant. This isolates the value of the representation itself.

## Representation

CreatorCut uses the frozen
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
encoder at pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The model was
pretrained and contrastively fine-tuned on external sentence pairs; CreatorCut does not
fine-tune its 22.7 million parameters on this small dataset.

The encoder converts each candidate transcript into a 384-dimensional dense vector. Inference
uses the model's official quantized ARM64 ONNX export, attention-mask-aware mean pooling, a
maximum length of 256 wordpieces, and L2 normalization. ONNX Runtime avoids adding a full
PyTorch installation to the application environment.

This is transfer learning: the frozen transformer supplies general semantic features, while a
small supervised model learns how those features relate to this reviewer's clip preferences.

## Controlled ablation

All variants use the same multi-output ridge regression with `alpha=10`:

1. **Handcrafted:** the existing 15 transcript-structure and timing features.
2. **Semantic:** only the 384 MiniLM embedding dimensions.
3. **Hybrid:** all 399 semantic and handcrafted features.

The regularization value was fixed before comparing representations. All reported predictions
come from the same four folds grouped by `video_id`; scaling and ridge fitting use only each
fold's training videos. Computing embeddings for all clips does not leak labels because the
encoder is frozen and was trained externally.

## Results

| Representation | Quality MAE ↓ | Quality RMSE ↓ | Spearman ↑ | Pairwise accuracy ↑ | Top-1 hit ↑ | Top-1 regret ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Fold training mean | 1.045 | 1.238 | -0.323 | 0.500 | 0.125 | 1.688 |
| Handcrafted ridge | 1.074 | 1.275 | 0.120 | 0.557 | 0.188 | 1.266 |
| Semantic ridge | 1.043 | 1.250 | 0.234 | 0.588 | 0.438 | 0.625 |
| Hybrid ridge | **1.028** | **1.229** | **0.256** | **0.624** | **0.562** | **0.422** |

The hybrid model selects a clip tied for the highest human score in 9 of 16 unseen videos. Its
average top-selection regret is 0.42 points, compared with 1.69 for the fold-mean baseline and
1.27 for handcrafted features alone.

Per-target hybrid results show that the representation helps some judgments more than others:

| Target | MAE ↓ | Spearman ↑ |
|---|---:|---:|
| Hook | 1.385 | 0.016 |
| Completeness | 1.076 | 0.273 |
| Payoff | 1.380 | 0.156 |
| Clarity | 1.291 | 0.233 |

The clearest improvement is completeness: its correlation changes from -0.152 with handcrafted
features to 0.273 with the hybrid representation. Hook remains difficult, suggesting that the
opening's delivery and immediate context may matter more than transcript semantics alone.

## Uncertainty

A paired cluster bootstrap resamples whole videos 10,000 times and compares the hybrid model
with the handcrafted model. This preserves the dependency among six clips from the same video.

| Hybrid minus handcrafted | Estimate | 95% bootstrap interval | Probability improved |
|---|---:|---:|---:|
| Quality MAE | -0.046 | [-0.240, 0.134] | 69.1% |
| Pairwise accuracy | +0.068 | [-0.014, 0.146] | 94.4% |
| Top-1 hit rate | +0.375 | [0.062, 0.625] | 98.9% |
| Top-1 regret | -0.844 | [-1.406, -0.312] | 99.9% |

The absolute-error and pairwise intervals include zero, so those gains are not yet conclusive.
The clip-selection metrics are much more convincing: the hybrid model more often finds the best
candidate and pays a substantially smaller penalty when it misses.

## Interpretation and limitations

Semantic content and transcript structure are complementary. The pretrained embedding provides
meaning-level information that the transparent rules lack, while timing and boundary features
still add useful signal on top of it.

The evaluation contains only 16 videos and one reviewer, and it compares six sampled candidates
per video rather than every possible interval. Cross-validation provides an honest small-data
estimate, but it is not a substitute for a larger untouched test collection. The experiment also
does not predict the separate technical-exportability gate.

The most targeted follow-up is to encode the sentence immediately before and after each clip as
separate boundary-context features. That tests whether missing context explains the remaining
completeness and hook errors without changing the successful evaluation protocol.

## Reproduce locally

Install the lightweight ONNX embedding dependencies, build the supervised rows, and run the
ablation:

```bash
python -m pip install -e '.[semantic]'
creatorcut-build-training-data
creatorcut-encode-transcripts
creatorcut-evaluate-semantic
```

The model revision and ONNX filename are recorded in the embedding artifact. Embeddings, model
weights, fitted coefficients, predictions, and review-derived data remain ignored by Git.
