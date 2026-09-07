import pytest

from creatorcut.training import (
    MODEL_FEATURE_FIELDS,
    build_group_folds,
    build_reviewed_training_records,
    cross_validate_ridge,
    paired_video_bootstrap,
    ranking_metrics,
)


def candidate(candidate_id: str = "candidate_001") -> dict:
    return {
        "candidate_id": candidate_id,
        "video_id": "video_001",
        "start_seconds": 1.0,
        "end_seconds": 31.0,
        "duration_seconds": 30.0,
        "sentence_count": 3,
        "word_count": 75,
        "text": "Why does this matter? Here is a complete and useful answer.",
    }


def queue_item() -> dict:
    item = candidate()
    return {
        "annotation_id": "review_001",
        "candidate_id": item["candidate_id"],
        "video_id": item["video_id"],
        "start_seconds": item["start_seconds"],
        "end_seconds": item["end_seconds"],
        "duration_seconds": item["duration_seconds"],
        "transcript_text": item["text"],
        "sampling": {"proxy_score": 99.0},
    }


def review() -> dict:
    return {
        "annotation_id": "review_001",
        "candidate_id": "candidate_001",
        "video_id": "video_001",
        "start_seconds": 1.0,
        "end_seconds": 31.0,
        "duration_seconds": 30.0,
        "hook": 5,
        "completeness": 4,
        "payoff": 5,
        "clarity": 4,
        "technically_exportable": True,
    }


def test_build_reviewed_training_records_joins_without_proxy_leakage() -> None:
    records = build_reviewed_training_records([queue_item()], [review()], [candidate()])

    assert len(records) == 1
    assert records[0]["targets"]["quality_score"] == pytest.approx(4.5)
    assert tuple(records[0]["features"]) == MODEL_FEATURE_FIELDS
    assert "proxy_score" not in records[0]
    assert "sampling" not in records[0]


def test_build_reviewed_training_records_rejects_identity_mismatch() -> None:
    invalid_review = {**review(), "video_id": "video_999"}

    with pytest.raises(ValueError, match="disagree on video_id"):
        build_reviewed_training_records([queue_item()], [invalid_review], [candidate()])


def modeled_records(video_count: int = 8, clips_per_video: int = 3) -> list[dict]:
    records = []
    for video_index in range(video_count):
        for clip_index in range(clips_per_video):
            signal = float(clip_index + 1)
            features = {field: 0.0 for field in MODEL_FEATURE_FIELDS}
            features["hook_signal"] = signal
            features["duration_seconds"] = 20.0 + signal
            score = min(5.0, 1.0 + signal)
            records.append(
                {
                    "annotation_id": f"annotation_{video_index}_{clip_index}",
                    "video_id": f"video_{video_index:03d}",
                    "technically_exportable": True,
                    "feature_schema": "handcrafted_transcript_v1",
                    "features": features,
                    "targets": {
                        "hook": score,
                        "completeness": score,
                        "payoff": score,
                        "clarity": score,
                        "quality_score": score,
                    },
                }
            )
    return records


def test_group_folds_never_split_a_video_between_train_and_test() -> None:
    records = modeled_records()
    folds = build_group_folds(records, n_splits=4, seed=42)

    assert len(folds) == 4
    assert sorted(index for fold in folds for index in fold["test_indices"]) == list(
        range(len(records))
    )
    for fold in folds:
        assert set(fold["train_videos"]).isdisjoint(fold["test_videos"])


def test_cross_validated_ridge_learns_unseen_video_ordering() -> None:
    result = cross_validate_ridge(modeled_records(), n_splits=4, seed=42, alpha=1.0)

    assert result["models"]["ridge"]["ranking"]["pairwise_accuracy"] == pytest.approx(1.0)
    assert result["models"]["ridge"]["regression"]["quality_score"]["spearman"] == pytest.approx(
        1.0
    )
    assert len(result["predictions"]) == 24


def test_ranking_metrics_awards_tied_prediction_half_credit() -> None:
    records = [{"video_id": "video_001"}, {"video_id": "video_001"}]

    result = ranking_metrics(records, [1.0, 5.0], [3.0, 3.0])

    assert result["pairwise_accuracy"] == pytest.approx(0.5)
    assert result["comparable_pairs"] == 1


def test_paired_video_bootstrap_resamples_whole_groups() -> None:
    records = [{"video_id": f"video_{index // 3}"} for index in range(12)]
    actual = [1.0, 2.0, 3.0] * 4
    reference = [3.0, 2.0, 1.0] * 4
    challenger = actual.copy()

    result = paired_video_bootstrap(
        records, actual, reference, challenger, iterations=100, seed=42
    )

    comparisons = result["challenger_minus_reference"]
    assert comparisons["pairwise_accuracy_delta"]["estimate"] == pytest.approx(1.0)
    assert comparisons["top_1_hit_rate_delta"]["bootstrap_probability_improved"] == 1.0
