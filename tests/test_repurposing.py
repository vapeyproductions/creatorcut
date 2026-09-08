from creatorcut.repurposing import extract_topics, generate_platform_pack


def clip():
    return {
        "id": "clip_one",
        "transcript_text": (
            "A clear pricing strategy starts with the customer problem. "
            "The strongest pricing strategy explains value before cost."
        ),
        "ranking_model_version": "frozen_v1",
    }


def test_topic_extraction_uses_document_frequency_and_audience_trends():
    topics = extract_topics(
        clip()["transcript_text"],
        [clip()["transcript_text"], "Another discussion about pricing."],
        positive_trends=[{"term": "pricing strategy"}],
    )

    assert topics[0]["term"] == "pricing strategy"
    assert topics[0]["positive_audience_trend"] is True


def test_platform_pack_is_grounded_and_contains_editable_platform_outputs():
    value = generate_platform_pack(
        clip(),
        [clip()["transcript_text"]],
        {
            "positive_semantic_trends": [{"term": "pricing strategy"}],
            "negative_semantic_trends": [],
        },
    )

    assert value["grounding"] == "extractive_transcript_only"
    assert value["personalization_active"] is True
    assert set(value["platforms"]) == {"youtube", "tiktok", "instagram", "linkedin"}
    assert value["platforms"]["youtube"]["title"]
    assert "pricing" in value["platforms"]["tiktok"]["text"].lower()
