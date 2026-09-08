"""Grounded platform copy using TF-IDF topics and creator performance trends."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from creatorcut.baseline import tokenize

REPURPOSING_VERSION = "creatorcut_extractive_repurposing_v1"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "all",
    "any",
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "our",
    "because",
    "been",
    "before",
    "being",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "for",
    "from",
    "had",
    "has",
    "have",
    "here",
    "how",
    "into",
    "its",
    "just",
    "like",
    "more",
    "not",
    "now",
    "really",
    "that",
    "the",
    "than",
    "their",
    "then",
    "there",
    "they",
    "this",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def _terms(text: str) -> list[str]:
    tokens = [token for token in tokenize(text) if len(token) >= 3 and token not in STOPWORDS]
    return tokens + [
        f"{left} {right}" for left, right in zip(tokens, tokens[1:], strict=False)
    ]


def extract_topics(
    text: str,
    creator_documents: list[str],
    positive_trends: list[dict[str, Any]] | None = None,
    negative_trends: list[dict[str, Any]] | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Rank transcript terms with local TF-IDF and bounded audience-trend boosts."""
    terms = _terms(text)
    if not terms:
        return []
    counts = Counter(terms)
    documents = creator_documents or [text]
    document_terms = [set(_terms(document)) for document in documents]
    positive = {str(item["term"]).casefold() for item in positive_trends or []}
    negative = {str(item["term"]).casefold() for item in negative_trends or []}
    scored: list[tuple[str, float, int, bool, bool]] = []
    for term, frequency in counts.items():
        document_frequency = sum(term in known for known in document_terms)
        inverse_document_frequency = math.log(
            (len(document_terms) + 1) / (document_frequency + 1)
        ) + 1
        audience_multiplier = 1.35 if term in positive else 0.75 if term in negative else 1.0
        length_bonus = 1.15 if " " in term else 1.0
        score = frequency * inverse_document_frequency * audience_multiplier * length_bonus
        scored.append(
            (term, score, document_frequency, term in positive, term in negative)
        )
    selected: list[dict[str, Any]] = []
    used_words: set[str] = set()
    for term, score, document_frequency, positive_match, negative_match in sorted(
        scored, key=lambda item: (-item[1], -len(item[0]), item[0])
    ):
        words = set(term.split())
        if words <= used_words:
            continue
        selected.append(
            {
                "term": term,
                "score": round(score, 4),
                "document_frequency": document_frequency,
                "positive_audience_trend": positive_match,
                "negative_audience_trend": negative_match,
            }
        )
        used_words.update(words)
        if len(selected) == limit:
            break
    return selected


def _truncate(text: str, limit: int) -> str:
    clean = " ".join(text.split()).strip()
    if len(clean) <= limit:
        return clean
    shortened = clean[: max(1, limit - 1)].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return (shortened or clean[: limit - 1]).rstrip() + "…"


def _hashtag(term: str) -> str | None:
    words = re.findall(r"[a-zA-Z0-9]+", term)
    if not words or all(word.isdigit() for word in words):
        return None
    value = "".join(word[:1].upper() + word[1:].lower() for word in words)
    return f"#{value[:40]}" if value else None


def _best_excerpt(text: str, topic_terms: list[str]) -> str:
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT.split(text) if sentence.strip()]
    if not sentences:
        return text.strip()
    topic_words = {word for term in topic_terms for word in term.split()}

    def score(item: tuple[int, str]) -> tuple[float, int]:
        index, sentence = item
        words = set(tokenize(sentence))
        overlap = len(words & topic_words)
        completeness = int(sentence.endswith((".", "?", "!")))
        length_fit = max(0.0, 1.0 - abs(len(sentence) - 120) / 160)
        return overlap + 0.4 * completeness + 0.2 * length_fit + 0.15 * (index == 0), -index

    return max(enumerate(sentences), key=score)[1]


def generate_platform_pack(
    clip: dict[str, Any],
    creator_documents: list[str],
    creator_summary: dict[str, Any],
) -> dict[str, Any]:
    """Generate editable, extractive posts without inventing claims outside the transcript."""
    text = str(clip["transcript_text"]).strip()
    if not text:
        raise ValueError("The clip transcript is empty")
    topics = extract_topics(
        text,
        creator_documents,
        creator_summary.get("positive_semantic_trends"),
        creator_summary.get("negative_semantic_trends"),
    )
    topic_terms = [topic["term"] for topic in topics]
    excerpt = _best_excerpt(text, topic_terms)
    title = _truncate(excerpt.rstrip("."), 70)
    hashtags = []
    for topic in topics:
        hashtag = _hashtag(topic["term"])
        if hashtag and hashtag.casefold() not in {item.casefold() for item in hashtags}:
            hashtags.append(hashtag)
        if len(hashtags) == 5:
            break
    if not hashtags:
        hashtags = ["#CreatorClip"]
    short_tags = " ".join(hashtags[:4])
    all_tags = " ".join(hashtags)
    youtube_description = _truncate(text, 420)
    instagram_body = _truncate(text, 520)
    linkedin_body = _truncate(text, 620)
    audience_matches = [
        topic["term"] for topic in topics if topic["positive_audience_trend"]
    ]
    return {
        "algorithm_version": REPURPOSING_VERSION,
        "grounding": "extractive_transcript_only",
        "personalization_active": bool(audience_matches),
        "evidence": {
            "topics": topics,
            "audience_aligned_topics": audience_matches,
            "source_clip_id": clip["id"],
            "source_model_version": clip.get("ranking_model_version"),
        },
        "platforms": {
            "youtube": {
                "label": "YouTube Shorts",
                "title": title,
                "text": f"{title}\n\n{youtube_description}\n\n{short_tags}",
            },
            "tiktok": {
                "label": "TikTok",
                "text": f"{_truncate(excerpt, 150)}\n\n{short_tags}",
            },
            "instagram": {
                "label": "Instagram Reels",
                "text": f"{instagram_body}\n\n{all_tags}",
            },
            "linkedin": {
                "label": "LinkedIn",
                "text": f"{linkedin_body}\n\n{short_tags}",
            },
        },
        "note": (
            "Copy is extracted from the transcript. Review names, punctuation, and platform tone "
            "before publishing."
        ),
    }
