# src/neutral_generation/generate.py
import re


def _topic_guard(summaries: list[dict], topic: str, min_score: float = 0.1) -> list[dict]:
    """Filter out summaries that don't mention topic words."""
    topic_words = [w for w in topic.lower().split() if len(w) > 2]
    if not topic_words:
        return summaries
    kept = []
    for item in summaries:
        text = (item.get("summary") or "").lower()
        matches = sum(1 for w in topic_words if w in text)
        if (matches / len(topic_words)) >= min_score:
            kept.append(item)
    return kept if kept else summaries


def _clean_sentence(sentence: str) -> str:
    """Clean a sentence for display."""
    sentence = sentence.strip()
    # Remove truncation artifacts
    sentence = re.sub(r'\[\+\d+\s*chars?\]', '', sentence)
    # Remove trailing incomplete words after last punctuation
    if sentence and sentence[-1] not in '.!?':
        last_punct = max(
            sentence.rfind('.'),
            sentence.rfind('!'),
            sentence.rfind('?'),
        )
        if last_punct > len(sentence) // 2:
            sentence = sentence[:last_punct + 1]
    return sentence.strip()


def _extract_best_sentences(summaries: list[dict], topic: str, max_sentences: int = 4) -> str:
    """
    Extract the best sentences from top-ranked article summaries.
    Prioritises articles by relevance score and picks complete sentences.
    """
    topic_words = set(w.lower() for w in topic.split() if len(w) > 2)

    # Sort by relevance score
    sorted_summaries = sorted(
        summaries,
        key=lambda x: x.get("relevance_score", 0),
        reverse=True
    )

    collected_sentences = []
    seen_content = set()

    for item in sorted_summaries[:5]:
        summary = (item.get("summary") or "").strip()
        if not summary:
            continue

        # Split into sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+', summary)

        for sentence in raw_sentences:
            sentence = _clean_sentence(sentence)

            # Skip too short or too long
            if len(sentence.split()) < 6 or len(sentence.split()) > 60:
                continue

            # Skip sentences with truncation artifacts
            if re.search(r'\[\+\d+', sentence):
                continue

            # Skip sentences that are just article titles or metadata
            if sentence.count('→') > 0 or sentence.count('└─') > 0:
                continue

            # Deduplicate by content fingerprint
            fingerprint = " ".join(sentence.lower().split()[:6])
            if fingerprint in seen_content:
                continue
            seen_content.add(fingerprint)

            # Score sentence by topic relevance
            sentence_words = set(sentence.lower().split())
            topic_overlap = len(sentence_words.intersection(topic_words))

            collected_sentences.append({
                "text":          sentence,
                "topic_overlap": topic_overlap,
                "relevance":     item.get("relevance_score", 0),
            })

    if not collected_sentences:
        return ""

    # Sort by topic overlap first, then by article relevance
    collected_sentences.sort(
        key=lambda x: (x["topic_overlap"], x["relevance"]),
        reverse=True
    )

    # Take top sentences
    best = collected_sentences[:max_sentences]

    # Re-sort by original order to make it read naturally
    # (just use them as selected — order by relevance score)
    best.sort(key=lambda x: x["relevance"], reverse=True)

    return " ".join(s["text"] for s in best)


def _bias_warning(perspectives: list[dict]) -> str | None:
    """Flag if coverage is heavily skewed toward one perspective."""
    if not perspectives:
        return None
    counts = {}
    for p in perspectives:
        ptype = p.get("perspective", "General")
        counts[ptype] = counts.get(ptype, 0) + 1
    total = len(perspectives)
    for ptype, count in counts.items():
        if count / total >= 0.80 and total >= 3:
            return (
                f"Coverage bias detected: {count}/{total} sources share a "
                f"{ptype} perspective. Consider searching for more diverse sources."
            )
    return None


def generate_neutral_summary(
    source_summaries: list[dict],
    comparison_result: dict | None = None,
    topic: str = ""
) -> str:
    if not source_summaries:
        return "No summaries available to generate a neutral summary."

    valid_summaries = [item for item in source_summaries if item.get("summary")]
    if not valid_summaries:
        return "No valid source summaries available."

    # Apply topic guard
    if topic:
        valid_summaries = _topic_guard(valid_summaries, topic)

    common_themes    = []
    perspectives     = []
    similarity_method    = "tf-idf + cosine similarity"
    perspective_diversity = 1

    if comparison_result:
        common_themes         = comparison_result.get("common_themes", [])
        perspectives          = comparison_result.get("perspectives", [])
        similarity_method     = comparison_result.get("similarity_method", "tf-idf + cosine similarity")
        perspective_diversity = comparison_result.get("perspective_diversity", 1)

    # Generate neutral summary from best sentences
    neutral_summary = _extract_best_sentences(valid_summaries, topic, max_sentences=4)

    if not neutral_summary:
        # Last resort fallback
        neutral_summary = valid_summaries[0].get("summary", "")[:400]

    # ── Build structured output ───────────────────────────────────────────────
    divider = "─" * 50
    output  = []

    output.append("=" * 50)
    output.append("  NEUTRAL SUMMARY")
    if topic:
        output.append(f"  Topic: {topic}")
    output.append("=" * 50)
    output.append("")
    output.append(neutral_summary.strip())
    output.append("")

    if common_themes:
        output.append(divider)
        output.append("  KEY THEMES")
        output.append(divider)
        for theme in common_themes[:6]:
            output.append(f"  • {theme}")
        output.append("")

    if perspectives:
        output.append(divider)
        output.append("  PERSPECTIVES DETECTED")
        output.append(divider)
        for item in perspectives[:8]:
            source      = item.get("source", "Unknown")
            perspective = item.get("perspective", "General")
            title       = item.get("title", "")
            output.append(f"  [{perspective}] {source}")
            if title:
                output.append(f"    └─ {title}")
        output.append(
            f"\n  Perspective diversity score: {perspective_diversity} unique viewpoint(s)"
        )
        output.append(f"  Similarity method used: {similarity_method}")
        output.append("")

    output.append(divider)
    output.append("  SOURCE CONTRIBUTIONS")
    output.append(divider)
    for item in valid_summaries[:5]:
        source = item.get("source", "Unknown")
        score  = item.get("relevance_score", 0)
        url    = item.get("url", "")
        bar    = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        output.append(f"  {source}")
        output.append(f"    Relevance: {bar} {score:.2f}")
        if url:
            output.append(f"    URL: {url}")
    output.append("")

    bias_msg = _bias_warning(perspectives)
    if bias_msg:
        output.append(divider)
        output.append(f"  ⚠  {bias_msg}")
        output.append("")

    output.append("=" * 50)
    return "\n".join(output)