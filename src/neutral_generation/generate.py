# src/neutral_generation/generate.py
from src.summarization.summarize import summarize_article


def _topic_guard(summaries: list[dict], topic: str, min_score: float = 0.1) -> list[dict]:
    topic_words = [w for w in topic.lower().split() if len(w) > 3]
    if not topic_words:
        return summaries
    kept = []
    for item in summaries:
        text = (item.get("summary") or "").lower()
        matches = sum(1 for w in topic_words if w in text)
        if (matches / len(topic_words)) >= min_score:
            kept.append(item)
    return kept if kept else summaries


def _bias_warning(perspectives: list[dict]) -> str | None:
    """
    Flags if coverage is heavily skewed toward one perspective type.
    """
    if not perspectives:
        return None
    counts = {}
    for p in perspectives:
        ptype = p.get("perspective", "General")
        counts[ptype] = counts.get(ptype, 0) + 1
    total = len(perspectives)
    for ptype, count in counts.items():
        if count / total >= 0.80 and total >= 3:
            return (f"Coverage bias detected: {count}/{total} sources share a "
                    f"{ptype} perspective. Consider searching for more diverse sources.")
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

    if topic:
        valid_summaries = _topic_guard(valid_summaries, topic)

    common_themes = []
    perspectives = []
    similarity_method = "tf-idf"
    perspective_diversity = 1

    if comparison_result:
        common_themes = comparison_result.get("common_themes", [])
        perspectives = comparison_result.get("perspectives", [])
        similarity_method = comparison_result.get("similarity_method", "tf-idf")
        perspective_diversity = comparison_result.get("perspective_diversity", 1)

    # Sort by relevance score so highest quality articles dominate the summary
    valid_summaries_sorted = sorted(
        valid_summaries,
        key=lambda x: x.get("relevance_score", 0),
        reverse=True
    )
    # Give each source equal weight — truncate each summary to 300 chars
    # so no single article dominates the final BART pass
    source_texts = []
    for item in valid_summaries_sorted[:5]:
        summary = item.get("summary", "").replace("\n", " ").strip()
        source_texts.append(summary[:300])
    combined_text = " ".join(source_texts)[:2500]

    if len(combined_text.split()) > 60:
        neutral_summary = summarize_article(combined_text)
    else:
        neutral_summary = combined_text

    # ── Build structured output ──────────────────────────────────────────
    divider = "─" * 50

    output = []
    output.append(f"{'=' * 50}")
    output.append(f"  NEUTRAL SUMMARY")
    if topic:
        output.append(f"  Topic: {topic}")
    output.append(f"{'=' * 50}")
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
            source = item.get("source", "Unknown")
            perspective = item.get("perspective", "General")
            title = item.get("title", "")
            output.append(f"  [{perspective}] {source}")
            if title:
                output.append(f"    └─ {title}")
        output.append(f"\n  Perspective diversity score: {perspective_diversity} unique viewpoint(s)")
        output.append(f"  Similarity method used: {similarity_method}")
        output.append("")

    output.append(divider)
    output.append("  SOURCE CONTRIBUTIONS")
    output.append(divider)
    for item in valid_summaries[:5]:
        source = item.get("source", "Unknown")
        score = item.get("relevance_score", 0)
        url = item.get("url", "")
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        output.append(f"  {source}")
        output.append(f"    Relevance: {bar} {score:.2f}")
        if url:
            output.append(f"    URL: {url}")
    output.append("")

    # Bias warning if applicable
    bias_msg = _bias_warning(perspectives)
    if bias_msg:
        output.append(divider)
        output.append(f"  ⚠  {bias_msg}")
        output.append("")

    output.append("=" * 50)

    return "\n".join(output)