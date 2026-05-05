# src/neutral_generation/generate.py
import re


def _topic_guard(summaries: list[dict], topic: str, min_score: float = 0.1) -> list[dict]:
    """
    Filters out summaries that are weakly related to the given topic.
    """
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

# #TODO
# def _clean_sentence(sentence: str) -> str:
#     """
#     Cleans a sentence for display — removes truncation artifacts and incomplete fragments.
#     """
#     sentence = sentence.strip()
#     sentence = re.sub(r'\[\+\d+\s*chars?\]', '', sentence)
#     if sentence and sentence[-1] not in '.!?':
#         last_punct = max(sentence.rfind('.'), sentence.rfind('!'), sentence.rfind('?'))
#         if last_punct > len(sentence) // 2:
#             sentence = sentence[:last_punct + 1]
#     return sentence.strip()

def _clean_sentence(sentence: str) -> str:
    """
    Cleans a sentence for display — removes truncation artifacts.
    Appends ellipsis instead of cutting mid-sentence fragments.
    """
    sentence = sentence.strip()

    # Remove truncation markers like [+123 chars]
    sentence = re.sub(r'\[\+\d+\s*chars?\]', '', sentence).strip()

    # If sentence ends without punctuation, add ellipsis
    # (do NOT cut — cutting creates worse fragments)
    if sentence and sentence[-1] not in '.!?':
        sentence = sentence + "..."

    return sentence.strip()


def _extract_best_sentences(summaries: list[dict], topic: str, max_sentences: int = 4) -> str:
    """
    Extracts high-quality sentences from summaries to form a neutral summary.
    Ranks by topic relevance + article relevance score.
    """
    topic_words = set(w.lower() for w in topic.split() if len(w) > 2)
    sorted_summaries = sorted(summaries, key=lambda x: x.get("relevance_score", 0), reverse=True)
    collected_sentences = []
    seen_content = set()

    for item in sorted_summaries[:5]:
        summary = (item.get("summary") or "").strip()
        if not summary:
            continue
        raw_sentences = re.split(r'(?<=[.!?])\s+', summary)
        for sentence in raw_sentences:
            sentence = _clean_sentence(sentence)
            if len(sentence.split()) < 6 or len(sentence.split()) > 60:
                continue
            if re.search(r'\[\+\d+', sentence):
                continue
            if sentence.count('→') > 0 or sentence.count('└─') > 0:
                continue
            fingerprint = " ".join(sentence.lower().split()[:6])
            if fingerprint in seen_content:
                continue
            seen_content.add(fingerprint)
            sentence_words = set(sentence.lower().split())
            topic_overlap = len(sentence_words.intersection(topic_words))
            collected_sentences.append({
                "text":          sentence,
                "topic_overlap": topic_overlap,
                "relevance":     item.get("relevance_score", 0),
            })

    if not collected_sentences:
        return ""
    collected_sentences.sort(key=lambda x: (x["topic_overlap"], x["relevance"]), reverse=True)
    best = collected_sentences[:max_sentences]
    best.sort(key=lambda x: x["relevance"], reverse=True)
    return " ".join(s["text"] for s in best)


def _bias_warning(perspectives: list[dict]) -> str | None:
    """
    Detects if news coverage is heavily biased toward a single perspective (>=80%, >=3 sources).
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
            return (
                f"Coverage bias detected: {count}/{total} sources share a "
                f"{ptype} perspective. Consider searching for more diverse sources."
            )
    return None


# ─────────────────────────────────────────────────────────────
# Biased summary generator
# ─────────────────────────────────────────────────────────────

def generate_biased_summary(
    source_summaries: list[dict],
    comparison_result: dict | None = None,
) -> dict:
    """
    Generates a biased summary from the single highest-relevance article.

    This intentionally represents what a reader gets when they rely on
    only one outlet — no aggregation, no perspective balancing.

    Args:
        source_summaries (list[dict]): Processed articles with relevance scores
        comparison_result (dict, optional): Output from compare_summaries()

    Returns:
        dict: {
            "summary":               str   — the biased summary text,
            "source":                str   — outlet name,
            "api":                   str   — API provider,
            "title":                 str   — article title,
            "url":                   str   — article URL,
            "perspective":           str   — perspective category of this source,
            "relevance":             float — relevance score,
            "missing_perspectives":  list[str] — perspectives ignored by this source,
        }
    """
    if not source_summaries:
        return {
            "summary":              "No articles available.",
            "source":               "",
            "api":                  "",
            "title":                "",
            "url":                  "",
            "perspective":          "",
            "relevance":            0.0,
            "missing_perspectives": [],
        }

    # Sort by relevance score descending
    ranked = sorted(
        source_summaries,
        key=lambda x: x.get("relevance_score", 0),
        reverse=True
    )

    # Pick the best article that has a meaningful summary (at least 20 words)
    top_article = None
    for candidate in ranked:
        summary = candidate.get("summary", "")
        if len(summary.split()) >= 20:
            top_article = candidate
            break

    # Final fallback — use the top ranked article regardless of summary length
    if not top_article:
        top_article = ranked[0]

    # Clean up the summary text
    summary_text = top_article.get("summary", "")

    # Remove NewsAPI truncation markers like [+2145 chars]
    summary_text = re.sub(r'\[\+\d+\s*chars?\]', '', summary_text).strip()

    # Remove trailing ellipsis artifacts
    summary_text = re.sub(r'\s*\.\.\.\s*$', '.', summary_text).strip()

    # Remove any double spaces created by cleanup
    summary_text = re.sub(r'\s+', ' ', summary_text).strip()

    # Ensure summary ends with proper punctuation
    if summary_text and summary_text[-1] not in '.!?':
        summary_text = summary_text + '.'

    # Determine which perspectives this single source misses
    all_perspectives = set()
    if comparison_result:
        for p in comparison_result.get("perspectives", []):
            ptype = p.get("perspective", "General")
            if ptype:
                all_perspectives.add(ptype)

    top_perspective = top_article.get("perspective", "General")

    # Missing = all perspectives in the full set minus the one this source covers
    missing = sorted(all_perspectives - {top_perspective})

    return {
        "summary":              summary_text,
        "source":               top_article.get("source", "Unknown"),
        "api":                  top_article.get("api", "Unknown"),
        "title":                top_article.get("title", ""),
        "url":                  top_article.get("url", ""),
        "perspective":          top_perspective,
        "relevance":            top_article.get("relevance_score", 0.0),
        "missing_perspectives": missing,
    }

# ─────────────────────────────────────────────────────────────
# UNCHANGED: Neutral summary generator
# ─────────────────────────────────────────────────────────────

def generate_neutral_summary(
    source_summaries: list[dict],
    comparison_result: dict | None = None,
    topic: str = ""
) -> str:
    """
    Generates a structured, neutral summary aggregated from multiple sources.
    """
    if not source_summaries:
        return "No summaries available to generate a neutral summary."

    valid_summaries = [item for item in source_summaries if item.get("summary")]
    if not valid_summaries:
        return "No valid source summaries available."

    if topic:
        valid_summaries = _topic_guard(valid_summaries, topic)

    common_themes         = []
    perspectives          = []
    similarity_method     = "tf-idf + cosine similarity"
    perspective_diversity = 1

    if comparison_result:
        common_themes         = comparison_result.get("common_themes", [])
        perspectives          = comparison_result.get("perspectives", [])
        similarity_method     = comparison_result.get("similarity_method", "tf-idf + cosine similarity")
        perspective_diversity = comparison_result.get("perspective_diversity", 1)

    neutral_summary = _extract_best_sentences(valid_summaries, topic, max_sentences=4)
    if not neutral_summary:
        neutral_summary = valid_summaries[0].get("summary", "")[:400]

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
        output.append(f"\n  Perspective diversity score: {perspective_diversity} unique viewpoint(s)")
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