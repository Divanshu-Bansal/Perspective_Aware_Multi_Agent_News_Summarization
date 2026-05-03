# src/neutral_generation/generate.py
import re


def _topic_guard(summaries: list[dict], topic: str, min_score: float = 0.1) -> list[dict]:
    """
    Filters out summaries that are weakly related to the given topic.

    Strategy:
    - Extract topic keywords
    - Keep summaries where a minimum fraction of keywords appear

    Args:
        summaries (list[dict]): List of summary objects
        topic (str): User query/topic
        min_score (float): Minimum fraction of keyword matches required

    Returns:
        list[dict]: Filtered summaries
    """
    topic_words = [w for w in topic.lower().split() if len(w) > 2]

    # If no meaningful topic words, return all summaries
    if not topic_words:
        return summaries
    
    kept = []

    for item in summaries:
        text = (item.get("summary") or "").lower()

        # Count keyword matches
        matches = sum(1 for w in topic_words if w in text)

        # Keep summary if it meets threshold
        if (matches / len(topic_words)) >= min_score:
            kept.append(item)

    # Fallback: return original if filtering removes everything
    return kept if kept else summaries


def _clean_sentence(sentence: str) -> str:
    """
    Cleans a sentence for display.

    Removes:
    - Truncation artifacts (e.g., "[+123 chars]")
    - Incomplete trailing fragments

    Returns:
        str: Cleaned sentence
    """
    sentence = sentence.strip()

    # Remove truncation artifacts
    sentence = re.sub(r'\[\+\d+\s*chars?\]', '', sentence)

    # If sentence ends abruptly, trim to last complete punctuation
    if sentence and sentence[-1] not in '.!?':
        last_punct = max(
            sentence.rfind('.'),
            sentence.rfind('!'),
            sentence.rfind('?'),
        )

        # Only trim if meaningful portion exists
        if last_punct > len(sentence) // 2:
            sentence = sentence[:last_punct + 1]
            
    return sentence.strip()


def _extract_best_sentences(summaries: list[dict], topic: str, max_sentences: int = 4) -> str:
    """
    Extracts high-quality sentences from summaries to form a neutral summary.

    Strategy:
    1. Sort articles by relevance score
    2. Extract clean sentences
    3. Filter noise (length, truncation, metadata)
    4. Rank by topic relevance + article relevance
    5. Select top N sentences

    Args:
        summaries (list[dict]): Input summaries
        topic (str): Topic for relevance scoring
        max_sentences (int): Max number of sentences in final output

    Returns:
        str: Combined neutral summary
    """
    topic_words = set(w.lower() for w in topic.split() if len(w) > 2)

    # Prioritize summaries by relevance score
    sorted_summaries = sorted(
        summaries,
        key=lambda x: x.get("relevance_score", 0),
        reverse=True
    )

    collected_sentences = []
    seen_content = set()  # Used for deduplication

    # Process top N summaries only (performance optimization)
    for item in sorted_summaries[:5]:
        summary = (item.get("summary") or "").strip()
        if not summary:
            continue

        # Split summary into sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+', summary)

        for sentence in raw_sentences:
            sentence = _clean_sentence(sentence)

            # Skip sentences that are too short or too long
            if len(sentence.split()) < 6 or len(sentence.split()) > 60:
                continue

            # Skip malformed/truncated sentences
            if re.search(r'\[\+\d+', sentence):
                continue

            # Skip UI/artifact lines (e.g., arrows, formatting symbols)
            if sentence.count('→') > 0 or sentence.count('└─') > 0:
                continue

            # Deduplicate using simple fingerprint (first few words)
            fingerprint = " ".join(sentence.lower().split()[:6])
            if fingerprint in seen_content:
                continue
            seen_content.add(fingerprint)

            # Compute topic relevance score
            sentence_words = set(sentence.lower().split())
            topic_overlap = len(sentence_words.intersection(topic_words))

            collected_sentences.append({
                "text":          sentence,
                "topic_overlap": topic_overlap,
                "relevance":     item.get("relevance_score", 0),
            })

    if not collected_sentences:
        return ""

    # Rank sentences by:
    # 1. Topic relevance
    # 2. Article relevance
    collected_sentences.sort(
        key=lambda x: (x["topic_overlap"], x["relevance"]),
        reverse=True
    )

    # Select top sentences
    best = collected_sentences[:max_sentences]

    # Re-sort for readability (natural flow)
    best.sort(key=lambda x: x["relevance"], reverse=True)

    return " ".join(s["text"] for s in best)


def _bias_warning(perspectives: list[dict]) -> str | None:
    """
    Detects if news coverage is heavily biased toward a single perspective.

    Criteria:
    - One perspective accounts for >= 80% of sources
    - At least 3 sources exist

    Returns:
        str | None: Warning message if bias detected, else None
    """
    if not perspectives:
        return None
    
    counts = {}

    # Count occurrences of each perspective
    for p in perspectives:
        ptype = p.get("perspective", "General")
        counts[ptype] = counts.get(ptype, 0) + 1

    total = len(perspectives)

    # Check dominance threshold
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
    """
    Main function to generate a structured, neutral summary from multiple sources.

    Pipeline:
    1. Validate input summaries
    2. Apply topic filtering
    3. Extract key sentences
    4. Integrate themes and perspectives
    5. Format final structured output

    Args:
        source_summaries (list[dict]): Input summaries from different sources
        comparison_result (dict, optional): Output from comparison module
        topic (str): User query/topic

    Returns:
        str: Formatted neutral summary report
    """

    # Validate input
    if not source_summaries:
        return "No summaries available to generate a neutral summary."

    valid_summaries = [item for item in source_summaries if item.get("summary")]
    if not valid_summaries:
        return "No valid source summaries available."

    # Apply topic relevance filtering
    if topic:
        valid_summaries = _topic_guard(valid_summaries, topic)

    # Default values
    common_themes    = []
    perspectives     = []
    similarity_method    = "tf-idf + cosine similarity"
    perspective_diversity = 1

    # Extract comparison metadata if available
    if comparison_result:
        common_themes         = comparison_result.get("common_themes", [])
        perspectives          = comparison_result.get("perspectives", [])
        similarity_method     = comparison_result.get("similarity_method", "tf-idf + cosine similarity")
        perspective_diversity = comparison_result.get("perspective_diversity", 1)

    # Generate neutral summary text
    neutral_summary = _extract_best_sentences(valid_summaries, topic, max_sentences=4)

    # Fallback if extraction fails
    if not neutral_summary:
        neutral_summary = valid_summaries[0].get("summary", "")[:400]

    # ─────────────────────────────────────────────────────────────
    # Build structured output (UI-friendly text format)
    # ─────────────────────────────────────────────────────────────

    divider = "─" * 50
    output  = []

    # Header
    output.append("=" * 50)
    output.append("  NEUTRAL SUMMARY")
    if topic:
        output.append(f"  Topic: {topic}")
    output.append("=" * 50)
    output.append("")

    # Summary content
    output.append(neutral_summary.strip())
    output.append("")

    # Key themes section
    if common_themes:
        output.append(divider)
        output.append("  KEY THEMES")
        output.append(divider)
        for theme in common_themes[:6]:
            output.append(f"  • {theme}")
        output.append("")

    # Perspectives section
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

    # Source contribution section
    output.append(divider)
    output.append("  SOURCE CONTRIBUTIONS")
    output.append(divider)
    for item in valid_summaries[:5]:
        source = item.get("source", "Unknown")
        score  = item.get("relevance_score", 0)
        url    = item.get("url", "")

        # Visual relevance bar (UX enhancement)
        bar    = "█" * int(score * 10) + "░" * (10 - int(score * 10))

        output.append(f"  {source}")
        output.append(f"    Relevance: {bar} {score:.2f}")
        if url:
            output.append(f"    URL: {url}")
    output.append("")

    # Bias detection warning
    bias_msg = _bias_warning(perspectives)
    if bias_msg:
        output.append(divider)
        output.append(f"  ⚠  {bias_msg}")
        output.append("")

    output.append("=" * 50)
    return "\n".join(output)