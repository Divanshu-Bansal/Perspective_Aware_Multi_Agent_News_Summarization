import os
import re
import torch
from transformers import pipeline

# Global singleton instance of the summarization pipeline
# Ensures model is loaded only once (performance optimization)
_summarizer = None

def get_summarizer():
    """
    Initializes and returns a HuggingFace summarization pipeline.

    Design considerations:
    - Uses a lightweight model in cloud environments (resource constraints)
    - Uses a larger, higher-quality model locally
    - Implements singleton pattern to avoid repeated model loading

    Returns:
        transformers.pipeline: Summarization pipeline
    """
    global _summarizer

    # Load model only once
    if _summarizer is None:

        # Detect if running in a cloud environment (e.g., Streamlit Cloud)
        is_cloud = (
            os.getenv("IS_STREAMLIT_CLOUD") == "true"
            or os.getenv("STREAMLIT_SHARING_MODE") is not None
        )
        if is_cloud:
            # Lightweight model for faster inference and lower memory usage
            model = "sshleifer/distilbart-cnn-12-6"
            
            _summarizer = pipeline(
                "summarization",
                model=model,
                framework="pt",
                device=-1,  # Force CPU (cloud environments often lack GPU)
                torch_dtype=torch.float32,
            )
        else:
            # High-quality model for local development (better summaries)
            model = "facebook/bart-large-cnn"

            _summarizer = pipeline(
                "summarization",
                model=model,
                framework="pt",
                # Automatically uses GPU if available
            )

        print(f"Loaded summarization model: {model}")

    return _summarizer


def chunk_text(text: str, max_words: int = 400):
    """
    Splits long text into smaller chunks for model processing.

    Reason:
    Transformer models have token/length limits, so large text must be split.

    Args:
        text (str): Input text
        max_words (int): Maximum words per chunk

    Yields:
        str: Text chunks
    """
    words = text.split()

    # Yield chunks of size `max_words`
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


def finish_at_sentence_boundary(text: str) -> str:
    """
    Cleans incomplete summary endings and ensures full sentence output.
    """

    if not text:
        return ""

    text = re.sub(r"\[\+\d+\s*chars?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove broken trailing fragments
    text = re.sub(r"\b[A-Za-z]{1,8}$", "", text).strip()

    # Remove trailing ellipsis
    text = re.sub(r"\.\.\.+$", ".", text).strip()

    # Find last proper sentence ending
    sentence_matches = list(re.finditer(r"[.!?]", text))

    if sentence_matches:
        text = text[:sentence_matches[-1].end()].strip()

    if text and text[-1] not in ".!?":
        text += "."

    return text


def is_probably_truncated(text: str) -> bool:
    """
    Detects API snippets that are not full articles.
    """
    if not text:
        return True

    markers = [
        "[+",
        "chars",
        "…",
        "...",
        "Listen to This Article",
        "Photo:",
    ]

    return any(marker.lower() in text.lower() for marker in markers)


def summarize_article(text: str) -> str:
    """
    Generates a summary for a given article.

    Workflow:
    1. Skip very short inputs (no need to summarize)
    2. Split text into manageable chunks
    3. Summarize each chunk individually
    4. Combine chunk summaries into final result

    Args:
        text (str): Full article text

    Returns:
        str: Final summarized text
    """

    # Handle empty input safely
    if not text:
        return ""
    
    # Skip summarization for very short text (avoid unnecessary processing)
    if len(text.split()) < 40:
        return text
    
    summarizer = get_summarizer()

    # Split text into chunks for model processing
    chunks = list(chunk_text(text))
    summaries = []
    for chunk in chunks:
        # Skip chunks that are too small to summarize effectively
        if len(chunk.split()) < 40:
            continue

        word_count = len(chunk.split())

        # Dynamically adjust summary length based on chunk size
        # Prevents overly long or too-short summaries
        max_len = min(130, max(30, word_count // 2))
        min_len = min(30, max_len - 10)

        try:
            result = summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,   # Deterministic output
                truncation=True,   # Ensure input fits model limits
            )

            summary_text = result[0]["summary_text"]

            # Remove title-like headline summaries
            summary_text = summary_text.strip()

            # Remove headline-style prefix if present
            headline_patterns = [
                r"^[A-Z][^?]+\?\s*",     # Question headline
                r"^[A-Z][^:]+:\s*",      # Colon headline
            ]

            for pattern in headline_patterns:
                summary_text = re.sub(pattern, "", summary_text).strip()

            # Skip only extremely weak summaries
            if len(summary_text.split()) < 10:
                continue

            summary_text = finish_at_sentence_boundary(summary_text)

            summaries.append(summary_text)

            # Remove accidental title duplication
            if len(summary_text.split()) < 18 and "?" in summary_text:
                continue

            summaries.append(summary_text)

        except Exception as e:
            # Fallback mechanism to prevent pipeline failure
            print(f"Summarization error: {e}")

            # Use truncated raw text as fallback
            summaries.append(finish_at_sentence_boundary(chunk[:500]))

    # Remove duplicate summary sentences
    seen = set()
    unique_sentences = []

    for sentence in re.split(r'(?<=[.!?])\s+', " ".join(summaries)):
        cleaned = sentence.strip()

        fingerprint = " ".join(cleaned.lower().split()[:8])

        if fingerprint in seen:
            continue

        seen.add(fingerprint)
        unique_sentences.append(cleaned)

    combined = " ".join(unique_sentences)

    return finish_at_sentence_boundary(combined)