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
    Removes incomplete final fragments and ensures the summary ends cleanly.
    """
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\[\+\d+\s*chars?\]", "", text).strip()
    text = re.sub(r"\s*\.\.\.\s*$", ".", text).strip()

    matches = list(re.finditer(r"[.!?]", text))
    if matches:
        last_end = matches[-1].end()
        text = text[:last_end].strip()

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

            summaries.append(result[0]["summary_text"])

        except Exception as e:
            # Fallback mechanism to prevent pipeline failure
            print(f"Summarization error: {e}")

            # Use truncated raw text as fallback
            summaries.append(finish_at_sentence_boundary(chunk[:500]))

    # Combine all chunk summaries into final output
    combined = " ".join(summaries)
    return finish_at_sentence_boundary(combined)