# src/preprocessing/clean_text.py
import re

def clean_text(text: str) -> str:
    """
    Cleans raw text data for downstream NLP processing.

    Cleaning steps include:
    - Removing API-specific truncation markers
    - Removing URLs
    - Stripping HTML tags and common artifacts
    - Removing unwanted special characters
    - Normalizing whitespace

    Args:
        text (str): Raw input text

    Returns:
        str: Cleaned and normalized text
    """

    # Handle null/empty input safely
    if not text:
        return ""

    # ─────────────────────────────────────────────────────────────
    # Remove API-specific artifacts (e.g., "[+2145 chars]")
    # These appear in truncated responses from NewsAPI
    # ─────────────────────────────────────────────────────────────
    text = re.sub(r"\[\+\d+\s*chars?\]", "", text)

    # ─────────────────────────────────────────────────────────────
    # Remove URLs (not useful for NLP tasks)
    # ─────────────────────────────────────────────────────────────
    text = re.sub(r"http\S+", "", text)

    # ─────────────────────────────────────────────────────────────
    # Remove HTML tags and common markup artifacts
    # Helps clean scraped or API-returned content
    # ─────────────────────────────────────────────────────────────
    text = re.sub(r"<[^>]+>", "", text)
    # Remove leftover HTML-related keywords
    text = re.sub(r"\b(li|ul|ol|href|span|div|class|html)\b", " ", text)

    # ─────────────────────────────────────────────────────────────
    # Remove non-printable/special characters
    # Keep basic punctuation for sentence structure
    # ─────────────────────────────────────────────────────────────
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)

    # ─────────────────────────────────────────────────────────────
    # Normalize whitespace (remove extra spaces, newlines, tabs)
    # ─────────────────────────────────────────────────────────────
    text = re.sub(r"\s+", " ", text)

    return text.strip()