# src/preprocessing/clean_text.py
import re

BOILERPLATE_PATTERNS = [
    r"Photo:\s*[^.。!?]+",
    r"Listen to This Article",
    r"Read more",
    r"Advertisement",
    r"Subscribe now",
    r"Sign up",
    r"All rights reserved",
    r"\b\d+\s*chars\b",
    r"\[\+\d+\s*chars?\]",
    r"Watch FIFA World Cup[^.]+",
    r"FREE and EXCLUSIVE[^.]+",
    r"LIVE[^.]+",
    r"\b[A-Z]{2,}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+[A-Z][a-z]+\s+\d{1,2}:",
    r"\bNew Delhi India,\s+[A-Z][a-z]+\s+\d{1,2}:",
    r"\bANI\b",
    r"\bPTI\b",
    r"\bReuters\b",
    r"\bAP\b",
    r"\bVMPL\b",
]

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\[\+\d+\s*chars?\]", "", text)

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\b(li|ul|ol|href|span|div|class|html)\b", " ", text)

    text = re.sub(r"[^\w\s.,!?;:'\"()%-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


##TODO

# # src/preprocessing/clean_text.py
# import re

# def clean_text(text: str) -> str:
#     """
#     Cleans raw text data for downstream NLP processing.

#     Cleaning steps include:
#     - Removing API-specific truncation markers
#     - Removing URLs
#     - Stripping HTML tags and common artifacts
#     - Removing unwanted special characters
#     - Normalizing whitespace

#     Args:
#         text (str): Raw input text

#     Returns:
#         str: Cleaned and normalized text
#     """

#     # Handle null/empty input safely
#     if not text:
#         return ""

#     # ─────────────────────────────────────────────────────────────
#     # Remove API-specific artifacts (e.g., "[+2145 chars]")
#     # These appear in truncated responses from NewsAPI
#     # ─────────────────────────────────────────────────────────────
#     text = re.sub(r"\[\+\d+\s*chars?\]", "", text)

#     # ─────────────────────────────────────────────────────────────
#     # Remove URLs (not useful for NLP tasks)
#     # ─────────────────────────────────────────────────────────────
#     text = re.sub(r"http\S+", "", text)

#     # ─────────────────────────────────────────────────────────────
#     # Remove HTML tags and common markup artifacts
#     # Helps clean scraped or API-returned content
#     # ─────────────────────────────────────────────────────────────
#     text = re.sub(r"<[^>]+>", "", text)
#     # Remove leftover HTML-related keywords
#     text = re.sub(r"\b(li|ul|ol|href|span|div|class|html)\b", " ", text)

#     # ─────────────────────────────────────────────────────────────
#     # Remove non-printable/special characters
#     # Keep basic punctuation for sentence structure
#     # ─────────────────────────────────────────────────────────────
#     text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)

#     # ─────────────────────────────────────────────────────────────
#     # Normalize whitespace (remove extra spaces, newlines, tabs)
#     # ─────────────────────────────────────────────────────────────
#     text = re.sub(r"\s+", " ", text)

#     return text.strip()