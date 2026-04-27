# src/preprocessing/clean_text.py
import re

def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove NewsAPI's truncation marker e.g. "[+2145 chars]"
    text = re.sub(r"\[\+\d+\s*chars?\]", "", text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove HTML tags and common artifacts
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\b(li|ul|ol|href|span|div|class|html)\b", " ", text)

    # Remove non-printable/special characters but keep punctuation
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()