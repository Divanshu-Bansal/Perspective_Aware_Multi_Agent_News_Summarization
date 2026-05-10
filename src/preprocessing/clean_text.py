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