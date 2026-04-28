import os
import torch
from transformers import pipeline

_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        is_cloud = (
            os.getenv("IS_STREAMLIT_CLOUD") == "true"
            or os.getenv("STREAMLIT_SHARING_MODE") is not None
        )
        if is_cloud:
            model = "sshleifer/distilbart-cnn-12-6"
            _summarizer = pipeline(
                "summarization",
                model=model,
                framework="pt",
                device=-1,  # force CPU
                torch_dtype=torch.float32,
            )
        else:
            model = "facebook/bart-large-cnn"
            _summarizer = pipeline(
                "summarization",
                model=model,
                framework="pt",
            )
        print(f"Loaded summarization model: {model}")
    return _summarizer


def chunk_text(text: str, max_words: int = 300):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


def summarize_article(text: str) -> str:
    if not text:
        return ""
    if len(text.split()) < 40:
        return text
    summarizer = get_summarizer()
    chunks = list(chunk_text(text))
    summaries = []
    for chunk in chunks:
        if len(chunk.split()) < 40:
            continue
        word_count = len(chunk.split())
        max_len = min(80, max(30, word_count // 2))
        min_len = min(20, max_len - 10)
        try:
            result = summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
            summaries.append(result[0]["summary_text"])
        except Exception as e:
            print(f"Summarization error: {e}")
            summaries.append(chunk[:200])
    return " ".join(summaries)