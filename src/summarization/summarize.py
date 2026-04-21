from transformers import pipeline

_summarizer = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer


def summarize_article(text: str) -> str:
    if not text:
        return ""

    if len(text.split()) < 40:
        return text

    summarizer = get_summarizer()
    result = summarizer(text, max_length=120, min_length=30, do_sample=False)
    return result[0]["summary_text"]