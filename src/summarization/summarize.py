from transformers import pipeline

_summarizer = None

def get_summarizer():
    global _summarizer

    if _summarizer is None:
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            framework="pt"
        )

    return _summarizer


def chunk_text(text: str, max_words: int = 400):
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

        result = summarizer(
            chunk,
            max_length=130,
            min_length=40,
            do_sample=False
        )

        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)