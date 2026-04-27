from src.summarization.summarize import summarize_article


def generate_neutral_summary(summaries: list[dict]) -> str:
    if not summaries:
        return "No summaries available."

    # Combine only summaries (NO PROMPT TEXT)
    combined_text = " ".join([item["summary"] for item in summaries])

    # Just summarize combined content
    neutral_summary = summarize_article(combined_text)

    return neutral_summary