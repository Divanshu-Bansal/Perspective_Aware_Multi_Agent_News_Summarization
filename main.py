import json
from pathlib import Path

from src.data_collection.fetch_news import fetch_news, save_articles
from src.preprocessing.clean_text import clean_text
from src.summarization.summarize import summarize_article
from src.comparison.compare import compare_summaries
from src.neutral_generation.generate import generate_neutral_summary


def run_pipeline(topic: str):
    print(f"Fetching articles for topic: {topic}")
    news_data = fetch_news(topic=topic, page_size=10)

    saved_path = save_articles(news_data, topic)
    print(f"Raw articles saved to: {saved_path}")

    articles = news_data.get("articles", [])
    source_summaries = []

    for article in articles:
        source_name = article.get("source", {}).get("name", "Unknown")
        content = article.get("content") or article.get("description") or article.get("title") or ""

        cleaned_text = clean_text(content)
        summary = summarize_article(cleaned_text)

        source_summaries.append({
            "source": source_name,
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "summary": summary
        })

    print("\nFetched relevant articles:")
    for i, item in enumerate(source_summaries, start=1):
        print(f"{i}. {item['source']} - {item['title']}")

    comparison_result = compare_summaries(source_summaries)

    print("\nSimilar article pairs (based on content):")
    for i, j, score in comparison_result["similar_pairs"]:
        print(f"{i+1} and {j+1} → similarity: {score:.2f}")

    # THEN generate neutral summary
    neutral_summary = generate_neutral_summary(source_summaries)

    output_dir = Path("outputs/summaries")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "latest_summary.json"
    final_output = {
        "topic": topic,
        "source_summaries": source_summaries,
        "comparison": comparison_result,
        "neutral_summary": neutral_summary,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"Final output saved to: {output_file}")
    print("\nNeutral Summary:\n")
    print(neutral_summary)


if __name__ == "__main__":
    topic = input("Enter news topic: ")
    run_pipeline(topic)