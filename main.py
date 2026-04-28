# main.py
import json
import argparse
from pathlib import Path

from src.data_collection.fetch_news import fetch_news, save_articles
from src.preprocessing.clean_text import clean_text
from src.summarization.summarize import summarize_article
from src.comparison.compare import compare_summaries
from src.neutral_generation.generate import generate_neutral_summary
from src.tracking.clearml_tracker import (
    init_task,
    log_pipeline_config,
    log_fetch_metrics,
    log_article_metrics,
    log_comparison_metrics,
    log_final_summary,
    close_task,
)


EXCLUDE_KEYWORDS = [
    "celebrity", "actor", "actress", "movie", "film",
    "hollywood", "entertainment", "reality tv", "fashion week",
    "music video", "box office", "grammy", "oscar",
]

EXCLUDE_SOURCES = [
    "yahoo entertainment", "tmz", "people", "e! news",
    "entertainment weekly", "variety", "deadline",
    "readtrung.com", "smashingapps.com", "cryptoprowl.com",
    "leadershipinseo.com", "devdiscourse",
]


def is_topically_relevant(text: str, topic: str, threshold: float = 0.15) -> bool:
    topic_words = [w for w in topic.lower().split() if len(w) > 2]
    if not topic_words:
        return True
    text_lower = text.lower()
    matches = sum(1 for w in topic_words if w in text_lower)
    return (matches / len(topic_words)) >= threshold


def run_pipeline(topic: str, page_size: int = 20, max_articles: int = 8):
    print(f"\nFetching articles for topic: '{topic}'\n")

    # ── ClearML: initialise task ──────────────────────────────────────────────
    task = init_task(topic)
    log_pipeline_config(task, topic, page_size, max_articles)

    # ── Fetch ─────────────────────────────────────────────────────────────────
    news_data  = fetch_news(topic=topic, page_size=page_size)
    saved_path = save_articles(news_data, topic)
    print(f"Raw articles saved to: {saved_path}")

    articles = news_data.get("articles", [])
    if not articles:
        print("\nNo articles returned from any source.")
        print("  - Free tier only covers the last 30 days of news")
        print("  - Try simpler keywords e.g. 'Iran Israel' instead of 'Iran Vs Israel war'")
        print("  - Check your API keys in .env")
        close_task(task)
        return

    # ── ClearML: log fetch metrics ────────────────────────────────────────────
    log_fetch_metrics(task, {
        "total_fetched":   news_data.get("total_fetched", len(articles)),
        "after_relevance": len(articles),
        "newsapi_count":   sum(1 for a in articles if a.get("api") == "NewsAPI"),
        "guardian_count":  sum(1 for a in articles if a.get("api") == "Guardian"),
        "gnews_count":     sum(1 for a in articles if a.get("api") == "GNews"),
    })

    # ── Filter + summarise ────────────────────────────────────────────────────
    source_summaries = []
    seen_sources     = set()
    skipped          = 0

    for article in articles:
        source_name  = article.get("source", "Unknown")
        content      = article.get("content") or article.get("title") or ""
        cleaned_text = clean_text(content)
        lower_text   = cleaned_text.lower()

        if any(word in lower_text for word in EXCLUDE_KEYWORDS):
            skipped += 1
            continue

        if source_name.lower() in EXCLUDE_SOURCES:
            skipped += 1
            continue

        if len(cleaned_text.split()) < 30:
            skipped += 1
            continue

        if not is_topically_relevant(cleaned_text, topic):
            skipped += 1
            continue
        
        if article.get("_relevance_score", 0) < 0.25:
            skipped += 1
            continue

        if source_name in seen_sources:
            continue
        seen_sources.add(source_name)

        summary = summarize_article(cleaned_text)
        if not summary:
            continue

        source_summaries.append({
            "source":          source_name,
            "title":           article.get("title", ""),
            "url":             article.get("url", ""),
            "summary":         summary,
            "relevance_score": article.get("_relevance_score", 0.0),
            "api":             article.get("api", "Unknown"),
        })

        if len(source_summaries) >= max_articles:
            break

    print(f"\nArticles processed: {len(source_summaries)} relevant, {skipped} skipped")

    if not source_summaries:
        print("\nNo relevant articles found after filtering. Try a broader topic.")
        close_task(task)
        return

    # ── ClearML: log article metrics ──────────────────────────────────────────
    log_article_metrics(task, source_summaries)

    print("\nFetched relevant articles:")
    for i, item in enumerate(source_summaries, start=1):
        score = item.get("relevance_score", 0)
        print(f"  {i}. [{score:.2f}] {item['source']} — {item['title']}")

    # ── Comparison ────────────────────────────────────────────────────────────
    comparison_result = compare_summaries(source_summaries, topic=topic)

    # ── ClearML: log comparison metrics ──────────────────────────────────────
    log_comparison_metrics(task, comparison_result)

    print("\nCommon Themes:")
    for theme in comparison_result["common_themes"]:
        print(f"  - {theme}")

    print("\nPerspective Categories:")
    for p in comparison_result["perspectives"]:
        print(f"  - {p['perspective']} → {p['source']}")

    print("\nSimilar article pairs:")
    if comparison_result["similar_pairs"]:
        for i, j, score in comparison_result["similar_pairs"]:
            print(f"  Articles {i+1} and {j+1} → similarity: {score:.2f}")
    else:
        print("  No strongly similar pairs found (articles cover different angles).")

    # ── Generate neutral summary ──────────────────────────────────────────────
    neutral_summary = generate_neutral_summary(
        source_summaries, comparison_result, topic=topic
    )

    # ── ClearML: log final summary ────────────────────────────────────────────
    log_final_summary(task, neutral_summary, topic)

    # ── Save output ───────────────────────────────────────────────────────────
    output_dir  = Path("outputs/summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "latest_summary.json"

    final_output = {
        "topic":            topic,
        "articles_found":   len(source_summaries),
        "source_summaries": source_summaries,
        "comparison":       comparison_result,
        "neutral_summary":  neutral_summary,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nFinal output saved to: {output_file}")

    # ── ClearML: close task ───────────────────────────────────────────────────
    close_task(task)

    print("\n================ FINAL OUTPUT ================\n")
    print(neutral_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perspective-Aware News Summarizer")
    parser.add_argument("--topic",   type=str, help="News topic to summarize")
    parser.add_argument("--sources", type=int, default=20, help="Number of articles to fetch (default: 20)")
    parser.add_argument("--max",     type=int, default=8,  help="Max articles to process (default: 8)")
    args = parser.parse_args()

    topic = args.topic or input("Enter news topic: ")
    run_pipeline(topic=topic, page_size=args.sources, max_articles=args.max)