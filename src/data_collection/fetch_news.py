# src/data_collection/fetch_news.py
import json
from datetime import datetime
from pathlib import Path
import requests
from src.config import NEWS_API_KEY


def relevance_score(article: dict, topic: str) -> float:
    """
    Returns a float 0.0–1.0 representing how relevant an article is to the topic.
    Weights title matches more heavily than description matches.
    """
    title = (article.get("title") or "").lower()
    description = (article.get("description") or "").lower()
    content = (article.get("content") or "").lower()

    topic_words = [w for w in topic.lower().split() if len(w) > 2]
    if not topic_words:
        return 0.0

    title_matches = sum(1 for w in topic_words if w in title)
    desc_matches = sum(1 for w in topic_words if w in description)
    content_matches = sum(1 for w in topic_words if w in content)

    # Weighted score: title counts 3x, description 2x, content 1x
    raw = (title_matches * 3 + desc_matches * 2 + content_matches * 1)
    max_possible = len(topic_words) * 6  # if all words match title + desc
    score = min(raw / max_possible, 1.0)
    return round(score, 3)


def fetch_news(topic: str, page_size: int = 20) -> dict:
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY is missing. Please add it to your .env file.")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError("NewsAPI request timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"NewsAPI returned an error: {e}")

    data = response.json()

    filtered_articles = []
    for article in data.get("articles", []):
        content = article.get("content") or article.get("description") or ""
        if len(content) < 150:
            continue

        score = relevance_score(article, topic)
        if score < 0.15:   # minimum relevance threshold
            continue

        article["_relevance_score"] = score
        filtered_articles.append(article)

    # Sort by relevance score descending
    filtered_articles.sort(key=lambda a: a["_relevance_score"], reverse=True)
    data["articles"] = filtered_articles
    return data


def save_articles(data: dict, topic: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = topic.replace(" ", "_").lower()
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_topic}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(output_path)