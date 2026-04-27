import json
from datetime import datetime
from pathlib import Path

import requests

from src.config import NEWS_API_KEY

def is_relevant(article: dict, topic: str) -> bool:
    title = (article.get("title") or "").lower()
    description = (article.get("description") or "").lower()

    text = f"{title} {description}"
    topic_words = topic.lower().split()

    # Require at least 2 keyword matches
    match_count = sum(1 for word in topic_words if word in text)

    return match_count >= 2


def fetch_news(topic: str, page_size: int = 10) -> dict:
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

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    filtered_articles = []

    for article in data.get("articles", []):
        content = article.get("content") or article.get("description")

        if content and len(content) > 200 and is_relevant(article, topic):
            filtered_articles.append(article)

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