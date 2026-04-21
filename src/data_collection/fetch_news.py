import json
from datetime import datetime
from pathlib import Path

import requests

from src.config import NEWS_API_KEY


def fetch_news(topic: str, page_size: int = 5) -> dict:
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY is missing. Please add it to your .env file.")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def save_articles(data: dict, topic: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = topic.replace(" ", "_").lower()
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{safe_topic}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return str(output_path)