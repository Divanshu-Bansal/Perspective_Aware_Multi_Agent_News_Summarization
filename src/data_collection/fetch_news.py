# src/data_collection/fetch_news.py

import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from src.config import NEWS_API_KEY, GUARDIAN_API_KEY, GNEWS_API_KEY


def _keyword_relevance_score(title: str, description: str, content: str, topic: str) -> float:
    """
    Computes a weighted relevance score for an article based on keyword matches.

    Strategy:
    - Title matches are weighted highest (most important signal)
    - Description matches are medium weight
    - Content matches are lowest weight

    Args:
        title (str): Article title
        description (str): Article description/summary
        content (str): Full article content
        topic (str): User search query

    Returns:
        float: Normalized relevance score between 0 and 1
    """

    # Extract meaningful words from topic (ignore very short words
    topic_words = [w for w in topic.lower().split() if len(w) > 2]
    if not topic_words:
        return 0.0
    
    # Normalize text inputs
    t = (title or "").lower()
    d = (description or "").lower()
    c = (content or "").lower()

    # Count keyword matches in each field
    title_matches   = sum(1 for w in topic_words if w in t)
    desc_matches    = sum(1 for w in topic_words if w in d)
    content_matches = sum(1 for w in topic_words if w in c)

    # Apply weighted scoring
    raw = (title_matches * 3 + desc_matches * 2 + content_matches)

    # Normalize score to [0, 1]
    max_possible = len(topic_words) * 6
    return round(min(raw / max_possible, 1.0), 3)


# ──────────────────────────────────────────────────────────────────────────────
# Source Fetchers (Each API normalized into a common schema)
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_newsapi(topic: str, page_size: int = 20) -> list[dict]:
    """
    Fetch articles from NewsAPI and normalize response structure.

    Returns:
        list[dict]: List of articles in unified schema
    """
    if not NEWS_API_KEY:
        print("  [NewsAPI] Key missing — skipping.")
        return []
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": topic,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": page_size,
                "apiKey": NEWS_API_KEY,
            },
            timeout=15,  # Prevent hanging requests
        )
        resp.raise_for_status()

        articles = []
        for a in resp.json().get("articles", []):
            content = a.get("content") or a.get("description") or ""

            # Filter out low-content articles (noise reduction)
            if len(content) < 150:
                continue
            articles.append({
                "source":      a.get("source", {}).get("name", "Unknown"),
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "content":     content,
                "url":         a.get("url", ""),
                "api":         "NewsAPI",
            })
        print(f"  [NewsAPI] {len(articles)} articles fetched.")
        return articles
    except Exception as e:
        print(f"  [NewsAPI] Failed: {e}")
        return []


def _fetch_guardian(topic: str, page_size: int = 20) -> list[dict]:
    """
    Fetch articles from The Guardian API and normalize structure.
    """
    if not GUARDIAN_API_KEY:
        print("  [Guardian] Key missing — skipping.")
        return []
    try:
        resp = requests.get(
            "https://content.guardianapis.com/search",
            params={
                "q":           topic,
                "lang":        "en",
                "page-size":   page_size,
                "show-fields": "bodyText,trailText",
                "api-key":     GUARDIAN_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("response", {}).get("results", [])
        articles = []
        for a in results:
            fields  = a.get("fields", {})
            content = fields.get("bodyText") or fields.get("trailText") or ""
            if len(content) < 150:
                continue
            articles.append({
                "source":      "The Guardian",
                "title":       a.get("webTitle", ""),
                "description": fields.get("trailText", ""),
                "content":     content,
                "url":         a.get("webUrl", ""),
                "api":         "Guardian",
            })
        print(f"  [Guardian] {len(articles)} articles fetched.")
        return articles
    except Exception as e:
        print(f"  [Guardian] Failed: {e}")
        return []


def _fetch_gnews(topic: str, page_size: int = 10) -> list[dict]:
    """
    Fetch articles from GNews API (limited by free tier).
    """
    if not GNEWS_API_KEY:
        print("  [GNews] Key missing — skipping.")
        return []
    try:
        resp = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q":        topic,
                "lang":     "en",
                "max":      min(page_size, 10),  # GNews free tier max is 10
                "apikey":   GNEWS_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        articles = []
        for a in resp.json().get("articles", []):
            content = a.get("content") or a.get("description") or ""
            if len(content) < 150:
                continue
            articles.append({
                "source":      a.get("source", {}).get("name", "Unknown"),
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "content":     content,
                "url":         a.get("url", ""),
                "api":         "GNews",
            })
        print(f"  [GNews] {len(articles)} articles fetched.")
        return articles
    except Exception as e:
        print(f"  [GNews] Failed: {e}")
        return []

# ──────────────────────────────────────────────────────────────────────────────
# Deduplication Logic
# ──────────────────────────────────────────────────────────────────────────────

def _deduplicate(articles: list[dict]) -> list[dict]:
    """
    Removes duplicate articles using normalized titles.

    Strategy:
    - Lowercase + remove whitespace
    - Use prefix match to handle minor variations

    Note:
    Priority is preserved based on fetch order:
    NewsAPI → Guardian → GNews

    Returns:
        list[dict]: Deduplicated articles
    """
    seen_titles = set()
    unique = []

    for a in articles:
        key = "".join(a["title"].lower().split())[:60]

        if key and key not in seen_titles:
            seen_titles.add(key)
            unique.append(a)

    return unique

# ──────────────────────────────────────────────────────────────────────────────
# Main Public API
# ──────────────────────────────────────────────────────────────────────────────

def fetch_news(topic: str, page_size: int = 20) -> dict:
    """
    Main pipeline to fetch, process, and rank news articles.

    Workflow:
    1. Fetch from multiple APIs in parallel
    2. Normalize and merge results
    3. Deduplicate articles
    4. Score relevance
    5. Sort and filter

    Returns:
        dict: {
            "articles": [...],
            "total_fetched": int
        }
    """
    print("\nQuerying news sources in parallel...")

    # Parallel API calls to reduce latency
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_fetch_newsapi,  topic, page_size): "NewsAPI",
            executor.submit(_fetch_guardian, topic, page_size): "Guardian",
            executor.submit(_fetch_gnews,    topic, 10):        "GNews",
        }
        all_articles = []
        for future in as_completed(futures):
            all_articles.extend(future.result())

    # Remove duplicates across sources
    all_articles = _deduplicate(all_articles)
    print(f"\nTotal unique articles after deduplication: {len(all_articles)}")

    # Compute relevance score for ranking
    for article in all_articles:
        article["_relevance_score"] = _keyword_relevance_score(
            article["title"],
            article["description"],
            article["content"],
            topic,
        )

    # Sort articles by relevance (descending)
    all_articles.sort(key=lambda a: a["_relevance_score"], reverse=True)

    # Filter out low-quality matches
    filtered = [a for a in all_articles if a["_relevance_score"] >= 0.25]
    print(f"Articles passing relevance threshold: {len(filtered)}")

    return {"articles": filtered, "total_fetched": len(all_articles)}


def save_articles(data: dict, topic: str) -> str:
    """
    Saves fetched articles to a timestamped JSON file.

    Args:
        data (dict): Output from fetch_news()
        topic (str): Topic used for naming file

    Returns:
        str: File path of saved JSON
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure filename is filesystem-safe
    safe_topic = topic.replace(" ", "_").lower()

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{safe_topic}_{timestamp}.json"

    # Write JSON with pretty formatting
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    return str(output_path)