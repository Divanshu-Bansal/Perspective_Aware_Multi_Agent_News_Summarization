import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


def classify_perspective(text: str) -> str:
    text = text.lower()
    categories = {
        "Economic":      ["market", "price", "economy", "cost", "jobs", "employment",
                          "business", "layoff", "stock", "revenue", "investment", "gdp"],
        "Political":     ["policy", "government", "minister", "law", "election",
                          "president", "congress", "senate", "regulation"],
        "Technological": ["technology", "ai", "innovation", "software", "system",
                          "data", "robot", "model", "algorithm", "compute", "chip"],
        "Social":        ["society", "students", "workers", "community", "public",
                          "youth", "education", "health", "culture", "people"],
        "Security":      ["war", "military", "conflict", "security", "attack",
                          "defence", "defense", "cyber", "threat"],
    }
    scores = {cat: sum(1 for w in words if w in text)
              for cat, words in categories.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


def clean_for_theme_extraction(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(li|ul|ol|html|href|span|div|class)\b", " ", text)
    text = re.sub(r"\[\+\d+\s*chars\]", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compare_summaries(source_summaries: list[dict], topic: str = "") -> dict:
    if not source_summaries:
        return {"similar_pairs": [], "common_themes": [], "perspectives": []}

    valid_items = [item for item in source_summaries if item.get("summary")]

    perspectives = [
        {
            "source": item.get("source", "Unknown"),
            "title": item.get("title", ""),
            "perspective": classify_perspective(item.get("summary", ""))
        }
        for item in valid_items
    ]

    if len(valid_items) < 2:
        return {"similar_pairs": [], "common_themes": [], "perspectives": perspectives}

    summaries = [item["summary"] for item in valid_items]
    cleaned_texts = [clean_for_theme_extraction(s) for s in summaries]

    custom_stop_words = {
        "chars", "continue", "said", "says", "year", "years",
        "people", "according", "article", "news", "reported",
        "also", "like", "make", "made", "many", "new", "way",
        "form", "week", "thing", "things", "using", "used",
        "would", "could", "should", "just", "including",
        "company", "companies", "percent", "million", "billion",
    }

    # Extract topic words and add them to stop words
    # This prevents the topic itself from appearing as a "theme"
    topic_stop_words = set(topic.lower().split()) if topic else set()

    # Merge all stop word lists together
    stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop_words).union(topic_stop_words))

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=200,
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    similar_pairs = []
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            score = similarity_matrix[i][j]
            if score > 0.15:
                similar_pairs.append((i, j, round(float(score), 2)))
    similar_pairs.sort(key=lambda x: x[2], reverse=True)

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = tfidf_scores.argsort()[-30:][::-1]

    noise_words = {
        "html", "chars", "continue", "according", "reported",
        "people", "years", "year", "said", "says", "just",
        "thing", "things", "company", "companies",
        "quarter", "results", "opened", "first", "second",
        "third", "fourth", "percent", "billion", "million",
    }

    common_themes = []
    for index in top_indices:
        theme = feature_names[index].strip()
        if (
            len(theme) > 4
            and not theme.isnumeric()
            and theme not in noise_words
            and not any(token in noise_words for token in theme.split())
            # Also skip any theme that contains a topic word
            and not any(token in topic_stop_words for token in theme.split())
        ):
            common_themes.append(theme)
        if len(common_themes) == 8:
            break

    perspective_types = list(set(p["perspective"] for p in perspectives))

    return {
        "similar_pairs": similar_pairs,
        "common_themes": common_themes,
        "perspectives": perspectives,
        "perspective_diversity": len(perspective_types),
        "similarity_method": "tf-idf + cosine similarity",
    }