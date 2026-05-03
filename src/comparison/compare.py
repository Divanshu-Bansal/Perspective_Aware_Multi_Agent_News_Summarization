import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


def classify_perspective(text: str) -> str:
    """
    Classifies a given text into a high-level perspective category
    based on simple keyword matching.

    Args:
        text (str): Input text (typically a news summary)

    Returns:
        str: Predicted perspective category (e.g., Economic, Political, etc.)
    """
    text = text.lower()

    # Keyword dictionary for each perspective category
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

    # Count keyword matches for each category
    scores = {cat: sum(1 for w in words if w in text)
              for cat, words in categories.items()}
    
    # Select category with highest score
    best = max(scores, key=scores.get)

    # Return category only if at least one keyword matched
    return best if scores[best] > 0 else "General"


def clean_for_theme_extraction(text: str) -> str:
    """
    Cleans raw text for better TF-IDF theme extraction.

    Removes:
    - HTML tags / artifacts
    - URLs
    - Special characters
    - Extra whitespace

    Args:
        text (str): Raw input text

    Returns:
        str: Cleaned text
    """
    text = text.lower()

    # Remove common HTML-related words/tags
    text = re.sub(r"\b(li|ul|ol|html|href|span|div|class)\b", " ", text)

    # Remove truncated text markers like "[+123 chars]"
    text = re.sub(r"\[\+\d+\s*chars\]", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Keep only alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compare_summaries(source_summaries: list[dict], topic: str = "") -> dict:
    """
    Compares multiple news summaries to:
    - Identify similarity between sources
    - Extract common themes
    - Classify perspectives

    Args:
        source_summaries (list[dict]): List of dicts containing:
            - summary (str)
            - source (str)
            - title (str)
        topic (str, optional): Topic used to filter out irrelevant keywords

    Returns:
        dict: Analysis results including:
            - similar_pairs
            - common_themes
            - perspectives
            - perspective_diversity
            - similarity_method
    """

    # Handle empty input
    if not source_summaries:
        return {"similar_pairs": [], "common_themes": [], "perspectives": []}

    # Filter out entries without summaries
    valid_items = [item for item in source_summaries if item.get("summary")]

    # Classify perspective for each summary
    perspectives = [
        {
            "source": item.get("source", "Unknown"),
            "title": item.get("title", ""),
            "perspective": classify_perspective(item.get("summary", ""))
        }
        for item in valid_items
    ]

    # If only one summary, skip similarity analysis
    if len(valid_items) < 2:
        return {"similar_pairs": [], "common_themes": [], "perspectives": perspectives}


    # Extract summaries and clean them
    summaries = [item["summary"] for item in valid_items]
    cleaned_texts = [clean_for_theme_extraction(s) for s in summaries]


    # Custom stop words to reduce noise in theme extraction
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

    # TF-IDF vectorization (captures importance of words/phrases)
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=200,
        ngram_range=(1, 2), # include unigrams and bigrams
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

    # Compute pairwise cosine similarity between summaries
    similarity_matrix = cosine_similarity(tfidf_matrix)


    # Identify similar summary pairs above threshold
    similar_pairs = []
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            score = similarity_matrix[i][j]
            if score > 0.15:    # threshold tuned for loose similarity
                similar_pairs.append((i, j, round(float(score), 2)))
    
    # Sort by highest similarity
    similar_pairs.sort(key=lambda x: x[2], reverse=True)

    # Extract feature importance scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    # Get top-ranked features
    top_indices = tfidf_scores.argsort()[-30:][::-1]

    # Additional noise filtering
    noise_words = {
        "html", "chars", "continue", "according", "reported",
        "people", "years", "year", "said", "says", "just",
        "thing", "things", "company", "companies",
        "quarter", "results", "opened", "first", "second",
        "third", "fourth", "percent", "billion", "million",
    }

    # Extract meaningful common themes
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

        # Limit to top 8 themes
        if len(common_themes) == 8:
            break
    
    # Calculate number of unique perspectives
    perspective_types = list(set(p["perspective"] for p in perspectives))

    return {
        "similar_pairs": similar_pairs,
        "common_themes": common_themes,
        "perspectives": perspectives,
        "perspective_diversity": len(perspective_types),
        "similarity_method": "tf-idf + cosine similarity",
    }