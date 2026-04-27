from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compare_summaries(source_summaries):
    texts = [item["summary"] for item in source_summaries if item["summary"]]

    if len(texts) < 2:
        return {"similar_pairs": []}

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    similar_pairs = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = similarity_matrix[i][j]
            if score > 0.1:  # threshold
                similar_pairs.append((i, j, score))

    return {"similar_pairs": similar_pairs}