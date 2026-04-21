def compare_summaries(summaries: list[dict]) -> dict:
    return {
        "total_sources": len(summaries),
        "sources": [item.get("source", "Unknown") for item in summaries],
    }