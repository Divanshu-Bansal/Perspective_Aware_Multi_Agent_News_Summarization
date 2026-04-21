def generate_neutral_summary(summaries: list[dict]) -> str:
    if not summaries:
        return ""

    combined = []
    for item in summaries:
        source = item.get("source", "Unknown")
        summary = item.get("summary", "")
        combined.append(f"{source}: {summary}")

    return "\n".join(combined)