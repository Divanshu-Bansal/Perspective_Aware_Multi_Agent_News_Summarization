import streamlit as st
import os
import re
import json
import time
import plotly.express as px
import pandas as pd
from pathlib import Path

from src.data_collection.fetch_news import fetch_news, save_articles
from src.preprocessing.clean_text import clean_text
from src.summarization.summarize import summarize_article
from src.comparison.compare import compare_summaries
from src.neutral_generation.generate import generate_neutral_summary
from src.tracking.clearml_tracker import (
    init_task, log_pipeline_config, log_fetch_metrics,
    log_article_metrics, log_comparison_metrics,
    log_final_summary, close_task,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Perspective-Aware News Summarizer",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .article-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .article-source {
        font-size: 0.75rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .article-title {
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0.3rem 0;
    }
    .article-summary {
        font-size: 0.85rem;
        color: #ccc;
        line-height: 1.5;
    }
    .relevance-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #4ade80, #22d3ee);
        margin-top: 0.5rem;
    }
    .perspective-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    .neutral-summary-box {
        background: #0f172a;
        border-left: 4px solid #4ade80;
        border-radius: 8px;
        padding: 1.5rem;
        font-size: 1.05rem;
        line-height: 1.8;
        margin-top: 1rem;
    }
    .theme-chip {
        display: inline-block;
        background: #1e3a5f;
        color: #7dd3fc;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        margin: 3px;
    }
    .step-header {
        font-size: 1rem;
        font-weight: 600;
        color: #4ade80;
        margin: 1rem 0 0.5rem;
    }
    .metric-box {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4ade80;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #aaa;
    }
</style>
""", unsafe_allow_html=True)

PERSPECTIVE_COLORS = {
    "Economic":      "#f59e0b",
    "Political":     "#ef4444",
    "Technological": "#3b82f6",
    "Social":        "#8b5cf6",
    "Security":      "#ec4899",
    "General":       "#6b7280",
}

EXCLUDE_KEYWORDS = [
    "celebrity", "actor", "actress", "movie", "film",
    "hollywood", "entertainment", "reality tv", "fashion week",
    "music video", "box office", "grammy", "oscar",
]

EXCLUDE_SOURCES = [
    "yahoo entertainment", "tmz", "people", "e! news",
    "entertainment weekly", "variety", "deadline",
]


def is_topically_relevant(text: str, topic: str, threshold: float = 0.15) -> bool:
    topic_words = [w for w in topic.lower().split() if len(w) > 2]
    if not topic_words:
        return True
    matches = sum(1 for w in topic_words if w in text.lower())
    return (matches / len(topic_words)) >= threshold


def relevance_bar_html(score: float) -> str:
    width = int(score * 100)
    return f"""
    <div style="background:#333;border-radius:3px;height:6px;margin-top:6px;">
        <div style="width:{width}%;height:6px;border-radius:3px;
                    background:linear-gradient(90deg,#4ade80,#22d3ee);"></div>
    </div>
    <div style="font-size:0.75rem;color:#aaa;margin-top:2px;">
        Relevance: {score:.2f}
    </div>
    """


def perspective_badge_html(perspective: str) -> str:
    color = PERSPECTIVE_COLORS.get(perspective, "#6b7280")
    return f'<span style="background:{color}22;color:{color};border:1px solid {color}55;' \
           f'border-radius:20px;padding:2px 10px;font-size:0.75rem;font-weight:600;">' \
           f'{perspective}</span>'


def render_article_card(item: dict, show_summary: bool = True):
    source   = item.get("source", "Unknown")
    title    = item.get("title", "")
    summary  = item.get("summary", "")
    score    = item.get("relevance_score", 0)
    url      = item.get("url", "")
    api      = item.get("api", "")
    persp    = item.get("perspective", "General")

    title_html = f'<a href="{url}" target="_blank" style="color:white;text-decoration:none;">{title}</a>' \
                 if url else title

    summary_html = f'<div class="article-summary">{summary}</div>' if show_summary and summary else ""

    st.markdown(f"""
    <div class="article-card">
        <div class="article-source">{source} &nbsp;·&nbsp; {api}</div>
        <div class="article-title">{title_html}</div>
        {summary_html}
        {relevance_bar_html(score)}
        {perspective_badge_html(persp)}
    </div>
    """, unsafe_allow_html=True)


def run_pipeline_streaming(topic: str, page_size: int, max_articles: int):
    """Main pipeline with progressive streaming UI updates."""

    # ── ClearML ───────────────────────────────────────────────────────────────
    task = init_task(topic)
    log_pipeline_config(task, topic, page_size, max_articles)

    # ── Step 1: Fetch ─────────────────────────────────────────────────────────
    st.markdown('<div class="step-header">Step 1 — Querying news sources...</div>',
                unsafe_allow_html=True)

    fetch_status = st.empty()
    fetch_status.info("Querying NewsAPI, The Guardian and GNews in parallel...")

    news_data  = fetch_news(topic=topic, page_size=page_size)
    save_articles(news_data, topic)
    articles   = news_data.get("articles", [])
    total      = news_data.get("total_fetched", len(articles))

    if not articles:
        fetch_status.error(
            "No articles found. Try a simpler topic like 'AI regulation' or 'Bitcoin price'."
        )
        close_task(task)
        return

    log_fetch_metrics(task, {
        "total_fetched":   total,
        "after_relevance": len(articles),
        "newsapi_count":   sum(1 for a in articles if a.get("api") == "NewsAPI"),
        "guardian_count":  sum(1 for a in articles if a.get("api") == "Guardian"),
        "gnews_count":     sum(1 for a in articles if a.get("api") == "GNews"),
    })

    fetch_status.success(
        f"Found {total} articles across 3 sources → {len(articles)} passed relevance filter"
    )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    newsapi_n  = sum(1 for a in articles if a.get("api") == "NewsAPI")
    guardian_n = sum(1 for a in articles if a.get("api") == "Guardian")
    gnews_n    = sum(1 for a in articles if a.get("api") == "GNews")

    col1.metric("Total fetched",    total)
    col2.metric("NewsAPI",          newsapi_n)
    col3.metric("The Guardian",     guardian_n)
    col4.metric("GNews",            gnews_n)

    st.divider()

    # ── Step 2: Filter + Summarise (progressive) ──────────────────────────────
    st.markdown('<div class="step-header">Step 2 — Reading and summarizing articles...</div>',
                unsafe_allow_html=True)

    source_summaries = []
    seen_sources     = set()
    skipped          = 0
    article_placeholders = []

    for article in articles:
        if len(source_summaries) >= max_articles:
            break

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
        if source_name in seen_sources:
            continue
        seen_sources.add(source_name)

        # Show article card immediately (without summary yet)
        placeholder = st.empty()
        placeholder.markdown(f"""
        <div class="article-card">
            <div class="article-source">{source_name} &nbsp;·&nbsp; {article.get('api','')}</div>
            <div class="article-title">{article.get('title','')}</div>
            <div class="article-summary" style="color:#666;font-style:italic;">
                Summarizing...
            </div>
            {relevance_bar_html(article.get('_relevance_score', 0))}
        </div>
        """, unsafe_allow_html=True)

        # Summarise
        summary = summarize_article(cleaned_text)
        # Free memory after each summarization on cloud
        if os.getenv("IS_STREAMLIT_CLOUD") == "true":
            import gc
            gc.collect()
        if not summary:
            placeholder.empty()
            continue

        item = {
            "source":          source_name,
            "title":           article.get("title", ""),
            "url":             article.get("url", ""),
            "summary":         summary,
            "relevance_score": article.get("_relevance_score", 0.0),
            "api":             article.get("api", "Unknown"),
        }
        source_summaries.append(item)
        article_placeholders.append((placeholder, item))

        # Update card with real summary
        placeholder.markdown(f"""
        <div class="article-card">
            <div class="article-source">{source_name} &nbsp;·&nbsp; {article.get('api','')}</div>
            <div class="article-title">
                <a href="{item['url']}" target="_blank" style="color:white;text-decoration:none;">
                    {item['title']}
                </a>
            </div>
            <div class="article-summary">{summary}</div>
            {relevance_bar_html(item['relevance_score'])}
        </div>
        """, unsafe_allow_html=True)

    if not source_summaries:
        st.error("No relevant articles found after filtering. Try a broader topic.")
        close_task(task)
        return

    log_article_metrics(task, source_summaries)
    st.caption(f"{len(source_summaries)} articles summarized · {skipped} skipped")
    st.divider()

    # ── Step 3: Comparison ────────────────────────────────────────────────────
    st.markdown('<div class="step-header">Step 3 — Comparing perspectives...</div>',
                unsafe_allow_html=True)

    comparison_result = compare_summaries(source_summaries, topic=topic)
    log_comparison_metrics(task, comparison_result)

    perspectives  = comparison_result.get("perspectives", [])
    common_themes = comparison_result.get("common_themes", [])
    diversity     = comparison_result.get("perspective_diversity", 0)

    # Update article cards with perspective badges
    for placeholder, item in article_placeholders:
        persp = next(
            (p["perspective"] for p in perspectives if p["source"] == item["source"]),
            "General"
        )
        item["perspective"] = persp
        placeholder.markdown(f"""
        <div class="article-card">
            <div class="article-source">{item['source']} &nbsp;·&nbsp; {item['api']}</div>
            <div class="article-title">
                <a href="{item['url']}" target="_blank" style="color:white;text-decoration:none;">
                    {item['title']}
                </a>
            </div>
            <div class="article-summary">{item['summary']}</div>
            {relevance_bar_html(item['relevance_score'])}
            {perspective_badge_html(persp)}
        </div>
        """, unsafe_allow_html=True)

    # Perspective chart + themes side by side
    col_chart, col_themes = st.columns([1, 1])

    with col_chart:
        st.subheader("Perspective breakdown")
        perspective_counts = {}
        for p in perspectives:
            ptype = p.get("perspective", "General")
            perspective_counts[ptype] = perspective_counts.get(ptype, 0) + 1

        if perspective_counts:
            df = pd.DataFrame({
                "Perspective": list(perspective_counts.keys()),
                "Count":       list(perspective_counts.values()),
            })
            colors = [PERSPECTIVE_COLORS.get(p, "#6b7280") for p in df["Perspective"]]
            fig = px.bar(
                df, x="Perspective", y="Count",
                color="Perspective",
                color_discrete_sequence=colors,
                text="Count",
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.metric("Perspective diversity", f"{diversity} unique viewpoints")

    with col_themes:
        st.subheader("Key themes")
        if common_themes:
            themes_html = " ".join(
                f'<span class="theme-chip">{t}</span>' for t in common_themes
            )
            st.markdown(themes_html, unsafe_allow_html=True)
        else:
            st.caption("No strong themes detected.")

        similar_pairs = comparison_result.get("similar_pairs", [])
        if similar_pairs:
            st.subheader("Similar article pairs")
            for i, j, score in similar_pairs:
                s1 = source_summaries[i]["source"] if i < len(source_summaries) else f"Article {i+1}"
                s2 = source_summaries[j]["source"] if j < len(source_summaries) else f"Article {j+1}"
                st.caption(f"{s1} ↔ {s2} — similarity: {score:.2f}")

    st.divider()

    # ── Step 4: Neutral summary ───────────────────────────────────────────────
    st.markdown('<div class="step-header">Step 4 — Generating neutral summary...</div>',
                unsafe_allow_html=True)

    summary_placeholder = st.empty()
    summary_placeholder.info("Combining perspectives into a neutral summary...")

    neutral_summary = generate_neutral_summary(
        source_summaries, comparison_result, topic=topic
    )
    log_final_summary(task, neutral_summary, topic)

    # TODO
    # # Extract just the summary text for display
    # lines = neutral_summary.strip().split("\n")
    # summary_text = ""
    # for line in lines:
    #     line = line.strip()
    #     if line and not line.startswith("=") and not line.startswith("-") \
    #             and not line.startswith("NEUTRAL") and not line.startswith("Topic:") \
    #             and not line.startswith("KEY") and not line.startswith("•") \
    #             and not line.startswith("PERSP") and not line.startswith("[") \
    #             and not line.startswith("SOURCE") and not line.startswith("Relevance") \
    #             and not line.startswith("URL") and not line.startswith("Perspective div") \
    #             and not line.startswith("Similarity"):
    #         summary_text += line + " "


    lines = neutral_summary.strip().split("\n")
    clean_lines = []

    for line in lines:
        line = line.strip()

        # skip empty
        if not line:
            continue

        # skip separators and metadata
        if re.match(r"^[=\-\─]+$", line):
            continue
        if any(keyword in line.lower() for keyword in [
            "topic:", "key themes", "perspectives", "source", "relevance",
            "similarity", "diversity", "http", "www"
        ]):
            continue

        # skip lines that look like titles (too short or weird formatting)
        if len(line.split()) < 5:
            continue

        clean_lines.append(line)

    # Take only first 2–3 meaningful lines
    summary_text = " ".join(clean_lines[:3])

    summary_placeholder.markdown(f"""
    <div class="neutral-summary-box">
        {summary_text.strip()}
    </div>
    """, unsafe_allow_html=True)

    # Save output
    output_dir  = Path("outputs/summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "latest_summary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "topic":            topic,
            "articles_found":   len(source_summaries),
            "source_summaries": source_summaries,
            "comparison":       comparison_result,
            "neutral_summary":  neutral_summary,
        }, f, indent=2)

    close_task(task)

    # Download button
    st.divider()
    st.download_button(
        label="Download full report (JSON)",
        data=open(output_file).read(),
        file_name=f"summary_{topic.replace(' ','_')}.json",
        mime="application/json",
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    is_cloud = os.getenv("IS_STREAMLIT_CLOUD") == "true"
    page_size    = st.slider("Articles to fetch per source", 5, 20, 5 if is_cloud else 10)
    max_articles = st.slider("Max articles to analyse",      3, 10, 4 if is_cloud else 8)
    st.divider()
    st.markdown("### Example topics")
    examples = [
        "US China trade war",
        "artificial intelligence regulation",
        "electric vehicle market",
        "cryptocurrency Bitcoin",
        "climate change policy",
        "generative AI jobs",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["topic_input"] = ex

    st.divider()
    st.caption("Powered by NewsAPI · The Guardian · GNews · BART · ClearML")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🗞️ Perspective-Aware News Summarizer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-source · Bias-aware · Transformer-powered</div>',
            unsafe_allow_html=True)

topic_input = st.text_input(
    "Enter a news topic",
    value=st.session_state.get("topic_input", ""),
    placeholder="e.g. US China trade war, AI regulation, Bitcoin...",
    key="topic_input",
)

analyse_btn = st.button("Analyse", type="primary", use_container_width=True)

if analyse_btn and topic_input.strip():
    run_pipeline_streaming(
        topic=topic_input.strip(),
        page_size=page_size,
        max_articles=max_articles,
    )
elif analyse_btn:
    st.warning("Please enter a topic first.")