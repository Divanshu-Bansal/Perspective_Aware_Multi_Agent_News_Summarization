# app.py
import streamlit as st
import os
import re
import json
import plotly.express as px
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Internal pipeline modules
# ─────────────────────────────────────────────────────────────
from src.data_collection.fetch_news import fetch_news, save_articles
from src.preprocessing.clean_text import clean_text
from src.summarization.summarize import summarize_article
from src.comparison.compare import compare_summaries
from src.neutral_generation.generate import generate_neutral_summary, generate_biased_summary
from src.tracking.clearml_tracker import (
    init_task, log_pipeline_config, log_fetch_metrics,
    log_article_metrics, log_comparison_metrics,
    log_final_summary, close_task,
)

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Perspective-Aware News Summarizer",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────────────────────────
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
    .neutral-summary-box {
        min-height: 180px;
        background: #0f172a;
        border-left: 4px solid #4ade80;
        border-radius: 8px;
        padding: 1.5rem;
        font-size: 1.05rem;
        line-height: 1.8;
        margin-top: 1rem;
        color: #e2e8f0;
    }
    .biased-summary-box {
        min-height: 180px;
        background: #1a0a0a;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1.5rem;
        font-size: 1.05rem;
        line-height: 1.8;
        margin-top: 1rem;
        color: #e2e8f0;
    }
    .step-comparison-card {
        min-height: 360px;
    }
    .summary-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .biased-label  { color: #ef4444; }
    .neutral-label { color: #4ade80; }
    .missing-perspective-chip {
        display: inline-block;
        background: #3f1a1a;
        color: #fca5a5;
        border: 1px solid #ef444455;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px;
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
    .comparison-meta {
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.5rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────
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
    return (
        f'<span style="background:{color}22;color:{color};'
        f'border:1px solid {color}55;border-radius:20px;'
        f'padding:2px 10px;font-size:0.75rem;font-weight:600;">'
        f'{perspective}</span>'
    )


def extract_summary_text(neutral_summary: str) -> str:
    lines = neutral_summary.strip().split("\n")
    capture = False
    collected = []

    for line in lines:
        line = line.strip()
        if line.startswith("Topic:"):
            capture = True
            continue
        if capture and (
            re.match(r"^[=\-\─]{5,}$", line) or
            line.upper().startswith("KEY THEMES") or
            line.upper().startswith("PERSPECTIVES") or
            line.upper().startswith("SOURCE CONTRIB") or
            line.upper().startswith("NEUTRAL SUMMARY")
        ):
            break
        if capture and line and len(line.split()) >= 5:
            collected.append(line)

    summary_text = " ".join(collected[:3]).strip()

    if not summary_text:
        for line in lines:
            line = line.strip()
            if (
                len(line.split()) >= 8
                and not re.match(r"^[=\-\─•\[\|]+", line)
                and not any(kw in line.lower() for kw in [
                    "topic:", "theme", "perspective", "source", "relevance",
                    "similarity", "http", "www", "diversity"
                ])
            ):
                summary_text = line
                break

    return summary_text or "Summary could not be extracted. Please check the full report."


def article_card_html(
    source: str, api: str, title: str, url: str,
    summary: str, score: float,
    perspective: str = "", summarizing: bool = False,
) -> str:
    """Build a consistent article card HTML string."""

    # ── Escape all user-supplied text first ──────────────────
    def esc(text: str) -> str:
        return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
        )

    source_esc = esc(source)
    api_esc    = esc(api)
    title_esc  = esc(title)

    # Title with optional link — url is not user content so safe to use directly
    title_html = (
        f'<a href="{url}" target="_blank" '
        f'style="color:white;text-decoration:none;">{title_esc}</a>'
        if url else title_esc
    )

    # Summary — escape before injecting
    if summarizing:
        summary_html = (
            '<div class="article-summary" '
            'style="color:#666;font-style:italic;">Summarizing...</div>'
        )
    else:
        summary_html = (
            f'<div class="article-summary">{esc(summary)}</div>'
        )

    # Perspective badge — built from our own controlled strings, no escaping needed
    badge_html = ""
    if perspective and not summarizing:
        color = PERSPECTIVE_COLORS.get(perspective, "#6b7280")
        persp_esc = esc(perspective)
        badge_html = (
            f'<div style="margin-top:6px;">'
            f'<span style="background:{color}22;color:{color};'
            f'border:1px solid {color}55;border-radius:20px;'
            f'padding:2px 10px;font-size:0.75rem;font-weight:600;">'
            f'{persp_esc}</span>'
            f'</div>'
        )

    # Relevance bar — score is a float, fully controlled
    width = max(0, min(100, int(score * 100)))
    bar_html = (
        f'<div style="background:#333;border-radius:3px;height:6px;margin-top:8px;">'
        f'<div style="width:{width}%;height:6px;border-radius:3px;'
        f'background:linear-gradient(90deg,#4ade80,#22d3ee);"></div>'
        f'</div>'
        f'<div style="font-size:0.75rem;color:#aaa;margin-top:2px;">'
        f'Relevance: {score:.2f}'
        f'</div>'
    )

    return (
        f'<div class="article-card">'
        f'<div class="article-source">{source_esc} &nbsp;·&nbsp; {api_esc}</div>'
        f'<div class="article-title">{title_html}</div>'
        f'{summary_html}'
        f'{bar_html}'
        f'{badge_html}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────
def run_pipeline_streaming(topic: str, page_size: int, max_articles: int):

    # ── ClearML tracking ──
    task = init_task(topic)
    log_pipeline_config(task, topic, page_size, max_articles)

    # ── Step 1: Fetch ──
    st.markdown('<div class="step-header">Step 1 — Querying news sources...</div>', unsafe_allow_html=True)
    fetch_status = st.empty()
    fetch_status.info("Querying NewsAPI, The Guardian and GNews in parallel...")

    news_data = fetch_news(topic=topic, page_size=page_size)
    save_articles(news_data, topic)
    articles  = news_data.get("articles", [])
    total     = news_data.get("total_fetched", len(articles))

    if not articles:
        fetch_status.error("No articles found. Try a simpler topic like 'AI regulation' or 'Bitcoin price'.")
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
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total fetched", total)
    col2.metric("NewsAPI",       sum(1 for a in articles if a.get("api") == "NewsAPI"))
    col3.metric("The Guardian",  sum(1 for a in articles if a.get("api") == "Guardian"))
    col4.metric("GNews",         sum(1 for a in articles if a.get("api") == "GNews"))
    st.divider()

    # ── Step 2: Filter + Summarize ──
    st.markdown('<div class="step-header">Step 2 — Reading and summarizing articles...</div>', unsafe_allow_html=True)

    source_summaries     = []
    seen_sources         = set()
    skipped              = 0
    article_placeholders = []

    for article in articles:
        if len(source_summaries) >= max_articles:
            break

        source_name  = article.get("source", "Unknown")
        content      = article.get("content") or article.get("title") or ""
        cleaned_text = clean_text(content)
        lower_text   = cleaned_text.lower()

        if any(word in lower_text for word in EXCLUDE_KEYWORDS):
            skipped += 1; continue
        if source_name.lower() in EXCLUDE_SOURCES:
            skipped += 1; continue
        if len(cleaned_text.split()) < 30:
            skipped += 1; continue
        if not is_topically_relevant(cleaned_text, topic):
            skipped += 1; continue
        if source_name in seen_sources:
            continue
        seen_sources.add(source_name)

        placeholder = st.empty()
        placeholder.markdown(
            article_card_html(
                source=source_name, api=article.get("api", ""),
                title=article.get("title", ""), url=article.get("url", ""),
                summary="", score=article.get("_relevance_score", 0), summarizing=True,
            ),
            unsafe_allow_html=True,
        )

        summary = summarize_article(cleaned_text)

        if os.getenv("IS_STREAMLIT_CLOUD") == "true":
            import gc; gc.collect()

        if not summary:
            placeholder.empty(); continue

        item = {
            "source":          source_name,
            "title":           article.get("title", ""),
            "url":             article.get("url", ""),
            "summary":         summary,
            "relevance_score": article.get("_relevance_score", 0.0),
            "api":             article.get("api", "Unknown"),
            "perspective":     "",
        }
        source_summaries.append(item)
        article_placeholders.append((placeholder, item))

        placeholder.markdown(
            article_card_html(
                source=item["source"], api=item["api"], title=item["title"],
                url=item["url"], summary=summary, score=item["relevance_score"],
                summarizing=False,
            ),
            unsafe_allow_html=True,
        )

    if not source_summaries:
        st.error(
            f"No high-quality articles found for '{topic}'. "
            f"Try a broader or more specific topic."
        )
        close_task(task)
        return

    log_article_metrics(task, source_summaries)
    st.caption(f"{len(source_summaries)} articles summarized · {skipped} skipped")
    st.divider()

    # ── Step 3: Comparison ──
    st.markdown('<div class="step-header">Step 3 — Comparing perspectives...</div>', unsafe_allow_html=True)

    comparison_result = compare_summaries(source_summaries, topic=topic)
    log_comparison_metrics(task, comparison_result)

    perspectives  = comparison_result.get("perspectives", [])
    common_themes = comparison_result.get("common_themes", [])
    diversity     = comparison_result.get("perspective_diversity", 0)

    for placeholder, item in article_placeholders:
        persp = next(
            (p["perspective"] for p in perspectives if p["source"] == item["source"]),
            "General"
        )
        item["perspective"] = persp
        placeholder.markdown(
            article_card_html(
                source=item["source"], api=item["api"], title=item["title"],
                url=item["url"], summary=item["summary"], score=item["relevance_score"],
                perspective=persp, summarizing=False,
            ),
            unsafe_allow_html=True,
        )

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
            fig = px.bar(df, x="Perspective", y="Count", color="Perspective",
                         color_discrete_sequence=colors, text="Count")
            fig.update_layout(
                showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.metric("Perspective diversity", f"{diversity} unique viewpoints")

    with col_themes:
        st.subheader("Key themes")
        if common_themes:
            themes_html = " ".join(f'<span class="theme-chip">{t}</span>' for t in common_themes)
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

        # ── Step 4 — Single-Source Summary vs. Multi-Source Synthesis ──
    st.markdown(
        '<div class="step-header">Step 4 — Single-Source Summary vs. Multi-Source Synthesis</div>',
        unsafe_allow_html=True
    )

    biased_result = generate_biased_summary(source_summaries, comparison_result)
    neutral_summary = generate_neutral_summary(source_summaries, comparison_result, topic=topic)
    # summary_text = extract_summary_text(neutral_summary)   #TODO
    summary_text = neutral_summary.strip()

    log_final_summary(task, neutral_summary, topic)

    perspective_counts = {}
    for p in perspectives:
        ptype = p.get("perspective", "General")
        perspective_counts[ptype] = perspective_counts.get(ptype, 0) + 1

    missing = biased_result.get("missing_perspectives", [])
    missing_count = len(missing)

    st.info(
        "💡 This section compares what a reader gets from a single outlet versus "
        "a synthesis built from multiple independent sources. More sources can broaden "
        "perspective coverage, but multi-source synthesis reduces single-outlet dominance "
        "rather than guaranteeing complete neutrality."
    )

    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Source coverage", "1 source", f"vs {len(source_summaries)} sources")
    stat2.metric("Perspective coverage", "1 perspective", f"vs {diversity} viewpoints")
    stat3.metric("Coverage gap", f"{missing_count} missing", "in biased version")

    st.markdown("### Comparison view")

    col_biased, col_neutral = st.columns(2)

    with col_biased:
        st.markdown(
            """
            <div style="font-size:0.82rem;font-weight:800;letter-spacing:0.08em;
                        text-transform:uppercase;color:#ef4444;margin-bottom:0.6rem;">
                📰 Single-Source Summary
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.caption(
                f"Source: {biased_result['source']} ({biased_result['api']})\n"
                f"Perspective: {biased_result['perspective']}\n"
                f"Relevance: {biased_result['relevance']:.2f}\n"
                f"Coverage: 1 of {len(source_summaries)} articles"
            )

            st.markdown(
                f"""
                <div style="background:#140707;border-left:4px solid #ef4444;border-radius:10px;
                            padding:1.2rem;color:#f3f4f6;font-size:1.02rem;line-height:1.8;">
                    {biased_result["summary"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if missing:
                chips = " ".join(
                    f'<span style="display:inline-block;background:#3f1a1a;color:#fca5a5;'
                    f'border:1px solid #ef444455;border-radius:20px;padding:3px 10px;'
                    f'font-size:0.75rem;margin:3px;">{p}</span>'
                    for p in missing
                )
                st.markdown(
                    f'<div style="margin-top:0.7rem;font-size:0.8rem;color:#999;">'
                    f'<strong>Missing perspectives:</strong><br>{chips}</div>',
                    unsafe_allow_html=True,
                )

            st.caption("This reflects what a reader gets from one outlet only.")

    with col_neutral:
        st.markdown(
            """
            <div style="font-size:0.82rem;font-weight:800;letter-spacing:0.08em;
                        text-transform:uppercase;color:#22c55e;margin-bottom:0.6rem;">
                🌐 Multi-Source Synthesis
            </div>
            """,
            unsafe_allow_html=True,
        )

        all_persp_str = " · ".join(
            f'<span style="color:{PERSPECTIVE_COLORS.get(p, "#6b7280")};font-weight:700;">{p}</span>'
            for p in perspective_counts
        )

        with st.container(border=True):
            st.markdown(
                f'<div style="font-size:0.8rem;color:#888;line-height:1.8;">'
                f'<b>Sources:</b> {len(source_summaries)} articles<br>'
                f'<b>Perspectives:</b> {diversity} unique viewpoints<br>'
                f'<b>Coverage:</b> {all_persp_str}<br>'
                f'<b>Method:</b> Multi-source perspective-balanced synthesis'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="background:#07101f;border-left:4px solid #22c55e;border-radius:10px;
                            padding:1.2rem;color:#f3f4f6;font-size:1.02rem;line-height:1.8;">
                    {summary_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption("This synthesises multiple independent sources to reduce single-outlet dominance.")

    if missing:
        what_this_shows = (
            f"<strong>What this shows:</strong> The single-source summary only reflects the "
            f"<strong>{biased_result['perspective']}</strong> perspective from one outlet. "
            f"The Multi-Source Synthesis incorporates "
            f"<strong>{', '.join(missing)}</strong> additional viewpoints, "
            f"giving a broader picture of how this topic is being covered."
        )

        st.markdown(
            f"""
            <div style="margin-top:1rem;background:#2a2412;border:1px solid #7c6a2a;
                        border-radius:12px;padding:1rem 1.2rem;color:#f3e7b3;line-height:1.7;">
                {what_this_shows}
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div style="margin-top:1rem;background:#10261a;border:1px solid #27543b;
                        border-radius:12px;padding:1rem 1.2rem;color:#c7f3d5;line-height:1.7;">
                <strong>Interpretation:</strong> In this case, the highest-relevance source is already
                close to the broader multi-source framing.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Step 5: Export ──
    output_dir  = Path("outputs/summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "latest_summary.json"

    final_output = {
        "topic":            topic,
        "articles_found":   len(source_summaries),
        "source_summaries": source_summaries,
        "comparison":       comparison_result,
        "biased_summary":   biased_result,
        "neutral_summary":  neutral_summary,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    close_task(task)

    st.download_button(
        label="Download full report (JSON)",
        data=open(output_file).read(),
        file_name=f"summary_{topic.replace(' ', '_')}.json",
        mime="application/json",
    )


# ─────────────────────────────────────────────────────────────
# Sidebar UI
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Settings")
    is_cloud     = os.getenv("IS_STREAMLIT_CLOUD") == "true"
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


# ─────────────────────────────────────────────────────────────
# Main UI Entry Point
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🗞️ Perspective-Aware News Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-source · Bias-aware · Transformer-powered</div>', unsafe_allow_html=True)

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