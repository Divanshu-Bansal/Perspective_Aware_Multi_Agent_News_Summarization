# Perspective-Aware Multi-Agent News Summarization System

> Automatically collects, analyzes, and summarizes news from multiple sources — producing a balanced, unbiased view of any topic with transparent source attribution and perspective detection.

---

## Overview

This system addresses a core problem in modern news consumption: **single-source bias**. When readers rely on one outlet, they get one perspective. This project fetches articles from multiple sources, summarizes each independently, compares them using NLP techniques, and generates a neutral combined summary that reflects the full landscape of coverage.

Built as a modular multi-agent pipeline where each component (collection, preprocessing, summarization, comparison, generation) operates independently and can be improved or swapped without affecting the others.

---

## Demo Output

```
==================================================
  NEUTRAL SUMMARY
  Topic: OpenAI vs Google AI competition
==================================================

OpenAI is on a mission to scale up its AI compute capacity to 30GW by the
end of this decade. Google's market share has broadly held firm in the wake
of everything AI.

──────────────────────────────────────────────────
  KEY THEMES
──────────────────────────────────────────────────
  • mission
  • openai
  • google
  • market
  • compute scale
  • ai race

──────────────────────────────────────────────────
  PERSPECTIVES DETECTED
──────────────────────────────────────────────────
  [Technological] Wccftech
    └─ OpenAI To Scale-Up AI Compute Capacity To A Whopping 30GW By 2030
  [Economic] Leadershipinseo.com
    └─ Why Google Has Changed & Who's Really Paying for It
  [Economic] Ibtimes.com.au
    └─ Grok vs Gemini vs ChatGPT: Who Wins the AI Race by 2031?

  Perspective diversity score: 2 unique viewpoint(s)
  Similarity method: tf-idf + cosine similarity

──────────────────────────────────────────────────
  SOURCE CONTRIBUTIONS
──────────────────────────────────────────────────
  Wccftech
    Relevance: █████░░░░░ 0.50
  Leadershipinseo.com
    Relevance: ███░░░░░░░ 0.33
  Ibtimes.com.au
    Relevance: ███░░░░░░░ 0.33
==================================================
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Data Collection   │  fetch_news.py
│   Agent             │  NewsAPI → relevance scoring → ranked articles
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Preprocessing     │  clean_text.py
│   Agent             │  URL removal, HTML artifacts, noise filtering
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Summarization     │  summarize.py
│   Agent             │  facebook/bart-large-cnn, chunked inference
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Comparison        │  compare.py
│   Agent             │  TF-IDF, cosine similarity, perspective classification
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Generation        │  generate.py
│   Agent             │  Neutral summary + bias detection + structured output
└────────┬────────────┘
         │
         ▼
  Structured Report
  (stdout + JSON)
```

---

## Key Features

**Multi-source aggregation** — fetches up to 20 articles per query from NewsAPI, ranked by a weighted relevance score (title matches weighted 3×, description 2×, content 1×).

**Intelligent filtering** — four-layer pipeline: minimum length check, topic relevance guard, source deduplication, and exclusion of off-topic categories (entertainment, celebrity, etc.).

**Transformer-based summarization** — uses `facebook/bart-large-cnn` with chunked inference to handle articles of any length without hitting token limits.

**Perspective classification** — classifies each source into Economic, Political, Technological, Social, or Security viewpoints using keyword-weighted scoring across 5 categories.

**Theme extraction** — TF-IDF with bigram support and domain-specific stop words surfaces meaningful shared themes across articles (e.g. "compute scale", "market share") rather than noise words.

**Cosine similarity analysis** — detects when multiple sources are covering the same angle, flagging redundant perspectives.

**Bias detection** — warns when ≥80% of sources share the same perspective type, prompting the user to seek more diverse coverage.

**Relevance scoring** — every source contribution is shown with a numeric score and visual bar, making the system's reasoning transparent.

**Structured output** — results saved as both a formatted terminal report and a JSON file for downstream use.

**CLI interface** — supports `--topic`, `--sources`, and `--max` flags for flexible usage.

---

## Tech Stack

| Component | Technology |
|---|---|
| News retrieval | NewsAPI (`/v2/everything`) |
| Summarization model | `facebook/bart-large-cnn` via HuggingFace Transformers |
| Theme extraction | TF-IDF with bigrams (scikit-learn) |
| Similarity analysis | Cosine similarity on TF-IDF vectors |
| Perspective classification | Keyword-weighted multi-category scoring |
| Runtime | Python 3.12, PyTorch |
| Output formats | Structured terminal report + JSON |

---

## Project Structure

```
├── src/
│   ├── data_collection/
│   │   └── fetch_news.py        # NewsAPI integration + relevance scoring
│   ├── preprocessing/
│   │   └── clean_text.py        # Text cleaning and noise removal
│   ├── summarization/
│   │   └── summarize.py         # BART-based chunked summarization
│   ├── comparison/
│   │   └── compare.py           # TF-IDF, cosine similarity, perspective classification
│   ├── neutral_generation/
│   │   └── generate.py          # Final summary generation + bias detection
│   └── config.py                # Environment config
├── data/
│   └── raw/                     # Saved raw article JSON per query
├── outputs/
│   └── summaries/               # Saved structured output JSON per run
├── main.py                      # Pipeline orchestrator + CLI
├── requirements.txt
└── .env.example
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/perspective-aware-news-summarization.git
cd perspective-aware-news-summarization
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your API key**
```bash
cp .env.example .env
# Edit .env and add your NewsAPI key:
# NEWS_API_KEY=your_key_here
```
Get a free API key at [newsapi.org](https://newsapi.org).

---

## Usage

**Interactive mode**
```bash
python main.py
# Enter news topic: climate change policy
```

**CLI mode**
```bash
python main.py --topic "climate change policy"
python main.py --topic "electric vehicles" --sources 30 --max 10
```

**CLI arguments**

| Argument | Default | Description |
|---|---|---|
| `--topic` | (prompted) | News topic to summarize |
| `--sources` | 20 | Number of articles to fetch from NewsAPI |
| `--max` | 8 | Maximum articles to process after filtering |

**Output files**

Each run saves two files automatically:
- `data/raw/<topic>_<timestamp>.json` — raw API response
- `outputs/summaries/latest_summary.json` — structured final output

---

## How It Works

### 1. Relevance Scoring
Rather than a simple keyword match, each article receives a weighted relevance score:

```
score = (title_matches × 3 + description_matches × 2 + content_matches × 1)
        ─────────────────────────────────────────────────────────────────────
                            max_possible_score
```

Articles below a 0.15 threshold are discarded before any NLP processing begins.

### 2. Chunked Summarization
BART has a 1024-token input limit. Long articles are split into 400-word chunks, each summarized independently, then the chunk summaries are joined and re-summarized into a single coherent output.

### 3. Perspective Classification
Each article summary is scored across five categories (Economic, Political, Technological, Social, Security) using weighted keyword matching. The category with the highest score wins. If no keywords match, the article is classified as "General".

### 4. Theme Extraction
TF-IDF is computed across all article summaries with bigram support (`ngram_range=(1,2)`). The top-scoring terms after stop word removal are surfaced as common themes — these represent what multiple sources are collectively emphasizing.

### 5. Neutral Summary Generation
The top 5 article summaries (filtered by a final topic guard) are concatenated and passed through BART a second time to produce a single neutral summary that blends perspectives rather than amplifying any single source.

---

## Limitations and Future Work

- **NewsAPI free tier** returns limited results for compound queries. A paid tier or alternative source (GDELT, MediaStack) would significantly improve coverage.
- **Perspective classification** uses keyword heuristics. A fine-tuned classifier (e.g. on AllSides or Media Bias/Fact Check data) would be more accurate.
- **Summarization** uses a single BART model. Future versions could use source-specific models or larger LLMs for richer summaries.
- **No caching** — repeated queries re-fetch and re-summarize. Adding Redis or SQLite caching would speed up repeat runs significantly.
- **Language support** — currently English only. Multilingual models (mBART, mT5) could extend coverage.

---

## Example Topics to Try

```bash
python main.py --topic "climate change 2025"
python main.py --topic "US China trade war"
python main.py --topic "electric vehicle market"
python main.py --topic "cryptocurrency regulation"
python main.py --topic "generative AI ethics"
```

---

## License

MIT License — see `LICENSE` for details.
