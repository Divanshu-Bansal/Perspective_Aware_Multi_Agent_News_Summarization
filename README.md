# 🗞️ Perspective-Aware Multi-Agent News Summarization System

> Automatically collects, processes, and summarizes news from multiple sources — producing a balanced, bias-aware view of any topic with transparent source attribution and perspective analysis.

---

## What it does

When you enter a news topic, the system:

1. **Fetches articles** from three independent news APIs in parallel
2. **Cleans and filters** content through a multi-layer relevance pipeline
3. **Summarizes each article** individually using a transformer-based NLP model
4. **Compares perspectives** across sources using TF-IDF and cosine similarity
5. **Generates a neutral summary** that reflects multiple viewpoints rather than a single biased source
6. **Tracks every run** as a ClearML experiment with full metrics logging

All of this runs through a Streamlit UI with progressive streaming — you watch each step happen in real time.

---

## Screenshots

**1. User Inputs a topic to search**
![User input to search for a topic](https://github.com/Divanshu-Bansal/Perspective_Aware_Multi_Agent_News_Summarization/blob/implementing_streamlit_UI/docs/screenshots/Step%201.png)

**2. Summarizing articles with relevance scores**
![Article cards with relevance scores](https://github.com/Divanshu-Bansal/Perspective_Aware_Multi_Agent_News_Summarization/blob/implementing_streamlit_UI/docs/screenshots/Step%202.png)

**3. Perspective breakdown and neutral summary generation**
![Perspective breakdown, themes and neutral summary output](https://github.com/Divanshu-Bansal/Perspective_Aware_Multi_Agent_News_Summarization/blob/implementing_streamlit_UI/docs/screenshots/Step%203%20%26%204.png)

**4. Task logs in ClearML**
![Task logs in ClearML](https://github.com/Divanshu-Bansal/Perspective_Aware_Multi_Agent_News_Summarization/blob/implementing_streamlit_UI/docs/screenshots/ClearML.png)

---

## Live demo

The app is deployed via GitHub Actions with a Cloudflare tunnel. To launch it:

1. Go to the **Actions** tab in this repository
2. Click **Deploy Streamlit App**
3. Click **Run workflow → Run workflow**
4. Wait ~6 minutes for the model to load
5. Click the Cloudflare URL printed in the workflow logs

---

## Architecture

![Architecture diagram of the application](https://github.com/Divanshu-Bansal/Perspective_Aware_Multi_Agent_News_Summarization/blob/implementing_streamlit_UI/docs/Architecture/multi_agent_news_summarization_pipeline.png)
---

## Key features

**Multi-source aggregation** — fetches up to 50 articles per query by querying NewsAPI, The Guardian, and GNews in parallel. Articles are deduplicated across sources and ranked by a weighted relevance score (title matches weighted 3×, description 2×, content 1×).

**Intelligent filtering** — five-layer pipeline that enforces minimum content length, topic relevance threshold (≥ 0.25), source diversity, exclusion of entertainment and spam domains, and keyword-based category filtering.

**Transformer-based summarization** — uses `facebook/bart-large-cnn` with chunked inference to handle articles of any length. Long articles are split into 400-word windows, each summarized independently, then combined.

**Perspective classification** — classifies each source into one of five viewpoint categories (Economic, Political, Technological, Social, Security) using keyword-weighted scoring. Reports a perspective diversity score per run.

**Theme extraction** — TF-IDF with bigram support and domain-specific stop words surfaces meaningful shared themes across articles. Topic words are automatically excluded from themes to avoid trivial results.

**Cosine similarity analysis** — detects when multiple sources are covering the same angle, flagging redundant perspectives and identifying the most unique coverage.

**Bias detection** — warns when ≥ 80% of sources share the same perspective type, prompting the user to seek more diverse coverage.

**ClearML experiment tracking** — every pipeline run is logged as a ClearML task with fetch metrics, per-article relevance scores, perspective distribution, common themes, and the final summary.

**Progressive streaming UI** — Streamlit app shows each step as it happens. Article cards appear one by one, summaries fill in as they complete, perspective badges appear after comparison, and the neutral summary appears last.

**CI/CD deployment** — GitHub Actions workflow installs dependencies, configures the environment, starts Streamlit, opens a Cloudflare tunnel, and prints the public URL in the workflow logs.

---

## Tech stack

| Component | Technology |
|---|---|
| News APIs | NewsAPI · The Guardian API · GNews API |
| Summarization | `facebook/bart-large-cnn` via HuggingFace Transformers |
| Theme extraction | TF-IDF with bigrams (scikit-learn) |
| Similarity analysis | Cosine similarity on TF-IDF vectors |
| Perspective classification | Keyword-weighted multi-category scoring |
| Experiment tracking | ClearML |
| UI framework | Streamlit |
| Visualizations | Plotly Express |
| Deployment | GitHub Actions + Cloudflare tunnel |
| Runtime | Python 3.10 · PyTorch |

---

## Project structure

```
├── src/
│   ├── data_collection/
│   │   └── fetch_news.py          # Multi-source parallel fetching + relevance scoring
│   ├── preprocessing/
│   │   └── clean_text.py          # Text cleaning and noise removal
│   ├── summarization/
│   │   └── summarize.py           # BART-based chunked summarization
│   ├── comparison/
│   │   └── compare.py             # TF-IDF, cosine similarity, perspective classification
│   ├── neutral_generation/
│   │   └── generate.py            # Neutral summary generation + bias detection
│   ├── tracking/
│   │   └── clearml_tracker.py     # ClearML experiment logging
│   └── config.py                  # Environment configuration
├── .github/
│   └── workflows/
│       └── deploy.yml             # GitHub Actions CI/CD pipeline
├── data/
│   └── raw/                       # Saved raw article JSON per query
├── outputs/
│   └── summaries/                 # Saved structured output JSON per run
├── app.py                         # Streamlit UI with progressive streaming
├── main.py                        # CLI pipeline orchestrator
├── run_on_colab.ipynb             # Google Colab deployment notebook
├── requirements.txt
└── .env.example
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/Divanshu-Bansal/Perspective_Aware_Multi_Agent_News_Summarization.git
cd Perspective_Aware_Multi_Agent_News_Summarization
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

**4. Configure API keys**
```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:
```
NEWS_API_KEY=your_newsapi_key
GUARDIAN_API_KEY=your_guardian_key
GNEWS_API_KEY=your_gnews_key
CLEARML_API_ACCESS_KEY=your_clearml_access_key
CLEARML_API_SECRET_KEY=your_clearml_secret_key
CLEARML_API_HOST=https://api.clear.ml
```

**5. Configure ClearML**
```bash
clearml-init
```

---

## Usage

### Streamlit UI
```bash
streamlit run app.py
```

### CLI
```bash
python main.py --topic "US China trade war"
python main.py --topic "electric vehicle market" --sources 30 --max 10
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--topic` | (prompted) | News topic to summarize |
| `--sources` | 20 | Articles to fetch per source |
| `--max` | 8 | Max articles to process after filtering |

---

## API keys — all free

| API | Free tier | Sign up |
|---|---|---|
| NewsAPI | 100 requests/day | [newsapi.org](https://newsapi.org) |
| The Guardian | Unlimited | [open-platform.theguardian.com](https://open-platform.theguardian.com/access/) |
| GNews | 100 requests/day | [gnews.io](https://gnews.io) |
| ClearML | Free hosted | [app.clear.ml](https://app.clear.ml) |

---

## GitHub Actions deployment

The repository includes a workflow that deploys the app automatically.

**Setup — add these secrets to your repository** (Settings → Secrets → Actions):
```
NEWS_API_KEY
GUARDIAN_API_KEY
GNEWS_API_KEY
CLEARML_API_ACCESS_KEY
CLEARML_API_SECRET_KEY
```

**Trigger options:**
- Push to `main` → auto-deploys
- Actions tab → Deploy Streamlit App → Run workflow → manual trigger

**The workflow:**
1. Sets up Python 3.10 on an Ubuntu runner
2. Installs dependencies with pip caching
3. Configures ClearML and environment variables
4. Installs Cloudflare tunnel
5. Starts Streamlit on port 8501
6. Opens a Cloudflare tunnel and prints the public URL
7. Keeps the app live for up to 6 hours

---

## Example topics

```bash
python main.py --topic "US China trade war"
python main.py --topic "artificial intelligence regulation"
python main.py --topic "electric vehicle market"
python main.py --topic "cryptocurrency Bitcoin"
python main.py --topic "climate change policy"
python main.py --topic "generative AI jobs"
```

**Tips for best results:**
- Use 2–4 keyword phrases rather than full sentences
- Avoid very niche or hyper-local topics — free API tiers have limited coverage
- Topics with recent news coverage return stronger results

---

## Limitations and future work

**Relevance filtering** currently uses weighted keyword matching. A production version would replace this with zero-shot semantic classification (e.g. `facebook/bart-large-mnli`) for topic-agnostic relevance scoring.

**Perspective classification** uses keyword heuristics. A fine-tuned classifier trained on AllSides or Media Bias/Fact Check data would be significantly more accurate.

**NewsAPI free tier** limits results to the last 30 days and returns at most 100 articles per query. A paid tier or additional sources (GDELT, MediaStack) would improve coverage for niche topics.

**No caching** — repeated queries re-fetch and re-summarize. Adding Redis or SQLite caching would speed up repeat runs significantly.

**Language support** — currently English only. Multilingual models (mBART, mT5) could extend coverage to other languages.
