# Perspective_Aware_Multi_Agent_News_Summarization
Perspective-Aware Multi-Agent News Summarization with Bias Detection and Neutral Narrative Generation

A multi-agent NLP system that collects news articles from multiple sources, detects bias and framing differences, generates source-wise summaries, compares perspectives, and produces a neutral narrative with explainable source attribution.

## System Architecture

The system consists of the following agents:

1. Data Collection Agent – fetches articles using News API
2. Preprocessing Agent – cleans and prepares text
3. Bias Detection Agent – identifies political leaning and tone
4. Summarization Agent – generates source-wise summaries
5. Comparison Agent – detects similarities and differences
6. Neutral Generation Agent – produces balanced summary
7. Explainability Module – traces source influence