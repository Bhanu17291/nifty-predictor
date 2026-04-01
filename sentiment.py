import os
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def fetch_sentiment():
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return None, []
    try:
        from newsapi import NewsApiClient
        client = NewsApiClient(api_key=api_key)
        articles = client.get_everything(
            q="Nifty OR NSE OR India stock market",
            language="en", page_size=10, sort_by="publishedAt"
        )["articles"]
    except Exception:
        return None, []

    analyzer = SentimentIntensityAnalyzer()
    scored = []
    for a in articles:
        score = analyzer.polarity_scores(a["title"])["compound"]
        scored.append({"title": a["title"], "score": score, "url": a["url"]})
    if not scored:
        return None, []
    avg = sum(s["score"] for s in scored) / len(scored)
    return round(avg, 4), scored[:3]

def render_sentiment():
    st.subheader("Market sentiment")
    avg, headlines = fetch_sentiment()
    if avg is None:
        st.info("Set NEWS_API_KEY in .env to enable sentiment.")
        return
    label = "Bullish" if avg > 0.05 else ("Bearish" if avg < -0.05 else "Neutral")
    color = "green" if avg > 0.05 else ("red" if avg < -0.05 else "gray")
    st.metric("Sentiment score", f"{avg:+.3f}", label)
    for h in headlines:
        score_str = f"{h['score']:+.2f}"
        st.markdown(f"- [{h['title'][:80]}...]({h['url']}) `{score_str}`")