"""
Streamlit Dashboard — Interactive frontend for the sentiment analyzer.

WHAT IS STREAMLIT?
A Python framework that turns scripts into web apps with zero frontend code.
You write Python, it renders as a web page. Perfect for data science demos.

RUN WITH: streamlit run dashboard/app.py
"""

import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page config
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Financial News Sentiment Analyzer")
st.markdown("*Powered by FinBERT — a transformer model fine-tuned on financial text*")
st.markdown("---")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
days = st.sidebar.slider("Days of News", min_value=1, max_value=30, value=7)

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

if st.sidebar.button("🔍 Analyze", type="primary"):
    
    with st.spinner(f"Fetching news and analyzing sentiment for {ticker}..."):
        try:
            # Call our FastAPI backend
            response = requests.get(f"{API_URL}/sentiment/{ticker}?days={days}", timeout=120)
            
            if response.status_code != 200:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            else:
                data = response.json()
                summary = data["summary"]
                headlines = data["headlines"]
                
                # ─── METRICS ROW ───
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Headlines Analyzed", data["headlines_analyzed"])
                col2.metric("Overall Sentiment", summary["overall_sentiment"].title())
                col3.metric("Positive", f"{summary['avg_positive']:.1%}")
                col4.metric("Negative", f"{summary['avg_negative']:.1%}")
                
                st.markdown("---")
                
                # ─── CHARTS ───
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.subheader("Sentiment Distribution")
                    sent_counts = pd.DataFrame({
                        "Sentiment": ["Positive", "Neutral", "Negative"],
                        "Count": [summary["positive_count"], summary["neutral_count"], summary["negative_count"]]
                    })
                    fig = px.pie(sent_counts, values="Count", names="Sentiment",
                                 color="Sentiment",
                                 color_discrete_map={"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"})
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                
                with chart_col2:
                    st.subheader("Confidence Scores")
                    scores_df = pd.DataFrame(headlines)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="Positive", x=list(range(len(scores_df))), y=scores_df["positive"], marker_color="#22c55e"))
                    fig.add_trace(go.Bar(name="Negative", x=list(range(len(scores_df))), y=scores_df["negative"], marker_color="#ef4444"))
                    fig.add_trace(go.Bar(name="Neutral", x=list(range(len(scores_df))), y=scores_df["neutral"], marker_color="#94a3b8"))
                    fig.update_layout(barmode="stack", height=350, xaxis_title="Headline #", yaxis_title="Score")
                    st.plotly_chart(fig, use_container_width=True)
                
                # ─── HEADLINES TABLE ───
                st.markdown("---")
                st.subheader(f"📰 Recent Headlines for {ticker}")
                
                for item in headlines:
                    sentiment = item["sentiment"]
                    emoji = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "🟡"
                    confidence = max(item["positive"], item["negative"], item["neutral"])
                    
                    st.markdown(
                        f"{emoji} **{sentiment.upper()}** ({confidence:.0%}) — {item['headline']}"
                    )
                
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to the API. Make sure the FastAPI server is running:\n\n`uvicorn api.main:app --reload`")
        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    st.info("👈 Enter a stock ticker and click **Analyze** to get started.")
    
    st.markdown("### How it works")
    st.markdown("""
    1. **Enter a ticker** — any US stock symbol (e.g., AAPL, TSLA, GOOGL)
    2. **Fetches recent news** — pulls headlines from Finnhub API
    3. **Runs FinBERT** — a transformer model fine-tuned on financial text classifies each headline
    4. **Shows results** — sentiment distribution, confidence scores, and individual headline analysis
    """)