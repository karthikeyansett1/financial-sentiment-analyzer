"""
dashboard/app.py — Upgraded Streamlit dashboard for the Financial Sentiment Analyzer.

TABS:
1. Predict      — ML prediction + sentiment + top factors + recent headlines
2. History      — 30-day sentiment trend chart
3. Compare      — side-by-side sector sentiment comparison
4. Model Info   — model performance metrics, architecture, key findings
"""

import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .prediction-up {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .prediction-down {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .factor-card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 4px solid #3b82f6;
    }
    .headline-card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
    }
    .metric-card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    h1 { color: #f1f5f9; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── COMPANY → TICKER LOOKUP ───
# Allows users to search by company name instead of memorizing tickers
COMPANY_TICKERS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Meta": "META",
    "Netflix": "NFLX",
    "JPMorgan": "JPM",
    "JP Morgan": "JPM",
    "Johnson & Johnson": "JNJ",
    "ExxonMobil": "XOM",
    "Walmart": "WMT",
    "Berkshire Hathaway": "BRK-B",
    "Visa": "V",
    "Mastercard": "MA",
    "UnitedHealth": "UNH",
    "Procter & Gamble": "PG",
    "Home Depot": "HD",
    "Salesforce": "CRM",
    "Adobe": "ADBE",
    "Intel": "INTC",
    "AMD": "AMD",
    "Advanced Micro Devices": "AMD",
    "Qualcomm": "QCOM",
    "Broadcom": "AVGO",
    "Oracle": "ORCL",
    "Cisco": "CSCO",
    "IBM": "IBM",
    "Uber": "UBER",
    "Airbnb": "ABNB",
    "Spotify": "SPOT",
    "Palantir": "PLTR",
    "Snowflake": "SNOW",
    "CrowdStrike": "CRWD",
    "Datadog": "DDOG",
    "Shopify": "SHOP",
    "Square": "SQ",
    "Block": "SQ",
    "PayPal": "PYPL",
    "Goldman Sachs": "GS",
    "Bank of America": "BAC",
    "Wells Fargo": "WFC",
    "Morgan Stanley": "MS",
    "Pfizer": "PFE",
    "Moderna": "MRNA",
    "Disney": "DIS",
    "Coca-Cola": "KO",
    "PepsiCo": "PEP",
    "Nike": "NKE",
    "Boeing": "BA",
    "Ford": "F",
    "General Motors": "GM",
    "Chevron": "CVX",
    "ConocoPhillips": "COP",
    "Caterpillar": "CAT",
    "Deere": "DE",
    "John Deere": "DE",
    "3M": "MMM",
    "AT&T": "T",
    "Verizon": "VZ",
    "T-Mobile": "TMUS",
    "Lockheed Martin": "LMT",
    "Raytheon": "RTX",
    "Twitter": "X",
    "X Corp": "X",
}

# Build reverse lookup: ticker → company name for display
TICKER_COMPANIES = {v: k for k, v in COMPANY_TICKERS.items()}

def find_matches(query: str) -> list:
    """Return list of (company, ticker) tuples matching the search query."""
    if not query or len(query) < 2:
        return []
    q = query.lower()
    matches = []
    seen_tickers = set()
    for company, ticker in COMPANY_TICKERS.items():
        if q in company.lower() or q in ticker.lower():
            if ticker not in seen_tickers:
                matches.append((company, ticker))
                seen_tickers.add(ticker)
    return matches[:6]  # cap at 6 suggestions

# ─── SIDEBAR ───
st.sidebar.image("https://img.shields.io/badge/FinBERT-Powered-blue?style=for-the-badge", use_container_width=True)
st.sidebar.title("⚙️ Settings")

# Search input
search_query = st.sidebar.text_input(
    "Search company or ticker",
    placeholder="e.g. Apple, TSLA, Microsoft...",
)

# Autocomplete suggestions
ticker = "AAPL"  # default
if search_query:
    matches = find_matches(search_query)
    if matches:
        # Show clickable buttons for each match
        st.sidebar.caption("Select a match:")
        for company, t in matches:
            if st.sidebar.button(f"{company} ({t})", key=f"btn_{t}", use_container_width=True):
                ticker = t
                st.session_state["selected_ticker"] = t
    else:
        # Treat input as a raw ticker if no matches
        ticker = search_query.upper().strip()
        st.sidebar.caption(f"Using ticker: **{ticker}**")

# Use session state to persist selection after button click
if "selected_ticker" in st.session_state:
    ticker = st.session_state["selected_ticker"]
    company_name = TICKER_COMPANIES.get(ticker, ticker)
    st.sidebar.success(f"✅ Selected: **{company_name}** ({ticker})")

days = st.sidebar.slider("Days of News", min_value=3, max_value=30, value=7)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack**")
st.sidebar.markdown("""
- 🤖 FinBERT (HuggingFace)
- ⚡ FastAPI backend
- 📊 Logistic Regression (AUC 0.778)
- 🗄️ SQLite + SQLAlchemy
- 🐳 Docker
""")

analyze = st.sidebar.button("🔍 Analyze", type="primary", use_container_width=True)

# ─── HEADER ───
st.title("📈 Financial News Sentiment Analyzer")
st.markdown("*Real-time NLP sentiment analysis + ML stock direction prediction powered by FinBERT*")
st.markdown("---")

# ─── TABS ───
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predict", "📅 History", "⚖️ Compare", "🧠 Model Info"])

# ════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════
with tab1:
    if not analyze:
        st.info("👈 Enter a stock ticker in the sidebar and click **Analyze** to get started.")
        st.markdown("""
        ### How it works
        1. **Enter a ticker** — any US stock symbol (AAPL, TSLA, NVDA, etc.)
        2. **Fetches live news** — pulls recent headlines from Finnhub API
        3. **Runs FinBERT** — transformer model fine-tuned on 50,000+ financial articles
        4. **Engineers features** — combines sentiment scores with technical indicators
        5. **ML Prediction** — Logistic Regression model predicts next-day direction
        6. **Explains why** — shows the top 3 features that drove the prediction
        """)
    else:
        with st.spinner(f"Fetching news and running prediction for {ticker}..."):
            try:
                pred_resp = requests.get(f"{API_URL}/predict/{ticker}?days={days}", timeout=180)

                if pred_resp.status_code == 503:
                    st.warning("⚠️ Model not loaded — showing sentiment only. Make sure models/best_model.pkl exists.")
                    pred_data = None
                elif pred_resp.status_code != 200:
                    st.error(f"Error: {pred_resp.json().get('detail', 'Unknown error')}")
                    pred_data = None
                else:
                    pred_data = pred_resp.json()

                if pred_data:
                    # ── PREDICTION CARD ──
                    direction = "up" if pred_data["prediction"] == "UP" else "down"
                    card_class = "prediction-up" if direction == "up" else "prediction-down"
                    arrow = "📈" if direction == "up" else "📉"
                    color = "#10b981" if direction == "up" else "#ef4444"

                    st.markdown(f"""
                    <div class="{card_class}">
                        <h1 style="color:{color}; font-size:3rem; margin:0">{arrow} {pred_data['prediction']}</h1>
                        <p style="color:#d1fae5 if direction=='up' else #fee2e2; font-size:1.2rem; margin:8px 0">
                            <b>{pred_data['confidence']}% confidence</b>
                        </p>
                        <p style="color:#9ca3af; margin:0">
                            ⬆️ {pred_data['probability_up']}% UP &nbsp;|&nbsp; ⬇️ {pred_data['probability_down']}% DOWN
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── METRICS ROW ──
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Current Price", f"${pred_data['current_price']}")
                    m2.metric("Today's Return", f"{pred_data['todays_return']}%",
                              delta=f"{pred_data['todays_return']}%")
                    m3.metric("Overall Sentiment", pred_data["overall_sentiment"].title())
                    m4.metric("Headlines Used", pred_data["headlines_used"])
                    m5.metric("Avg Positive", f"{pred_data['sentiment_summary']['avg_positive']:.1%}")

                    st.markdown("---")

                    # ── TWO COLUMNS: FACTORS + SENTIMENT CHART ──
                    left, right = st.columns([1, 1])

                    with left:
                        st.subheader("🔍 Top Prediction Factors")
                        st.caption("Features that most influenced the model's decision")
                        for factor in pred_data["top_factors"]:
                            bull = factor["direction"] == "bullish"
                            icon = "🟢" if bull else "🔴"
                            direction_label = "Bullish signal" if bull else "Bearish signal"
                            st.markdown(f"""
                            <div class="factor-card">
                                <b>{icon} {factor['description']}</b><br>
                                <span style="color:#9ca3af; font-size:0.85rem">
                                    {direction_label} · contribution: {factor['contribution']:+.4f}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)

                    with right:
                        st.subheader("📊 Sentiment Distribution")
                        summary = pred_data["sentiment_summary"]
                        sent_df = pd.DataFrame({
                            "Sentiment": ["Positive", "Neutral", "Negative"],
                            "Score": [
                                summary["avg_positive"],
                                summary["avg_neutral"],
                                summary["avg_negative"]
                            ]
                        })
                        fig = px.pie(
                            sent_df, values="Score", names="Sentiment",
                            color="Sentiment",
                            color_discrete_map={
                                "Positive": "#10b981",
                                "Neutral": "#6b7280",
                                "Negative": "#ef4444"
                            },
                            hole=0.4
                        )
                        fig.update_layout(
                            height=300,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#f1f5f9",
                            margin=dict(t=20, b=20, l=20, r=20),
                            showlegend=True,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")

                    # ── RECENT HEADLINES ──
                    st.subheader(f"📰 Recent Headlines for {ticker}")
                    for item in pred_data["recent_headlines"]:
                        sentiment = item["sentiment"]
                        emoji = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "🟡"
                        confidence = max(item["positive"], item["negative"], item["neutral"])
                        st.markdown(f"""
                        <div class="headline-card">
                            {emoji} <b>{sentiment.upper()}</b>
                            <span style="color:#9ca3af"> ({confidence:.0%})</span>
                            — {item['headline']}
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.caption("⚠️ For educational purposes only. Not financial advice.")

            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot connect to the API. Make sure it's running: `uvicorn api.main:app --reload`")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ════════════════════════════════════════
# TAB 2 — HISTORY
# ════════════════════════════════════════
with tab2:
    st.subheader(f"📅 30-Day Sentiment Trend")

    if not analyze:
        st.info("👈 Click **Analyze** in the sidebar to load sentiment history.")
    else:
        with st.spinner(f"Loading sentiment history for {ticker}..."):
            try:
                hist_resp = requests.get(f"{API_URL}/history/{ticker}?days=30", timeout=300)

                if hist_resp.status_code == 200:
                    hist_data = hist_resp.json()
                    history = hist_data["history"]

                    if not history:
                        st.warning("No historical data available.")
                    else:
                        df = pd.DataFrame(history)
                        df["date"] = pd.to_datetime(df["date"])

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df["date"], y=df["avg_positive"],
                            name="Positive", line=dict(color="#10b981", width=2),
                            fill="tozeroy", fillcolor="rgba(16,185,129,0.1)"
                        ))
                        fig.add_trace(go.Scatter(
                            x=df["date"], y=df["avg_negative"],
                            name="Negative", line=dict(color="#ef4444", width=2),
                            fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"
                        ))
                        fig.add_trace(go.Scatter(
                            x=df["date"], y=df["avg_neutral"],
                            name="Neutral", line=dict(color="#6b7280", width=1.5, dash="dot"),
                        ))
                        fig.update_layout(
                            title=f"Daily Average Sentiment Scores — {ticker}",
                            xaxis_title="Date",
                            yaxis_title="Sentiment Score",
                            height=450,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#f1f5f9",
                            legend=dict(bgcolor="rgba(0,0,0,0)"),
                            xaxis=dict(gridcolor="#2d3748"),
                            yaxis=dict(gridcolor="#2d3748"),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # News volume bar chart
                        fig2 = px.bar(
                            df, x="date", y="news_count",
                            title="Daily News Volume",
                            color="news_count",
                            color_continuous_scale="Blues",
                        )
                        fig2.update_layout(
                            height=250,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#f1f5f9",
                            xaxis=dict(gridcolor="#2d3748"),
                            yaxis=dict(gridcolor="#2d3748"),
                            coloraxis_showscale=False,
                        )
                        st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading history: {str(e)}")

# ════════════════════════════════════════
# TAB 3 — COMPARE
# ════════════════════════════════════════
with tab3:
    st.subheader("⚖️ Sector Sentiment Comparison")
    st.caption("Compare sentiment across multiple tickers side by side")

    default_tickers = "AAPL,MSFT,GOOGL,TSLA,NVDA,JPM"
    compare_input = st.text_input("Tickers to compare (comma-separated)", value=default_tickers)
    compare_days = st.slider("Days of news", min_value=3, max_value=14, value=7, key="compare_days")
    run_compare = st.button("⚖️ Compare", type="primary")

    if run_compare:
        with st.spinner("Analyzing sentiment across tickers..."):
            try:
                comp_resp = requests.get(
                    f"{API_URL}/compare?tickers={compare_input}&days={compare_days}",
                    timeout=600
                )

                if comp_resp.status_code == 200:
                    comp_data = comp_resp.json()
                    comparison = comp_data["comparison"]

                    rows = []
                    for t, v in comparison.items():
                        rows.append({
                            "Ticker": t,
                            "Positive": v["avg_positive"],
                            "Negative": v["avg_negative"],
                            "Neutral": v["avg_neutral"],
                            "News Count": v["news_count"],
                            "Overall": v["overall_sentiment"].title(),
                        })
                    df = pd.DataFrame(rows)

                    # Grouped bar chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="Positive", x=df["Ticker"], y=df["Positive"],
                        marker_color="#10b981"
                    ))
                    fig.add_trace(go.Bar(
                        name="Negative", x=df["Ticker"], y=df["Negative"],
                        marker_color="#ef4444"
                    ))
                    fig.add_trace(go.Bar(
                        name="Neutral", x=df["Ticker"], y=df["Neutral"],
                        marker_color="#6b7280"
                    ))
                    fig.update_layout(
                        barmode="group",
                        title="Sentiment Comparison Across Tickers",
                        xaxis_title="Ticker",
                        yaxis_title="Average Sentiment Score",
                        height=400,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#f1f5f9",
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                        xaxis=dict(gridcolor="#2d3748"),
                        yaxis=dict(gridcolor="#2d3748"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Sentiment heatmap
                    heat_df = df.set_index("Ticker")[["Positive", "Negative", "Neutral"]]
                    fig2 = px.imshow(
                        heat_df.T,
                        color_continuous_scale="RdYlGn",
                        title="Sentiment Heatmap",
                        aspect="auto",
                    )
                    fig2.update_layout(
                        height=250,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#f1f5f9",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    # Summary table
                    st.subheader("Summary Table")
                    st.dataframe(
                        df.style.background_gradient(subset=["Positive"], cmap="Greens")
                                .background_gradient(subset=["Negative"], cmap="Reds"),
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ════════════════════════════════════════
# TAB 4 — MODEL INFO
# ════════════════════════════════════════
with tab4:
    st.subheader("🧠 Model Architecture & Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", "Logistic Regression")
    col2.metric("AUC Score", "0.778")
    col3.metric("F1 Score", "0.667")
    col4.metric("Accuracy", "~60%")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.markdown("### Pipeline Architecture")
        st.markdown("""
        ```
        Finnhub API (news headlines)
              ↓
        FinBERT Sentiment Scoring
        (ProsusAI/finbert — 438MB transformer)
              ↓
        Feature Engineering (14 features)
        ├── Sentiment: avg_positive, avg_negative,
        │   avg_neutral, sentiment_std, news_count,
        │   pos_ratio, neg_ratio, sentiment_spread
        └── Technical: daily_return, price_vs_ma5,
            price_vs_ma10, volatility_5d,
            volume_ratio, momentum_3d
              ↓
        Logistic Regression Classifier
        (time-based train/test split)
              ↓
        FastAPI → Streamlit Dashboard
        ```
        """)

        st.markdown("### Models Compared")
        model_df = pd.DataFrame({
            "Model": ["Logistic Regression ✅", "Random Forest", "XGBoost"],
            "AUC": [0.778, 0.712, 0.695],
            "F1": [0.667, 0.601, 0.589],
            "Accuracy": ["62%", "58%", "56%"],
        })
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown("### Key Findings")
        st.success("""
        **📌 Buy the Rumor, Sell the News**
        Positive sentiment has a *negative* correlation (−0.16)
        with next-day returns. Markets price in good news before
        it's published — negative headlines often precede recovery.
        """)

        st.info("""
        **📌 Technical > Sentiment**
        SHAP analysis shows news_count, volatility_5d, and
        momentum_3d are stronger predictors than raw sentiment
        scores. Context matters more than tone alone.
        """)

        st.warning("""
        **📌 Modest Accuracy Is Expected**
        60% accuracy on stock direction is realistic. If a model
        hit 90%, it would indicate data leakage. Markets are
        efficient — easy predictions get arbitraged away.
        """)

        st.markdown("### Top SHAP Features")
        shap_df = pd.DataFrame({
            "Feature": ["news_count", "volatility_5d", "momentum_3d",
                        "avg_negative", "sentiment_spread"],
            "Importance": [0.312, 0.287, 0.241, 0.198, 0.156],
        })
        fig = px.bar(
            shap_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            title="Feature Importance (SHAP values)"
        )
        fig.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9",
            coloraxis_showscale=False,
            yaxis=dict(gridcolor="#2d3748"),
            xaxis=dict(gridcolor="#2d3748"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Dataset")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    stats_col1.metric("Total Headlines", "2,469")
    stats_col2.metric("Stock-Days", "1,250")
    stats_col3.metric("Tickers", "10")
    stats_col4.metric("Training Rows", "64 (merged)")

    st.caption("""
    Small dataset due to Finnhub free tier date range limitation.
    With a paid tier or additional sources (NewsAPI, Alpha Vantage),
    the dataset could scale to 10,000+ rows improving model robustness.
    """)