"""
dashboard/app.py — Financial Sentiment Analyzer Dashboard
Tabs: Predict | History | Compare | Model Info
"""

import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS: clean, theme-agnostic ───
st.markdown("""
<style>
    .prediction-box {
        border-radius: 10px;
        padding: 28px 24px;
        text-align: center;
        margin-bottom: 8px;
    }
    .prediction-up {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1.5px solid #10b981;
    }
    .prediction-down {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1.5px solid #ef4444;
    }
    .factor-card {
        border-radius: 8px;
        padding: 12px 16px;
        margin: 5px 0;
        border-left: 3px solid #6366f1;
        background-color: rgba(99, 102, 241, 0.06);
    }
    .headline-row {
        padding: 9px 12px;
        margin: 3px 0;
        border-radius: 6px;
        background-color: rgba(100, 116, 139, 0.08);
        font-size: 0.92rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── COMPANY LOOKUP ───
COMPANY_TICKERS = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL",
    "Alphabet": "GOOGL", "Amazon": "AMZN", "Tesla": "TSLA",
    "NVIDIA": "NVDA", "Meta": "META", "Netflix": "NFLX",
    "JPMorgan": "JPM", "JP Morgan": "JPM", "Johnson & Johnson": "JNJ",
    "ExxonMobil": "XOM", "Walmart": "WMT", "Berkshire Hathaway": "BRK-B",
    "Visa": "V", "Mastercard": "MA", "UnitedHealth": "UNH",
    "Procter & Gamble": "PG", "Home Depot": "HD", "Salesforce": "CRM",
    "Adobe": "ADBE", "Intel": "INTC", "AMD": "AMD",
    "Advanced Micro Devices": "AMD", "Qualcomm": "QCOM",
    "Broadcom": "AVGO", "Oracle": "ORCL", "Cisco": "CSCO",
    "IBM": "IBM", "Uber": "UBER", "Airbnb": "ABNB",
    "Spotify": "SPOT", "Palantir": "PLTR", "Snowflake": "SNOW",
    "CrowdStrike": "CRWD", "Datadog": "DDOG", "Shopify": "SHOP",
    "PayPal": "PYPL", "Goldman Sachs": "GS", "Bank of America": "BAC",
    "Wells Fargo": "WFC", "Morgan Stanley": "MS", "Pfizer": "PFE",
    "Moderna": "MRNA", "Disney": "DIS", "Coca-Cola": "KO",
    "PepsiCo": "PEP", "Nike": "NKE", "Boeing": "BA",
    "Ford": "F", "General Motors": "GM", "Chevron": "CVX",
    "AT&T": "T", "Verizon": "VZ", "T-Mobile": "TMUS",
    "Lockheed Martin": "LMT", "Raytheon": "RTX",
}
TICKER_COMPANIES = {v: k for k, v in COMPANY_TICKERS.items()}

def find_matches(query: str) -> list:
    if not query or len(query) < 2:
        return []
    q = query.lower()
    matches, seen = [], set()
    for company, ticker in COMPANY_TICKERS.items():
        if q in company.lower() or q in ticker.lower():
            if ticker not in seen:
                matches.append((company, ticker))
                seen.add(ticker)
    return matches[:5]

# ─── SIDEBAR ───
st.sidebar.title("Settings")
search_query = st.sidebar.text_input("Search company or ticker", placeholder="Apple, TSLA, NVDA...")

ticker = "AAPL"

# When query changes, clear old selection
if search_query != st.session_state.get("last_query", ""):
    st.session_state.pop("selected_ticker", None)
    st.session_state["last_query"] = search_query

if search_query:
    matches = find_matches(search_query)
    if matches:
        st.sidebar.caption("Select a match:")
        for company, t in matches:
            if st.sidebar.button(f"{company}  ({t})", key=f"btn_{t}", use_container_width=True):
                st.session_state["selected_ticker"] = t
                st.session_state["last_query"] = search_query
    else:
        # No company match — use raw input as ticker
        raw = search_query.upper().strip()
        st.session_state["selected_ticker"] = raw

if "selected_ticker" in st.session_state:
    ticker = st.session_state["selected_ticker"]
    company_name = TICKER_COMPANIES.get(ticker, ticker)
    st.sidebar.success(f"Selected: **{company_name}** ({ticker})")
    if st.sidebar.button("Clear selection", use_container_width=True):
        st.session_state.pop("selected_ticker", None)
        st.session_state.pop("last_query", None)
        st.rerun()

days = st.sidebar.slider("Days of news", min_value=3, max_value=30, value=7)

st.sidebar.markdown("---")
st.sidebar.markdown("**Stack**")
st.sidebar.markdown(
    "FinBERT · FastAPI · Logistic Regression · SQLite · Docker",
    unsafe_allow_html=False
)

analyze = st.sidebar.button("Analyze", type="primary", use_container_width=True)

# ─── HEADER ───
st.title("Financial News Sentiment Analyzer")
st.caption("Real-time NLP sentiment analysis and ML stock direction prediction powered by FinBERT")
st.divider()

# ─── TABS ───
tab1, tab2, tab3, tab4 = st.tabs(["Predict", "History", "Compare", "Model Info"])

# ════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════
with tab1:
    if not analyze:
        st.markdown("#### How it works")
        st.markdown("""
        1. Search for a company or enter a ticker symbol in the sidebar
        2. The system fetches recent news headlines from Finnhub API
        3. Each headline is scored by FinBERT — a transformer trained on financial text
        4. Scores are combined with technical stock indicators to build a feature vector
        5. A Logistic Regression model predicts the next-day price direction
        6. The top 3 features driving the prediction are surfaced for transparency
        """)
    else:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                pred_resp = requests.get(f"{API_URL}/predict/{ticker}?days={days}", timeout=180)

                if pred_resp.status_code == 503:
                    st.warning("Model not loaded. Ensure models/best_model.pkl exists.")
                    pred_data = None
                elif pred_resp.status_code != 200:
                    st.error(f"Error: {pred_resp.json().get('detail', 'Unknown error')}")
                    pred_data = None
                else:
                    pred_data = pred_resp.json()

                if pred_data:
                    direction = "up" if pred_data["prediction"] == "UP" else "down"
                    box_class = "prediction-up" if direction == "up" else "prediction-down"
                    color = "#10b981" if direction == "up" else "#ef4444"
                    label = "UP" if direction == "up" else "DOWN"

                    st.markdown(f"""
                    <div class="prediction-box {box_class}">
                        <div style="font-size:2.4rem; font-weight:700; color:{color}; letter-spacing:2px">{label}</div>
                        <div style="font-size:1.1rem; margin-top:6px; opacity:0.85">
                            {pred_data['confidence']}% confidence
                        </div>
                        <div style="font-size:0.9rem; margin-top:4px; opacity:0.6">
                            {pred_data['probability_up']}% up &nbsp;/&nbsp; {pred_data['probability_down']}% down
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("")

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Price", f"${pred_data['current_price']}")
                    m2.metric("Today's Return", f"{pred_data['todays_return']}%",
                              delta=f"{pred_data['todays_return']}%")
                    m3.metric("Sentiment", pred_data["overall_sentiment"].title())
                    m4.metric("Headlines", pred_data["headlines_used"])
                    m5.metric("Avg Positive", f"{pred_data['sentiment_summary']['avg_positive']:.1%}")

                    st.divider()

                    left, right = st.columns(2)

                    with left:
                        st.markdown("#### Top Prediction Factors")
                        st.caption("Features that most influenced the model's decision")
                        for factor in pred_data["top_factors"]:
                            bull = factor["direction"] == "bullish"
                            signal_color = "#10b981" if bull else "#ef4444"
                            signal_label = "Bullish" if bull else "Bearish"
                            st.markdown(f"""
                            <div class="factor-card">
                                <div style="font-weight:600">{factor['description']}</div>
                                <div style="font-size:0.82rem; margin-top:3px; color:{signal_color}">
                                    {signal_label} &nbsp;·&nbsp; contribution: {factor['contribution']:+.4f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    with right:
                        st.markdown("#### Sentiment Distribution")
                        summary = pred_data["sentiment_summary"]
                        sent_df = pd.DataFrame({
                            "Sentiment": ["Positive", "Neutral", "Negative"],
                            "Score": [summary["avg_positive"], summary["avg_neutral"], summary["avg_negative"]]
                        })
                        fig = px.pie(
                            sent_df, values="Score", names="Sentiment",
                            color="Sentiment",
                            color_discrete_map={
                                "Positive": "#10b981",
                                "Neutral": "#94a3b8",
                                "Negative": "#ef4444"
                            },
                            hole=0.45
                        )
                        fig.update_layout(
                            height=280,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=10, b=10, l=10, r=10),
                            showlegend=True,
                        )
                        fig.update_traces(textinfo="percent", textfont_size=13)
                        st.plotly_chart(fig, use_container_width=True)

                    st.divider()

                    st.markdown("#### Recent Headlines")
                    for item in pred_data["recent_headlines"]:
                        sentiment = item["sentiment"]
                        dot_color = "#10b981" if sentiment == "positive" else "#ef4444" if sentiment == "negative" else "#94a3b8"
                        confidence = max(item["positive"], item["negative"], item["neutral"])
                        st.markdown(f"""
                        <div class="headline-row">
                            <span style="color:{dot_color}; font-weight:600">{sentiment.upper()}</span>
                            <span style="opacity:0.5; font-size:0.82rem"> {confidence:.0%}</span>
                            &nbsp;—&nbsp;{item['headline']}
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("")
                    st.caption("For educational purposes only. Not financial advice.")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Make sure it is running: `uvicorn api.main:app --reload`")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ════════════════════════════════════════
# TAB 2 — HISTORY
# ════════════════════════════════════════
with tab2:
    st.markdown("#### 30-Day Sentiment Trend")
    if not analyze:
        st.info("Click Analyze in the sidebar to load sentiment history.")
    else:
        with st.spinner(f"Loading history for {ticker}..."):
            try:
                hist_resp = requests.get(f"{API_URL}/history/{ticker}?days=30", timeout=300)
                if hist_resp.status_code == 200:
                    history = hist_resp.json()["history"]
                    if not history:
                        st.warning("No historical data available.")
                    else:
                        df = pd.DataFrame(history)
                        df["date"] = pd.to_datetime(df["date"])

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df["date"], y=df["avg_positive"], name="Positive",
                            line=dict(color="#10b981", width=2),
                            fill="tozeroy", fillcolor="rgba(16,185,129,0.08)"
                        ))
                        fig.add_trace(go.Scatter(
                            x=df["date"], y=df["avg_negative"], name="Negative",
                            line=dict(color="#ef4444", width=2),
                            fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"
                        ))
                        fig.add_trace(go.Scatter(
                            x=df["date"], y=df["avg_neutral"], name="Neutral",
                            line=dict(color="#94a3b8", width=1.5, dash="dot"),
                        ))
                        fig.update_layout(
                            title=f"Daily Sentiment — {ticker}",
                            xaxis_title="Date",
                            yaxis_title="Avg Sentiment Score",
                            height=420,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            legend=dict(bgcolor="rgba(0,0,0,0)"),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        fig2 = px.bar(
                            df, x="date", y="news_count",
                            title="Daily News Volume",
                            color="news_count",
                            color_continuous_scale="Blues",
                        )
                        fig2.update_layout(
                            height=220,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            coloraxis_showscale=False,
                        )
                        st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ════════════════════════════════════════
# TAB 3 — COMPARE
# ════════════════════════════════════════
with tab3:
    st.markdown("#### Sector Sentiment Comparison")
    st.caption("Compare sentiment across multiple tickers side by side")

    compare_input = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,GOOGL,TSLA,NVDA,JPM")
    compare_days = st.slider("Days of news", min_value=3, max_value=14, value=7, key="compare_days")
    run_compare = st.button("Compare", type="primary")

    if run_compare:
        with st.spinner("Analyzing tickers..."):
            try:
                comp_resp = requests.get(
                    f"{API_URL}/compare?tickers={compare_input}&days={compare_days}", timeout=600
                )
                if comp_resp.status_code == 200:
                    comparison = comp_resp.json()["comparison"]
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

                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="Positive", x=df["Ticker"], y=df["Positive"], marker_color="#10b981"))
                    fig.add_trace(go.Bar(name="Negative", x=df["Ticker"], y=df["Negative"], marker_color="#ef4444"))
                    fig.add_trace(go.Bar(name="Neutral", x=df["Ticker"], y=df["Neutral"], marker_color="#94a3b8"))
                    fig.update_layout(
                        barmode="group",
                        title="Sentiment by Ticker",
                        height=380,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    heat_df = df.set_index("Ticker")[["Positive", "Negative", "Neutral"]]
                    fig2 = px.imshow(heat_df.T, color_continuous_scale="RdYlGn",
                                     title="Sentiment Heatmap", aspect="auto")
                    fig2.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig2, use_container_width=True)

                    st.markdown("#### Summary")
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
    st.markdown("#### Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", "Logistic Regression")
    c2.metric("AUC", "0.778")
    c3.metric("F1 Score", "0.667")
    c4.metric("Accuracy", "~60%")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("#### Pipeline")
        st.code("""
Finnhub API  →  FinBERT Sentiment Scoring
                       ↓
Yahoo Finance →  Technical Indicators
                       ↓
           Feature Engineering (14 features)
           ├── Sentiment: avg_positive, avg_negative,
           │   avg_neutral, news_count, pos_ratio...
           └── Technical: volatility_5d, momentum_3d,
               price_vs_ma5, volume_ratio...
                       ↓
           Logistic Regression (time-based split)
                       ↓
           FastAPI  →  Streamlit Dashboard
        """, language=None)

        st.markdown("#### Models Compared")
        st.dataframe(pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "AUC": [0.778, 0.712, 0.695],
            "F1": [0.667, 0.601, 0.589],
            "Accuracy": ["62%", "58%", "56%"],
            "Selected": ["Yes", "No", "No"],
        }), use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Key Findings")
        st.success("""
        **Buy the Rumor, Sell the News**
        Positive sentiment has a negative correlation (−0.16) with next-day returns.
        Markets price in good news before publication.
        """)
        st.info("""
        **Technical Features Outperform Sentiment**
        SHAP shows news_count, volatility_5d, and momentum_3d are stronger
        predictors than raw sentiment scores alone.
        """)
        st.warning("""
        **Modest Accuracy is Expected**
        ~60% accuracy on stock direction is realistic — not a flaw.
        A 90% accurate model would indicate data leakage.
        """)

        st.markdown("#### SHAP Feature Importance")
        shap_df = pd.DataFrame({
            "Feature": ["news_count", "volatility_5d", "momentum_3d", "avg_negative", "sentiment_spread"],
            "Importance": [0.312, 0.287, 0.241, 0.198, 0.156],
        })
        fig = px.bar(shap_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues")
        fig.update_layout(
            height=260, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Dataset")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Headlines", "2,469")
    d2.metric("Stock-Days", "1,250")
    d3.metric("Tickers", "10")
    d4.metric("Training Rows", "64")
    st.caption("Dataset size limited by Finnhub free tier. Pipeline scales to 10,000+ rows with a paid data source.")
