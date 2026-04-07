"""
main.py — FastAPI backend serving the sentiment analysis pipeline.

ENDPOINTS:
- GET /                     → health check
- GET /sentiment/{ticker}   → fetch news + run FinBERT sentiment
- GET /stock/{ticker}       → get recent stock price data
- GET /predict/{ticker}     → full pipeline: sentiment + technical features → ML prediction
- GET /history/{ticker}     → sentiment trend over last N days
- GET /compare              → compare sentiment across multiple tickers
"""

from fastapi import FastAPI, HTTPException, Query
from src.data_pipeline import get_news, get_stock_data
from src.sentiment import get_sentiment
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(
    title="Financial Sentiment Analyzer",
    description="Analyze financial news sentiment using FinBERT and predict stock movement.",
    version="2.0.0",
)

# Load model and scaler once at startup
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully")
else:
    print("⚠️  Model files not found. /predict endpoint will be unavailable.")

FEATURE_COLS = [
    "avg_positive", "avg_negative", "avg_neutral", "sentiment_std",
    "news_count", "pos_ratio", "neg_ratio", "sentiment_spread",
    "daily_return", "price_vs_ma5", "price_vs_ma10",
    "volatility_5d", "volume_ratio", "momentum_3d",
]

FEATURE_DESCRIPTIONS = {
    "avg_positive": "Average positive sentiment score",
    "avg_negative": "Average negative sentiment score",
    "avg_neutral": "Average neutral sentiment score",
    "sentiment_std": "Sentiment score variability",
    "news_count": "Number of news articles today",
    "pos_ratio": "Ratio of positive headlines",
    "neg_ratio": "Ratio of negative headlines",
    "sentiment_spread": "Positive minus negative sentiment",
    "daily_return": "Today's stock return",
    "price_vs_ma5": "Price relative to 5-day average",
    "price_vs_ma10": "Price relative to 10-day average",
    "volatility_5d": "5-day rolling volatility",
    "volume_ratio": "Today's volume vs 5-day average",
    "momentum_3d": "3-day price momentum",
}


def build_sentiment_features(news_df):
    sentiments = []
    headlines = news_df["headline"].tolist()[:30]
    for headline in headlines:
        s = get_sentiment(headline)
        sentiments.append(s)
    if not sentiments:
        return None
    pos_scores = [s["positive"] for s in sentiments]
    neg_scores = [s["negative"] for s in sentiments]
    neu_scores = [s["neutral"] for s in sentiments]
    labels = [s["label"] for s in sentiments]
    pos_count = labels.count("positive")
    neg_count = labels.count("negative")
    n = len(sentiments)
    return {
        "avg_positive": float(np.mean(pos_scores)),
        "avg_negative": float(np.mean(neg_scores)),
        "avg_neutral": float(np.mean(neu_scores)),
        "sentiment_std": float(np.std(pos_scores)),
        "news_count": n,
        "pos_ratio": pos_count / n,
        "neg_ratio": neg_count / n,
        "sentiment_spread": float(np.mean(pos_scores) - np.mean(neg_scores)),
        "_headlines": [
            {
                "headline": headlines[i],
                "sentiment": sentiments[i]["label"],
                "positive": round(sentiments[i]["positive"], 3),
                "negative": round(sentiments[i]["negative"], 3),
                "neutral": round(sentiments[i]["neutral"], 3),
            }
            for i in range(n)
        ]
    }


def build_technical_features(stock_df):
    if stock_df.empty or len(stock_df) < 5:
        return None
    df = stock_df.copy().sort_values("date")
    df["ma_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["ma_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["price_vs_ma5"] = df["Close"] / df["ma_5"]
    df["price_vs_ma10"] = df["Close"] / df["ma_10"]
    df["volatility_5d"] = df["daily_return"].rolling(window=5, min_periods=1).std()
    df["vol_ma5"] = df["Volume"].rolling(window=5, min_periods=1).mean()
    df["volume_ratio"] = df["Volume"] / df["vol_ma5"]
    df["momentum_3d"] = df["Close"].pct_change(periods=3)
    latest = df.iloc[-1]
    return {
        "daily_return": float(latest["daily_return"]) if pd.notna(latest["daily_return"]) else 0.0,
        "price_vs_ma5": float(latest["price_vs_ma5"]),
        "price_vs_ma10": float(latest["price_vs_ma10"]),
        "volatility_5d": float(latest["volatility_5d"]) if pd.notna(latest["volatility_5d"]) else 0.0,
        "volume_ratio": float(latest["volume_ratio"]) if pd.notna(latest["volume_ratio"]) else 1.0,
        "momentum_3d": float(latest["momentum_3d"]) if pd.notna(latest["momentum_3d"]) else 0.0,
        "_latest_close": round(float(latest["Close"]), 2),
        "_daily_return_pct": round(float(latest["daily_return"]) * 100, 2) if pd.notna(latest["daily_return"]) else 0.0,
    }


@app.get("/")
def health_check():
    return {
        "status": "running",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "endpoints": ["/sentiment/{ticker}", "/stock/{ticker}", "/predict/{ticker}", "/history/{ticker}", "/compare"]
    }


@app.get("/sentiment/{ticker}")
def analyze_sentiment(ticker: str, days: int = 7):
    """Fetch recent news for a ticker and analyze sentiment with FinBERT."""
    ticker = ticker.upper()
    news_df = get_news(ticker, days_back=days)
    if news_df.empty:
        raise HTTPException(status_code=404, detail=f"No news found for {ticker}")
    headlines = news_df["headline"].tolist()[:20]
    results = []
    for headline in headlines:
        sentiment = get_sentiment(headline)
        results.append({
            "headline": headline,
            "sentiment": sentiment["label"],
            "positive": round(sentiment["positive"], 3),
            "negative": round(sentiment["negative"], 3),
            "neutral": round(sentiment["neutral"], 3),
        })
    avg_pos = sum(r["positive"] for r in results) / len(results)
    avg_neg = sum(r["negative"] for r in results) / len(results)
    avg_neu = sum(r["neutral"] for r in results) / len(results)
    return {
        "ticker": ticker,
        "headlines_analyzed": len(results),
        "summary": {
            "avg_positive": round(avg_pos, 3),
            "avg_negative": round(avg_neg, 3),
            "avg_neutral": round(avg_neu, 3),
            "positive_count": sum(1 for r in results if r["sentiment"] == "positive"),
            "negative_count": sum(1 for r in results if r["sentiment"] == "negative"),
            "neutral_count": sum(1 for r in results if r["sentiment"] == "neutral"),
            "overall_sentiment": max(
                [("positive", avg_pos), ("negative", avg_neg), ("neutral", avg_neu)],
                key=lambda x: x[1]
            )[0],
        },
        "headlines": results,
    }


@app.get("/stock/{ticker}")
def get_stock_info(ticker: str):
    """Get recent stock price data for a ticker."""
    ticker = ticker.upper()
    stock_df = get_stock_data(ticker, period="1mo")
    if stock_df.empty:
        raise HTTPException(status_code=404, detail=f"No stock data for {ticker}")
    latest = stock_df.iloc[-1]
    prev = stock_df.iloc[-2] if len(stock_df) > 1 else latest
    return {
        "ticker": ticker,
        "latest_close": round(float(latest["Close"]), 2),
        "previous_close": round(float(prev["Close"]), 2),
        "daily_return": round(float(latest["daily_return"]) * 100, 2) if pd.notna(latest["daily_return"]) else 0,
        "data_points": len(stock_df),
    }


@app.get("/predict/{ticker}")
def predict(ticker: str, days: int = 7):
    """
    Full pipeline: fetch live news + stock data → build features → ML prediction.
    Returns UP/DOWN with confidence score and top contributing factors.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Make sure models/best_model.pkl exists."
        )
    ticker = ticker.upper()

    news_df = get_news(ticker, days_back=days)
    if news_df.empty:
        raise HTTPException(status_code=404, detail=f"No news found for {ticker}")

    sentiment_features = build_sentiment_features(news_df)
    if sentiment_features is None:
        raise HTTPException(status_code=422, detail="Could not compute sentiment features")

    stock_df = get_stock_data(ticker, period="3mo")
    if stock_df.empty:
        raise HTTPException(status_code=404, detail=f"No stock data for {ticker}")

    technical_features = build_technical_features(stock_df)
    if technical_features is None:
        raise HTTPException(status_code=422, detail="Not enough stock data for technical features")

    all_features = {**sentiment_features, **technical_features}
    feature_vector = pd.DataFrame([[all_features[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    feature_vector_scaled = scaler.transform(feature_vector)

    prediction = int(model.predict(feature_vector_scaled)[0])
    probabilities = model.predict_proba(feature_vector_scaled)[0]
    confidence = float(max(probabilities))

    top_factors = []
    if hasattr(model, "coef_"):
        contributions = model.coef_[0] * feature_vector_scaled[0]
        top_idx = np.argsort(np.abs(contributions))[::-1][:3]
        for idx in top_idx:
            feat_name = FEATURE_COLS[idx]
            direction = "bullish" if contributions[idx] > 0 else "bearish"
            top_factors.append({
                "feature": feat_name,
                "description": FEATURE_DESCRIPTIONS[feat_name],
                "direction": direction,
                "contribution": round(float(contributions[idx]), 4),
            })

    return {
        "ticker": ticker,
        "prediction": "UP" if prediction == 1 else "DOWN",
        "confidence": round(confidence * 100, 1),
        "probability_up": round(float(probabilities[1]) * 100, 1),
        "probability_down": round(float(probabilities[0]) * 100, 1),
        "top_factors": top_factors,
        "current_price": technical_features["_latest_close"],
        "todays_return": technical_features["_daily_return_pct"],
        "headlines_used": len(sentiment_features["_headlines"]),
        "overall_sentiment": max(
            [("positive", sentiment_features["avg_positive"]),
             ("negative", sentiment_features["avg_negative"]),
             ("neutral", sentiment_features["avg_neutral"])],
            key=lambda x: x[1]
        )[0],
        "sentiment_summary": {
            "avg_positive": round(sentiment_features["avg_positive"], 3),
            "avg_negative": round(sentiment_features["avg_negative"], 3),
            "avg_neutral": round(sentiment_features["avg_neutral"], 3),
        },
        "recent_headlines": sentiment_features["_headlines"][:5],
        "disclaimer": "For educational purposes only. Not financial advice.",
    }


@app.get("/history/{ticker}")
def sentiment_history(ticker: str, days: int = 30):
    """Returns day-by-day sentiment trend for the historical chart."""
    ticker = ticker.upper()
    news_df = get_news(ticker, days_back=days)
    if news_df.empty:
        raise HTTPException(status_code=404, detail=f"No news found for {ticker}")

    results = []
    for _, row in news_df.iterrows():
        s = get_sentiment(row["headline"])
        results.append({
            "date": str(row["date"]),
            "positive": s["positive"],
            "negative": s["negative"],
            "neutral": s["neutral"],
            "headline": row["headline"],
        })

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date").agg(
        avg_positive=("positive", "mean"),
        avg_negative=("negative", "mean"),
        avg_neutral=("neutral", "mean"),
        news_count=("headline", "count"),
    ).reset_index().sort_values("date")

    return {
        "ticker": ticker,
        "days": days,
        "history": [
            {
                "date": str(row["date"].date()),
                "avg_positive": round(row["avg_positive"], 3),
                "avg_negative": round(row["avg_negative"], 3),
                "avg_neutral": round(row["avg_neutral"], 3),
                "news_count": int(row["news_count"]),
            }
            for _, row in daily.iterrows()
        ]
    }


@app.get("/compare")
def compare_tickers(
    tickers: str = Query(..., description="Comma-separated tickers e.g. AAPL,MSFT,TSLA"),
    days: int = 7
):
    """Compare sentiment across multiple tickers side by side."""
    ticker_list = [t.strip().upper() for t in tickers.split(",")][:6]
    results = {}
    for ticker in ticker_list:
        try:
            news_df = get_news(ticker, days_back=days)
            if news_df.empty:
                continue
            headlines = news_df["headline"].tolist()[:15]
            sentiments = [get_sentiment(h) for h in headlines]
            avg_pos = float(np.mean([s["positive"] for s in sentiments]))
            avg_neg = float(np.mean([s["negative"] for s in sentiments]))
            avg_neu = float(np.mean([s["neutral"] for s in sentiments]))
            results[ticker] = {
                "avg_positive": round(avg_pos, 3),
                "avg_negative": round(avg_neg, 3),
                "avg_neutral": round(avg_neu, 3),
                "news_count": len(headlines),
                "overall_sentiment": max(
                    [("positive", avg_pos), ("negative", avg_neg), ("neutral", avg_neu)],
                    key=lambda x: x[1]
                )[0],
            }
        except Exception:
            continue
    if not results:
        raise HTTPException(status_code=404, detail="No data found for any tickers")
    return {"tickers": list(results.keys()), "days": days, "comparison": results}