"""
main.py — FastAPI backend serving the sentiment analysis pipeline.

WHAT IS FastAPI?
A modern Python web framework for building APIs. It's:
- Fast (async by default)
- Auto-generates documentation (Swagger UI)
- Has built-in request validation
- The industry standard for serving ML models

HOW THIS WORKS:
1. User sends a GET request with a stock ticker: /analyze/AAPL
2. Our API fetches recent news for that ticker
3. Runs FinBERT sentiment on each headline
4. Returns sentiment scores + model prediction

WHY API OVER JUST A NOTEBOOK:
- Other applications can call your model programmatically
- It separates the model from the interface
- It's how ML models work in production at real companies
"""

from fastapi import FastAPI, HTTPException
from src.data_pipeline import get_news, get_stock_data
from src.sentiment import get_sentiment
import pandas as pd

app = FastAPI(
    title="Financial Sentiment Analyzer",
    description="Analyze financial news sentiment using FinBERT and predict stock movement.",
    version="1.0.0",
)


@app.get("/")
def health_check():
    """Health check endpoint — confirms the API is running."""
    return {"status": "running", "model": "FinBERT", "version": "1.0.0"}


@app.get("/sentiment/{ticker}")
def analyze_sentiment(ticker: str, days: int = 7):
    """
    Fetch recent news for a ticker and analyze sentiment.
    
    - **ticker**: Stock symbol (e.g., AAPL, MSFT, TSLA)
    - **days**: Number of days of news to fetch (default: 7)
    """
    ticker = ticker.upper()
    
    # Fetch news
    news_df = get_news(ticker, days_back=days)
    if news_df.empty:
        raise HTTPException(status_code=404, detail=f"No news found for {ticker}")
    
    # Limit to most recent 20 headlines
    headlines = news_df["headline"].tolist()[:20]
    
    # Analyze sentiment
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
    
    # Compute summary
    avg_pos = sum(r["positive"] for r in results) / len(results)
    avg_neg = sum(r["negative"] for r in results) / len(results)
    avg_neu = sum(r["neutral"] for r in results) / len(results)
    
    pos_count = sum(1 for r in results if r["sentiment"] == "positive")
    neg_count = sum(1 for r in results if r["sentiment"] == "negative")
    neu_count = sum(1 for r in results if r["sentiment"] == "neutral")
    
    return {
        "ticker": ticker,
        "headlines_analyzed": len(results),
        "summary": {
            "avg_positive": round(avg_pos, 3),
            "avg_negative": round(avg_neg, 3),
            "avg_neutral": round(avg_neu, 3),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "overall_sentiment": max(
                [("positive", avg_pos), ("negative", avg_neg), ("neutral", avg_neu)],
                key=lambda x: x[1]
            )[0],
        },
        "headlines": results,
    }


@app.get("/stock/{ticker}")
def get_stock_info(ticker: str):
    """
    Get recent stock price data for a ticker.
    
    - **ticker**: Stock symbol (e.g., AAPL, MSFT)
    """
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