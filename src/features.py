"""
features.py — Feature engineering for the prediction model.

WHAT IS FEATURE ENGINEERING?
Raw data isn't directly useful for ML models. We need to create
meaningful "features" (input variables) that capture patterns.

We create two types of features:
1. SENTIMENT FEATURES — derived from news headlines
2. TECHNICAL FEATURES — derived from stock price history

WHY BOTH?
Sentiment alone is a weak signal. Stock technical indicators alone
are also imperfect. But COMBINING them can give the model more
information to work with. The model learns which features matter.
"""

import pandas as pd
import numpy as np
from src.database import load_from_db, save_to_db


def build_sentiment_features(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate headline-level sentiment into daily features per ticker.
    
    Input: one row per headline
    Output: one row per ticker per day
    """
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
    
    daily = news_df.groupby(["ticker", "date"]).agg(
        # Average sentiment scores for the day
        avg_positive=("sentiment_positive", "mean"),
        avg_negative=("sentiment_negative", "mean"),
        avg_neutral=("sentiment_neutral", "mean"),
        
        # How much do sentiment scores vary? High std = mixed opinions
        sentiment_std=("sentiment_positive", "std"),
        
        # Volume of coverage
        news_count=("headline", "count"),
        
        # Counts by label
        pos_count=("sentiment_label", lambda x: (x == "positive").sum()),
        neg_count=("sentiment_label", lambda x: (x == "negative").sum()),
        neu_count=("sentiment_label", lambda x: (x == "neutral").sum()),
    ).reset_index()
    
    # Derived ratios
    daily["pos_ratio"] = daily["pos_count"] / daily["news_count"]
    daily["neg_ratio"] = daily["neg_count"] / daily["news_count"]
    
    # Sentiment spread: how polarized is the coverage?
    daily["sentiment_spread"] = daily["avg_positive"] - daily["avg_negative"]
    
    # Fill NaN std (happens when only 1 headline that day)
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    
    return daily


def build_technical_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create stock technical indicator features.
    
    WHAT THESE MEAN:
    - Moving averages (MA): smoothed price over N days. If price > MA, trend is up.
    - Price vs MA ratio: how far current price is from the trend
    - Rolling volatility: how much the stock has been swinging recently
    - Volume ratio: is today's volume unusual compared to recent average?
    """
    stock_df = stock_df.copy()
    stock_df = stock_df.sort_values(["ticker", "date"])
    
    features = []
    for ticker in stock_df["ticker"].unique():
        df = stock_df[stock_df["ticker"] == ticker].copy()
        
        # Moving averages
        df["ma_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
        df["ma_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
        
        # Price relative to moving average (>1 means price is above trend)
        df["price_vs_ma5"] = df["Close"] / df["ma_5"]
        df["price_vs_ma10"] = df["Close"] / df["ma_10"]
        
        # Rolling volatility (standard deviation of returns over 5 days)
        df["volatility_5d"] = df["daily_return"].rolling(window=5, min_periods=1).std()
        
        # Volume ratio vs 5-day average
        df["vol_ma5"] = df["Volume"].rolling(window=5, min_periods=1).mean()
        df["volume_ratio"] = df["Volume"] / df["vol_ma5"]
        
        # Momentum: return over last 3 days
        df["momentum_3d"] = df["Close"].pct_change(periods=3)
        
        features.append(df)
    
    return pd.concat(features, ignore_index=True)


def build_full_feature_set():
    """
    Main function: load data, build all features, merge, and save.
    """
    print("Loading data...")
    news = load_from_db("SELECT * FROM news_with_sentiment")
    stocks = load_from_db("SELECT * FROM stock_prices")
    
    print("Building sentiment features...")
    sentiment_features = build_sentiment_features(news)
    print(f"  → {len(sentiment_features)} ticker-day rows with sentiment")
    
    print("Building technical features...")
    stocks["date"] = pd.to_datetime(stocks["date"]).dt.date
    tech_features = build_technical_features(stocks)
    print(f"  → {len(tech_features)} ticker-day rows with technical indicators")
    
    print("Merging features...")
    merged = sentiment_features.merge(
        tech_features[["ticker", "date", "Close", "Volume", "daily_return",
                       "next_day_return", "target", "ma_5", "ma_10",
                       "price_vs_ma5", "price_vs_ma10", "volatility_5d",
                       "volume_ratio", "momentum_3d"]],
        on=["ticker", "date"],
        how="inner"
    )
    
    # Drop rows where target is NaN (last row per ticker)
    merged = merged.dropna(subset=["next_day_return"])
    
    print(f"\n✅ Final feature set: {len(merged)} rows, {len(merged.columns)} columns")
    print(f"Features: {list(merged.columns)}")
    
    # Save to database
    save_to_db(merged, "features", if_exists="replace")
    
    return merged


if __name__ == "__main__":
    df = build_full_feature_set()
    print(f"\nTarget distribution:")
    print(df["target"].value_counts())
    print(f"\nSample:")
    print(df.head())