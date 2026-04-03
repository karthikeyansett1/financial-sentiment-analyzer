"""
Run FinBERT sentiment analysis on all headlines in the database.
On M3 MacBook Air, ~2500 headlines should take about 5-8 minutes.
"""
from src.database import load_from_db, save_to_db
from src.sentiment import analyze_headlines
import pandas as pd

# Load all news headlines
print("Loading headlines from database...")
news = load_from_db("SELECT * FROM news")
print(f"Found {len(news)} headlines to analyze.\n")

# Run sentiment analysis
headlines = news["headline"].tolist()
results = analyze_headlines(headlines, show_progress=True)

# Convert results to columns and add to the dataframe
sentiment_df = pd.DataFrame(results)
news["sentiment_label"] = sentiment_df["label"]
news["sentiment_positive"] = sentiment_df["positive"]
news["sentiment_negative"] = sentiment_df["negative"]
news["sentiment_neutral"] = sentiment_df["neutral"]

# Save enriched data back to database
save_to_db(news, "news_with_sentiment", if_exists="replace")

# Quick summary
print("\n=== SENTIMENT SUMMARY ===")
print(news["sentiment_label"].value_counts())
print(f"\nAvg positive score: {news['sentiment_positive'].mean():.3f}")
print(f"Avg negative score: {news['sentiment_negative'].mean():.3f}")
print(f"Avg neutral score:  {news['sentiment_neutral'].mean():.3f}")

# Per ticker breakdown
print("\n=== SENTIMENT BY TICKER ===")
ticker_sentiment = news.groupby("ticker")["sentiment_label"].value_counts().unstack(fill_value=0)
print(ticker_sentiment)