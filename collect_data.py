"""
collect_data.py — Run this script to collect all news and stock data.

Run with: python collect_data.py
"""

from src.data_pipeline import collect_all_data
from src.database import save_to_db

# These are 10 major stocks across different sectors
# Diversity matters — we don't want all tech stocks
TICKERS = [
    "AAPL",   # Apple — Tech
    "MSFT",   # Microsoft — Tech
    "GOOGL",  # Google — Tech
    "AMZN",   # Amazon — E-commerce/Cloud
    "TSLA",   # Tesla — Auto/EV
    "NVDA",   # Nvidia — Semiconductors
    "JPM",    # JPMorgan — Banking
    "JNJ",    # Johnson & Johnson — Healthcare
    "XOM",    # ExxonMobil — Energy
    "WMT",    # Walmart — Retail
]

print("🚀 Starting data collection...\n")

# Pull everything
news_df, stocks_df = collect_all_data(TICKERS, news_days=90, stock_period="6mo")

# Save to database
if not news_df.empty:
    save_to_db(news_df, "news", if_exists="replace")

if not stocks_df.empty:
    save_to_db(stocks_df, "stock_prices", if_exists="replace")

print("\n✅ Data collection complete!")
print(f"📰 News headlines: {len(news_df)}")
print(f"📈 Stock price days: {len(stocks_df)}")