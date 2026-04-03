"""Quick check that our data looks right."""
from src.database import load_from_db

# Check news
print("=== NEWS DATA ===")
news = load_from_db("SELECT * FROM news LIMIT 5")
print(f"Columns: {list(news.columns)}")
print(f"\nSample headlines:")
for _, row in news.head(3).iterrows():
    print(f"  [{row['ticker']}] {row['headline'][:80]}...")

print(f"\nHeadlines per ticker:")
counts = load_from_db("SELECT ticker, COUNT(*) as count FROM news GROUP BY ticker")
print(counts.to_string(index=False))

# Check stock prices
print("\n=== STOCK DATA ===")
stocks = load_from_db("SELECT * FROM stock_prices LIMIT 5")
print(f"Columns: {list(stocks.columns)}")
print(f"\nSample:")
print(stocks[["ticker", "date", "Close", "daily_return", "target"]].head())

# Check target balance
print("\nTarget distribution (all tickers):")
target = load_from_db("SELECT target, COUNT(*) as count FROM stock_prices GROUP BY target")
print(target.to_string(index=False))