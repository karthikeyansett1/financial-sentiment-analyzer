"""
data_pipeline.py — Fetches financial news and stock price data.

THE CONCEPT:
We're pulling two types of data:
1. NEWS HEADLINES — unstructured text data (what people are saying about a stock)
2. STOCK PRICES — structured numerical data (what the stock actually did)

Later, we'll combine these to ask: "Does what people SAY predict what the stock DOES?"

WHY THESE SOURCES?
- Finnhub: free, reliable, gives us headlines with timestamps
- yfinance: free, no API key, pulls directly from Yahoo Finance
"""

import finnhub
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

# Initialize Finnhub client
client = finnhub.Client(api_key=FINNHUB_KEY)


def get_news(ticker: str, days_back: int = 90) -> pd.DataFrame:
    """
    Pull news headlines for a stock from Finnhub.
    
    Parameters:
    - ticker: stock symbol (e.g., 'AAPL' for Apple)
    - days_back: how many days of history to pull
    
    Returns: DataFrame with columns [ticker, headline, source, datetime, url]
    
    HOW THE API WORKS:
    - Finnhub's company_news endpoint takes a ticker and date range
    - Returns a list of dictionaries, each representing one news article
    - Free tier allows 60 calls per minute (more than enough)
    """
    end = datetime.now()
    start = end - timedelta(days=days_back)
    
    try:
        raw_news = client.company_news(
            ticker,
            _from=start.strftime("%Y-%m-%d"),
            to=end.strftime("%Y-%m-%d")
        )
    except Exception as e:
        print(f"❌ Error fetching news for {ticker}: {e}")
        return pd.DataFrame()
    
    if not raw_news:
        print(f"⚠️  No news found for {ticker}")
        return pd.DataFrame()
    
    # Convert to DataFrame and keep only what we need
    df = pd.DataFrame(raw_news)
    df = df[["headline", "source", "datetime", "url"]].copy()
    df["ticker"] = ticker
    
    # Convert Unix timestamp to readable datetime
    # Finnhub gives timestamps as Unix epoch (seconds since 1970-01-01)
    df["date"] = pd.to_datetime(df["datetime"], unit="s").dt.date
    df["datetime_readable"] = pd.to_datetime(df["datetime"], unit="s")
    
    # Remove duplicates (same headline can appear multiple times)
    df = df.drop_duplicates(subset=["headline"])
    
    print(f"✅ Pulled {len(df)} headlines for {ticker}")
    return df


def get_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Pull stock price data from Yahoo Finance.
    
    Parameters:
    - ticker: stock symbol
    - period: how far back ('6mo' = 6 months, '1y' = 1 year)
    
    Returns: DataFrame with columns [Date, Open, High, Low, Close, Volume, ticker]
    
    WHAT'S IN STOCK DATA:
    - Open: price when market opened that day
    - High/Low: highest/lowest price during the day
    - Close: price when market closed (this is the most commonly used)
    - Volume: how many shares were traded (high volume = lots of activity)
    
    WHY THIS MATTERS:
    We'll compute the 'next_day_return' from Close prices.
    If next_day_return > 0, the stock went UP. That's our prediction target.
    """
    try:
        stock = yf.download(ticker, period=period, progress=False)
    except Exception as e:
        print(f"❌ Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()
    
    if stock.empty:
        print(f"⚠️  No stock data found for {ticker}")
        return pd.DataFrame()
    
    stock = stock.reset_index()
    
    # FIX: yfinance sometimes returns MultiIndex columns even for single tickers
    # This flattens them back to simple column names
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = [col[0] if col[1] == '' or col[1] == ticker else col[0] 
                         for col in stock.columns]
    
    stock["ticker"] = ticker
    stock["date"] = pd.to_datetime(stock["Date"]).dt.date
    stock = stock.drop(columns=["Date"])  
    
    # Calculate daily return
    stock["daily_return"] = stock["Close"].pct_change()
    
    # Calculate NEXT day's return — this is what we want to PREDICT
    stock["next_day_return"] = stock["daily_return"].shift(-1)
    
    # Binary target: 1 = stock goes up tomorrow, 0 = stock goes down
    stock["target"] = (stock["next_day_return"] > 0).astype(int)
    
    print(f"✅ Pulled {len(stock)} days of stock data for {ticker}")
    return stock


def collect_all_data(tickers: list, news_days: int = 90, stock_period: str = "6mo"):
    """
    Pull news and stock data for a list of tickers.
    
    This is the main function you'll run to collect everything.
    """
    all_news = []
    all_stocks = []
    
    for ticker in tickers:
        print(f"\n📊 Processing {ticker}...")
        
        # Pull news
        news_df = get_news(ticker, days_back=news_days)
        if not news_df.empty:
            all_news.append(news_df)
        
        # Pull stock data
        stock_df = get_stock_data(ticker, period=stock_period)
        if not stock_df.empty:
            all_stocks.append(stock_df)
        
        # Small delay to respect API rate limits
        import time
        time.sleep(1)
    
    # Combine all tickers into single DataFrames
    news_combined = pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame()
    stocks_combined = pd.concat(all_stocks, ignore_index=True) if all_stocks else pd.DataFrame()
    
    print(f"\n📦 Total: {len(news_combined)} headlines, {len(stocks_combined)} stock-days")
    return news_combined, stocks_combined