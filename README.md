# 📈 Financial News Sentiment Analyzer

> Real-time financial news sentiment analysis + ML stock direction prediction, powered by FinBERT and deployed via FastAPI + Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FinBERT](https://img.shields.io/badge/FinBERT-HuggingFace-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-2.0-green?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## What This Does

This system pulls live financial news headlines for any US stock ticker, runs each headline through **FinBERT** (a BERT transformer fine-tuned on 50,000+ financial articles) to score sentiment, combines those scores with technical stock indicators, and feeds everything into a trained **Logistic Regression model** that predicts whether the stock will go **UP or DOWN** the next day — with a confidence score and plain-English explanation of which features drove the prediction.

---

## Architecture

```
Finnhub API (live news)          Yahoo Finance (stock prices)
        │                                    │
        ▼                                    ▼
FinBERT Sentiment Scoring         Technical Indicators
(positive / negative / neutral)   (MA, volatility, momentum)
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
           Feature Engineering (14 features)
                       │
                       ▼
         Logistic Regression Classifier
         (time-based train/test split)
                       │
                       ▼
              FastAPI Backend
          ┌────────────┴────────────┐
          ▼                         ▼
   /predict/{ticker}          /history/{ticker}
   /sentiment/{ticker}        /compare
          │
          ▼
   Streamlit Dashboard
   ├── 🎯 Predict Tab    — UP/DOWN prediction + top factors
   ├── 📅 History Tab    — 30-day sentiment trend chart
   ├── ⚖️  Compare Tab   — sector sentiment comparison
   └── 🧠 Model Info Tab — performance metrics + SHAP findings
```

---

## Key Findings

- **"Buy the rumor, sell the news"** — Positive sentiment has a *negative* correlation (−0.16) with next-day returns. Markets price in good news before it's published; negative headlines often precede recovery.
- **Technical features > sentiment alone** — SHAP analysis shows `news_count`, `volatility_5d`, and `momentum_3d` are stronger predictors than raw sentiment scores.
- **Modest accuracy is the honest result** — ~60% accuracy on stock direction prediction. If a model hit 90%, it would indicate data leakage. Markets are efficient.
- **Sector patterns** — TSLA and GOOGL receive the most negative coverage; JNJ and WMT skew positive, reflecting tech volatility vs. defensive sector stability.

---

## Screenshots

### Prediction Dashboard
![Dashboard](docs/dashboard_screenshot.png)

### Sentiment Distribution by Ticker
![Sentiment by Ticker](docs/02_sentiment_by_ticker.png)

### SHAP Feature Importance
![SHAP](docs/06_shap_summary.png)

### Correlation Heatmap
![Heatmap](docs/05_correlation_heatmap.png)

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| NLP Model | FinBERT (ProsusAI/finbert) | Fine-tuned on financial text — understands "guidance raised" vs "margin compression" |
| ML Models | Logistic Regression, Random Forest, XGBoost | Three models compared; LR won (AUC 0.778) |
| Explainability | SHAP | Explains which features drove each prediction |
| Data Sources | Finnhub API, yfinance | Live news + OHLCV stock prices |
| Storage | SQLite + SQLAlchemy | Persistent pipeline — each stage re-runnable |
| API | FastAPI | Async, auto-docs at `/docs`, Pydantic validation |
| Dashboard | Streamlit + Plotly | 4-tab interactive UI with company name autocomplete |
| Containerization | Docker + docker-compose | One-command deployment on any machine |
| Testing | pytest | 9 tests covering validity, direction, and edge cases |
| Language | Python 3.11 | Pandas, NumPy, Scikit-learn, HuggingFace Transformers |

---

## Model Performance

| Model | AUC | F1 | Accuracy |
|---|---|---|---|
| **Logistic Regression ✅** | **0.778** | **0.667** | **62%** |
| Random Forest | 0.712 | 0.601 | 58% |
| XGBoost | 0.695 | 0.589 | 56% |

**Why time-based split:** Data is split chronologically (first 75% = train, last 25% = test) to prevent data leakage — a critical requirement in financial ML.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Health check |
| `GET /predict/{ticker}` | Full pipeline: news → sentiment → features → UP/DOWN prediction |
| `GET /sentiment/{ticker}` | FinBERT sentiment scores for recent headlines |
| `GET /stock/{ticker}` | Recent stock price data |
| `GET /history/{ticker}` | 30-day day-by-day sentiment trend |
| `GET /compare?tickers=AAPL,MSFT,TSLA` | Side-by-side sentiment comparison |

Auto-generated docs available at `http://localhost:8000/docs`

---

## Quick Start

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/karthikeyansett1/financial-sentiment-analyzer.git
cd financial-sentiment-analyzer
echo "FINNHUB_API_KEY=your_key_here" > .env
docker-compose up --build
```

- Dashboard: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

### Option 2 — Local

```bash
git clone https://github.com/karthikeyansett1/financial-sentiment-analyzer.git
cd financial-sentiment-analyzer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "FINNHUB_API_KEY=your_key_here" > .env
```

Get a free API key at [finnhub.io](https://finnhub.io)

```bash
# Terminal 1 — API
uvicorn api.main:app --reload

# Terminal 2 — Dashboard
streamlit run dashboard/app.py
```

---

## Project Structure

```
financial-sentiment-analyzer/
├── api/
│   └── main.py              # FastAPI — 5 endpoints including /predict
├── dashboard/
│   └── app.py               # Streamlit — 4-tab dashboard with autocomplete
├── src/
│   ├── data_pipeline.py     # Finnhub + yfinance data fetching
│   ├── sentiment.py         # FinBERT inference
│   ├── features.py          # Feature engineering (14 features)
│   ├── model.py             # Model training + evaluation
│   └── database.py          # SQLite helpers
├── tests/
│   └── test_pipeline.py     # 9 pytest tests
├── notebooks/
│   └── 01_EDA_and_Sentiment_Analysis.ipynb
├── docs/                    # Charts and screenshots
├── models/                  # Saved model + scaler (gitignored)
├── data/                    # SQLite database (gitignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Dataset

- **2,469 headlines** across 10 tickers (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, JPM, JNJ, XOM, WMT)
- **1,250 stock-days** of OHLCV data
- **64 merged rows** for modeling (limited by Finnhub free tier date range)
- With a paid data source, this pipeline scales to 10,000+ rows

---

## Limitations & Future Work

- **Small dataset** — Finnhub free tier limits historical range. Next step: integrate NewsAPI or Alpha Vantage for more history.
- **Single-day prediction** — model predicts next-day direction only. Could extend to multi-day horizons.
- **No real-time streaming** — currently pulls on demand. A production version would use Kafka for real-time ingestion.
- **Model drift** — no monitoring in place. Would add Evidently AI or Prometheus in production.
- **Fine-tuning FinBERT** — the model uses pre-trained weights. Fine-tuning on domain-specific data could improve sentiment accuracy.

---

## Author

**Karthikeyan Setti** — M.S. Data Science, Indiana University Bloomington

[LinkedIn](https://linkedin.com/in/karthikeyansetti) · [GitHub](https://github.com/karthikeyansett1)
