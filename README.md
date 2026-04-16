# Financial News Sentiment Analyzer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow?style=flat-square)](https://huggingface.co/spaces/karthikeyansett1/financial-sentiment-analyzer)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://python.org)
[![FinBERT](https://img.shields.io/badge/NLP-FinBERT-orange?style=flat-square)](https://huggingface.co/ProsusAI/finbert)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green?style=flat-square)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Deploy-Docker-blue?style=flat-square)](https://docker.com)

**[Try it live →](https://huggingface.co/spaces/karthikeyansett1/financial-sentiment-analyzer)**

---

## The Idea

I invest in stocks through Robinhood. Like most retail investors, I'd spend time every morning scrolling through financial news trying to figure out the general direction a stock might move that day — positive earnings coverage, geopolitical risk, sector downturns. It's slow, it's subjective, and it's easy to miss things.

I wanted to see if I could automate that intuition.

Most of my prior work had been in healthcare ML. Finance was new territory. The stock market felt like the most honest test of a prediction model — if your signal is real, the market will tell you. If it isn't, no amount of overfitting hides that.

This project pulls live financial news, scores every headline using a transformer model trained on financial text, combines those scores with technical stock indicators, and makes a next-day direction prediction — UP or DOWN — with a plain-English explanation of what drove it.

---

## What I Built

An end-to-end ML pipeline, fully deployed:

```
Live News (Finnhub API)          Stock Prices (Yahoo Finance)
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
          ┌────────────┴──────────────┐
          ▼                           ▼
   /predict/{ticker}           /history/{ticker}
   /sentiment/{ticker}         /compare
                       │
                       ▼
            Streamlit Dashboard
     ┌─────────┬──────────┬──────────┐
  Predict   History   Compare   Model Info
```

---

## Screenshots

### Prediction Dashboard
![Dashboard](docs/dashboard_screenshot.png)

### Sentiment by Ticker
![Sentiment](docs/02_sentiment_by_ticker.png)

### SHAP Feature Importance
![SHAP](docs/06_shap_summary.png)

### Correlation Heatmap
![Heatmap](docs/05_correlation_heatmap.png)

---

## What I Found

**The result that surprised me most:** positive news sentiment has a *negative* correlation (−0.16) with next-day returns.

At first that seems wrong. But it makes sense once you think about it — by the time positive news is published, the market has already priced it in. Traders who bought on the rumor are now selling on the news. This "buy the rumor, sell the news" effect is well-documented in finance, and my model learned it from the data without me telling it to.

When I compared the model's UP/DOWN calls against what actually happened on Robinhood, it wasn't hitting exact price targets — that would be unrealistic. But the directional accuracy was meaningful enough that I started thinking about how much time this could save versus manually reading through headlines every morning.

**Other findings:**
- Technical features (news volume, 5-day volatility, 3-day momentum) outperformed raw sentiment scores as predictors — context matters more than tone alone
- TSLA and GOOGL receive the most negative news coverage; JNJ and WMT skew positive — reflecting tech volatility vs. defensive sector stability
- ~60% directional accuracy is honest and expected. A 90% accurate stock model would almost certainly be leaking future data

---

## Model Performance

| Model | AUC | F1 | Accuracy |
|---|---|---|---|
| **Logistic Regression ✅** | **0.778** | **0.667** | **62%** |
| Random Forest | 0.712 | 0.601 | 58% |
| XGBoost | 0.695 | 0.589 | 56% |

Three models were trained and compared. Logistic Regression won — meaning the patterns in this dataset are mostly linear, and simpler models generalize better than complex ones on limited data. The time-based train/test split (first 75% of dates = train, last 25% = test) prevents data leakage — a critical requirement in any financial ML system.

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| NLP Model | FinBERT (ProsusAI) | Fine-tuned on 50,000+ financial articles — understands "guidance raised" vs "margin compression" better than general-purpose models |
| ML | Logistic Regression, Random Forest, XGBoost | Three models compared; best selected by F1 score |
| Explainability | SHAP | Shows which features drove each prediction in plain English |
| Data | Finnhub API, yfinance | Live news headlines + OHLCV stock prices |
| Storage | SQLite + SQLAlchemy | Persistent pipeline — each stage independently re-runnable |
| Backend | FastAPI | Async, auto-generates Swagger docs at `/docs` |
| Frontend | Streamlit + Plotly | 4-tab dashboard with company name autocomplete search |
| Containerization | Docker + docker-compose | One-command deployment on any machine |
| Tests | pytest | 9 tests covering validity, directional accuracy, and edge cases |

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /predict/{ticker}` | Full pipeline: live news → sentiment → features → UP/DOWN prediction with confidence and top factors |
| `GET /sentiment/{ticker}` | FinBERT sentiment scores for recent headlines |
| `GET /history/{ticker}` | 30-day day-by-day sentiment trend |
| `GET /compare?tickers=AAPL,MSFT,TSLA` | Side-by-side sentiment comparison across tickers |
| `GET /stock/{ticker}` | Recent stock price data |

Swagger docs: `http://localhost:8000/docs`

---

## Quick Start

### Option 1 — Live Demo
**[huggingface.co/spaces/karthikeyansett1/financial-sentiment-analyzer](https://huggingface.co/spaces/karthikeyansett1/financial-sentiment-analyzer)**

No setup required.

### Option 2 — Docker (local)

```bash
git clone https://github.com/karthikeyansett1/financial-sentiment-analyzer.git
cd financial-sentiment-analyzer
echo "FINNHUB_API_KEY=your_key_here" > .env
docker-compose up --build
```

Get a free API key at [finnhub.io](https://finnhub.io)

- Dashboard: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

### Option 3 — Local without Docker

```bash
git clone https://github.com/karthikeyansett1/financial-sentiment-analyzer.git
cd financial-sentiment-analyzer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "FINNHUB_API_KEY=your_key_here" > .env

# Terminal 1
uvicorn api.main:app --reload

# Terminal 2
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
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Limitations & What I'd Do Next

The biggest constraint was data. Finnhub's free tier only gives 30 days of news history, which left me with 64 rows for modeling after merging news and stock data by date. The pipeline itself is solid — with a paid data source like Bloomberg or even NewsAPI, this scales to 10,000+ rows and the model accuracy would improve meaningfully.

If I were to extend this further I would also look at fine-tuning FinBERT on a financial dataset specific to my tickers rather than using the pre-trained weights, and add model drift monitoring so the system flags when market conditions shift enough that the model's predictions become unreliable.

---

## Dataset

- 2,469 headlines across 10 tickers (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, JPM, JNJ, XOM, WMT)
- 1,250 stock-days of OHLCV price data
- 64 merged rows used for modeling (limited by Finnhub free tier)

---

**Karthikeyan Setti** — M.S. Data Science, Indiana University Bloomington

[LinkedIn](https://www.linkedin.com/in/karthikeyan-setti-ks/) · [GitHub](https://github.com/karthikeyansett1) · [Live Demo](https://huggingface.co/spaces/karthikeyansett1/financial-sentiment-analyzer)
