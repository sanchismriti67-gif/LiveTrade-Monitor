# Quantum Trading Engine

A microservices-based algorithmic trading platform with real-time market data ingestion, Redis pub/sub streaming, CouchDB persistence, LSTM-based ML price prediction, and a multi-node Flask web dashboard — deployed on Railway.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3.3-black?logo=flask)
![Redis](https://img.shields.io/badge/Redis-Pub%2FSub-red?logo=redis)
![CouchDB](https://img.shields.io/badge/CouchDB-3.3-orange?logo=apache)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![Railway](https://img.shields.io/badge/Deployed-Railway-purple?logo=railway)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    nginx (port 80)                  │
│              Load Balancer (round-robin)            │
└──────────┬──────────────┬──────────────┬────────────┘
           │              │              │
      web1:5001      web2:5002      web3:5003
           │              │              │
           └──────────────┼──────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
         Redis (pub/sub)         CouchDB (persist)
              │
        ingestion-worker
        (broadcasts every 4s)
```

The platform is split into two independently deployed services:

- **Web App** (`app.py`) — Flask + gunicorn, handles auth, portfolio management, and real-time dashboard
- **Ingestion Worker** (`worker.py`) — fetches market data, computes RSI signals, publishes to Redis every 4 seconds

---

## Features

- **Real-time market feed** via Redis pub/sub — 23 tickers updated every 4 seconds
- **Circuit breaker** — automatically falls back to synthetic price simulation if upstream APIs are rate-limited
- **LSTM ML predictions** — TensorFlow model predicts next-day price per ticker (lazy-loaded to avoid startup crashes)
- **Portfolio management** — buy/sell with live prices, persistent in CouchDB with in-memory fallback
- **Technical indicators** — RSI(14) computed per ticker with BUY/SELL/HOLD signals
- **News feed** — Google News RSS per ticker
- **Multi-node web** — 3 Flask instances behind nginx locally, single gunicorn on Railway
- **Full Docker Compose stack** — one command to run everything locally

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask 2.3.3 + gunicorn |
| Auth | Flask-Login + Werkzeug password hashing |
| Real-time | Redis pub/sub |
| Database | CouchDB 3.3 |
| ML model | TensorFlow (CPU) LSTM |
| Market data | yfinance + synthetic fallback |
| Reverse proxy | nginx |
| Containerisation | Docker + Docker Compose |
| Deployment | Railway |

---

## Tickers Tracked

| Exchange | Tickers |
|---|---|
| NASDAQ (US) | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, NFLX |
| NSE (India) | RELIANCE, TCS, HDFCBANK, INFY, ITC, SBIN, BHARTIARTL, WIPRO, ZOMATO, TATAMOTORS |
| Crypto | BTC-USD, ETH-USD, BNB-USD, SOL-USD, DOGE-USD |

---

## Local Setup

### Prerequisites

- Docker Desktop
- Docker Compose v2

### Run the full stack

```bash
git clone https://github.com/your-username/algo_trading_project.git
cd algo_trading_project
docker-compose up --build
```

This starts all services:

| Service | URL |
|---|---|
| Dashboard (via nginx) | http://localhost |
| Web node 1 | http://localhost:5001 |
| Web node 2 | http://localhost:5002 |
| Web node 3 | http://localhost:5003 |
| CouchDB admin | http://localhost:5984/_utils |
| Redis | localhost:6379 |

### Initialise CouchDB system databases (first run only)

```bash
# Linux / macOS
curl -X PUT http://admin:password@localhost:5984/_users
curl -X PUT http://admin:password@localhost:5984/_replicator
curl -X PUT http://admin:password@localhost:5984/_global_changes

# Windows PowerShell
Invoke-WebRequest -Method PUT -Uri "http://admin:password@localhost:5984/_users"
Invoke-WebRequest -Method PUT -Uri "http://admin:password@localhost:5984/_replicator"
Invoke-WebRequest -Method PUT -Uri "http://admin:password@localhost:5984/_global_changes"
```

### Run without Docker (dev mode)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
python app.py
```

The app runs without Redis and CouchDB — it falls back to in-memory storage automatically.

---

## Project Structure

```
algo_trading_project/
├── app.py                  # Flask web app (auth, portfolio, dashboard)
├── worker.py               # Market data ingestion + Redis publisher
├── ml_model.py             # LSTM price prediction (lazy TF import)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Single image for both web + worker
├── docker-compose.yml      # Full local stack
├── nginx.conf              # Load balancer config
├── Dockerfile.couchdb      # CouchDB with pre-init
└── README.md
```

---

## Deployment (Railway)

The project is deployed as two separate Railway services from the same GitHub repo, using different start commands.

### Environment Variables

Set these in Railway for the **web app** service:

| Variable | Value |
|---|---|
| `REDIS_URL` | Auto-injected when Redis service is linked |
| `COUCHDB_URL` | `http://admin:password@couchdb.railway.internal:5984/` |
| `SECRET_KEY` | Any long random string |
| `PORT` | `8080` |

### Start Commands

**Web app (`remarkable-charm`):**
```
gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 120 --workers 1 --log-level info app:app
```

**Ingestion worker (`Ingestion-Worker`):**
```
python worker.py
```

---

## API Endpoints

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| GET | `/` | Landing page | No |
| GET | `/dashboard` | Trading dashboard | Yes |
| GET | `/health` | Health check (Redis + CouchDB status) | No |
| GET | `/api/data` | All stock tickers with signals | No |
| GET | `/api/history/<ticker>` | 7-day price history | No |
| GET | `/api/news/<ticker>` | Latest news for ticker | No |
| GET | `/api/predict/<ticker>` | LSTM price prediction | No |
| GET | `/api/portfolio` | User's holdings | Yes |
| POST | `/api/buy` | Buy shares | Yes |
| POST | `/api/sell` | Sell shares | Yes |

---

## ML Model

The LSTM model in `ml_model.py` trains on 2 years of historical close prices per request:

- Sequence length: 60 days
- Architecture: LSTM(50) → Dense(25) → Dense(1)
- Trains for 1 epoch per prediction (fast inference)
- Falls back to synthetic data if yfinance is rate-limited
- TensorFlow is imported lazily inside the prediction function — a TF crash only breaks `/api/predict`, not the rest of the app

---

## Known Behaviours

| Behaviour | Reason |
|---|---|
| Circuit breaker activates on Railway | yfinance is rate-limited in cloud environments — synthetic data kicks in automatically |
| CouchDB 409 conflicts in worker logs | Worker does blind PUT every 4s; harmless, data is already current |
| No real-time prices on first load | Redis cache populates after the first worker broadcast (~4 seconds) |

---

## License

MIT
