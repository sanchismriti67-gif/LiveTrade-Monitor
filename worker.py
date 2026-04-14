import time, json, os, requests, logging, random
import pandas as pd
import redis
from datetime import datetime
import yfinance as yf

logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

COUCHDB_URL = os.getenv("COUCHDB_URL", "http://admin:password@couchdb.railway.internal:5984/")
DB_NAME = "stocks"
REDIS_URL = os.getenv("REDIS_URL", "")

# ==========================================
# REDIS CONNECTION — with retry on startup.
# The original code had no timeout or retry,
# so if Redis wasn't ready yet the worker
# crashed immediately on deploy.
# ==========================================
def connect_redis(max_attempts=10):
    """Retry Redis connection with backoff. Returns client or raises after max_attempts."""
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL environment variable is not set.")

    for attempt in range(1, max_attempts + 1):
        try:
            if REDIS_URL.startswith("redis"):
                client = redis.Redis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
            else:
                client = redis.Redis(
                    host=REDIS_URL,
                    port=6379,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
            client.ping()
            logging.info(f"Redis connected on attempt {attempt}.")
            return client
        except Exception as e:
            wait = min(2 ** attempt, 30)
            logging.warning(f"Redis connection attempt {attempt}/{max_attempts} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Could not connect to Redis after {max_attempts} attempts.")


redis_client = connect_redis()

TICKERS = {
    'AAPL': 'NASDAQ (US)', 'MSFT': 'NASDAQ (US)', 'GOOGL': 'NASDAQ (US)', 'AMZN': 'NASDAQ (US)',
    'NVDA': 'NASDAQ (US)', 'META': 'NASDAQ (US)', 'TSLA': 'NASDAQ (US)', 'NFLX': 'NASDAQ (US)',
    'RELIANCE.NS': 'NSE (India)', 'TCS.NS': 'NSE (India)', 'HDFCBANK.NS': 'NSE (India)',
    'INFY.NS': 'NSE (India)', 'ITC.NS': 'NSE (India)', 'SBIN.NS': 'NSE (India)',
    'BHARTIARTL.NS': 'NSE (India)', 'WIPRO.NS': 'NSE (India)', 'ZOMATO.NS': 'NSE (India)',
    'TATAMOTORS.NS': 'NSE (India)',
    'BTC-USD': 'Crypto', 'ETH-USD': 'Crypto', 'BNB-USD': 'Crypto', 'SOL-USD': 'Crypto', 'DOGE-USD': 'Crypto'
}

API_BLOCKED = False
SYNTHETIC_PRICES = {}


def init_db():
    try:
        requests.put(f"{COUCHDB_URL}{DB_NAME}", timeout=5)
    except Exception:
        pass


def init_synthetic_base():
    for t, ex in TICKERS.items():
        base = 150.0 if 'US' in ex else 2000.0 if 'India' in ex else 40000.0
        SYNTHETIC_PRICES[t] = base + random.uniform(-10, 10)


init_synthetic_base()


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def fetch_data(ticker):
    global API_BLOCKED
    if not API_BLOCKED:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if hist.empty or len(hist) < 3:
                raise ValueError("Insufficient data — likely rate limited")
            return hist
        except Exception as e:
            API_BLOCKED = True
            logging.warning(f"[CIRCUIT BREAKER] Upstream blocked ({e}). Switching to synthetic simulation.")

    # Synthetic tick
    curr = SYNTHETIC_PRICES[ticker]
    curr = curr * (1 + random.uniform(-0.003, 0.003))
    SYNTHETIC_PRICES[ticker] = curr
    prices = [curr * (1 + random.uniform(-0.01, 0.01)) for _ in range(15)]
    prices.append(curr)
    return pd.DataFrame({
        'Close': prices,
        'Volume': [int(random.uniform(1_000_000, 5_000_000)) for _ in prices]
    })


def process_tickers():
    payload = []
    for ticker, exchange in TICKERS.items():
        try:
            hist = fetch_data(ticker)
            closes = hist['Close']
            current_price = closes.iloc[-1]
            prev_price = closes.iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            rsi_series = calc_rsi(closes)
            current_rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0
            signal = "BUY" if current_rsi < 30 else "SELL" if current_rsi > 70 else "HOLD"
            volume = int(hist['Volume'].iloc[-1])
            doc = {
                "ticker": ticker, "exchange": exchange,
                "price": round(current_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_pct, 2),
                "rsi": round(current_rsi, 2),
                "signal": signal, "volume": volume,
                "last_updated": datetime.now().isoformat()
            }
            payload.append(doc)
            # Fire-and-forget CouchDB persistence
            couch_doc = dict(doc)
            couch_doc['_id'] = ticker
            try:
                requests.put(f"{COUCHDB_URL}{DB_NAME}/{ticker}", json=couch_doc, timeout=1)
            except Exception:
                pass
        except Exception as e:
            logging.warning(f"Skipping {ticker}: {e}")
    return payload


def publish_with_retry(data, max_retries=3):
    """Publish to Redis with reconnect on failure."""
    global redis_client
    for attempt in range(max_retries):
        try:
            redis_client.publish('live_prices', json.dumps(data))
            return True
        except redis.ConnectionError as e:
            logging.warning(f"Redis publish failed (attempt {attempt+1}): {e}. Reconnecting...")
            try:
                redis_client = connect_redis(max_attempts=5)
            except Exception:
                pass
        except Exception as e:
            logging.error(f"Unexpected publish error: {e}")
            break
    return False


if __name__ == "__main__":
    logging.info("Starting High-Frequency Engine...")
    init_db()
    while True:
        try:
            data = process_tickers()
            success = publish_with_retry(data)
            if success:
                logging.info(f"[PULSE] Broadcasted {len(data)} market ticks via Redis.")
            else:
                logging.error("[PULSE] Failed to broadcast — Redis publish failed.")
        except Exception as e:
            logging.error(f"[FATAL] Iteration failed: {e}")
        time.sleep(4)