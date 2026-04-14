import sys
print("APP STARTUP: Python imports beginning...", flush=True, file=sys.stderr)

from flask import Flask, jsonify, request, Response, redirect, url_for, render_template_string
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import socket
import requests
import time
import threading
import random
import os
import logging
from datetime import datetime, timedelta
import csv
import io
import feedparser
import urllib.parse
import pytz
import json
import redis
import yfinance as yf

# NOTE: ml_model is NOT imported here at the top level.
# It is imported lazily inside the /api/predict route to prevent
# TensorFlow/CUDA crashes from killing the entire app on startup.

print("APP STARTUP: All standard imports done.", flush=True, file=sys.stderr)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "PSG_TECH_SECRET_KEY_CHANGE_IN_PROD")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("APP STARTUP: Flask app created.", flush=True, file=sys.stderr)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'  # type: ignore

COUCHDB_URL = os.getenv("COUCHDB_URL", "http://admin:password@couchdb:5984/")
DB_NAME = "stocks"
ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')
REDIS_URL = os.getenv("REDIS_URL", "")

redis_client = None
local_cache = []
cache_lock = threading.Lock()


def _init_redis():
    """Try to connect to Redis. Returns client or None. Never raises."""
    global redis_client
    if not REDIS_URL:
        logger.warning("REDIS_URL not set — running without live market feed.")
        return None
    try:
        client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=3,
            # socket_timeout intentionally omitted —
            # pub/sub connections must block indefinitely while waiting
            # for messages. Setting socket_timeout causes spurious
            # "Timeout reading from socket" warnings every few seconds.
            retry_on_timeout=False,
            health_check_interval=30
        )
        client.ping()
        logger.info("Redis connected successfully.")
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable at startup (will run without live updates): {e}")
        return None


print("APP STARTUP: Attempting Redis connection...", flush=True, file=sys.stderr)
redis_client = _init_redis()
print(f"APP STARTUP: Redis {'connected' if redis_client else 'unavailable — continuing without it'}.", flush=True, file=sys.stderr)


def redis_subscriber():
    """Background thread: subscribe to Redis and populate local_cache."""
    global local_cache, redis_client
    if redis_client is None:
        logger.warning("Redis subscriber not started — no Redis connection.")
        return

    retry_count = 0
    while True:
        try:
            pubsub = redis_client.pubsub()  #type:ignore
            pubsub.subscribe('live_prices')
            logger.info("Subscribed to Redis channel 'live_prices'")
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        with cache_lock:
                            local_cache = data
                        retry_count = 0
                    except Exception as e:
                        logger.warning(f"Redis message parse error: {e}")
        except redis.ConnectionError as e:
            retry_count += 1
            wait = min(2 ** retry_count, 60)
            logger.warning(f"Redis connection lost (attempt {retry_count}), retrying in {wait}s...")
            time.sleep(wait)
            redis_client = _init_redis()
            if redis_client is None:
                time.sleep(wait)
        except Exception as e:
            logger.warning(f"Unexpected Redis error in subscriber: {e}")
            time.sleep(5)


if redis_client is not None:
    t = threading.Thread(target=redis_subscriber, daemon=True)
    t.start()
else:
    logger.warning("Running without Redis — live price updates disabled.")

print("APP STARTUP: Redis subscriber configured.", flush=True, file=sys.stderr)

start_time = time.time()
request_count = 0
request_lock = threading.Lock()

state = {'couchdb_available': False}
news_cache = {}
news_cache_time = {}
CACHE_DURATION = 1800
memory_users = {}
memory_users_lock = threading.Lock()


def check_couchdb_connection():
    try:
        response = requests.get(COUCHDB_URL + "_up", timeout=3)
        if response.status_code == 200:
            state['couchdb_available'] = True
            return True
    except Exception:
        pass
    state['couchdb_available'] = False
    return False


print("APP STARTUP: Checking CouchDB...", flush=True, file=sys.stderr)
try:
    check_couchdb_connection()
    print(f"APP STARTUP: CouchDB available: {state['couchdb_available']}", flush=True, file=sys.stderr)
except Exception as e:
    print(f"APP STARTUP: CouchDB check failed (non-fatal): {e}", flush=True, file=sys.stderr)


class User(UserMixin):
    def __init__(self, user_id, username, portfolio=None, password_hash=None):
        self.id = user_id
        self.username = username
        self.portfolio = portfolio or {}
        self.password_hash = password_hash

    @staticmethod
    def get(user_id):
        with memory_users_lock:
            if user_id in memory_users:
                return memory_users[user_id]
        if state['couchdb_available']:
            try:
                resp = requests.get(COUCHDB_URL + "users/" + user_id, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    user = User(data['_id'], data['username'], data.get('portfolio', {}), data.get('password_hash'))
                    with memory_users_lock:
                        memory_users[user_id] = user
                    return user
            except Exception:
                pass
        return None

    def save(self):
        with memory_users_lock:
            memory_users[self.id] = self
        if state['couchdb_available']:
            try:
                doc = {
                    "_id": self.id,
                    "username": self.username,
                    "portfolio": self.portfolio,
                    "password_hash": self.password_hash
                }
                resp = requests.get(COUCHDB_URL + "users/" + self.id, timeout=5)
                if resp.status_code == 200:
                    doc['_rev'] = resp.json()['_rev']
                requests.put(COUCHDB_URL + "users/" + self.id, json=doc, timeout=5)
            except Exception:
                pass


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() >= 5:
        return "Closed (Weekend)"
    hour = now.hour
    minute = now.minute
    current_minutes = hour * 60 + minute
    market_open = 9 * 60 + 15
    market_close = 15 * 60 + 30
    if market_open <= current_minutes <= market_close:
        return "Open"
    return "Closed"


@app.before_request
def before_request():
    global request_count
    with request_lock:
        request_count += 1


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "redis": redis_client is not None,
        "couchdb": state['couchdb_available'],
        "uptime_seconds": round(time.time() - start_time, 1)
    }), 200


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            return render_template_string(LOGIN_PAGE, error="Username and password required")
        user = None
        if state['couchdb_available']:
            try:
                resp = requests.get(COUCHDB_URL + "users/_all_docs?include_docs=true", timeout=5)
                rows = resp.json().get('rows', [])
                for row in rows:
                    doc = row.get('doc', {})
                    if doc.get('username') == username:
                        if check_password_hash(doc.get('password_hash', ''), password):
                            user = User(doc['_id'], username, doc.get('portfolio', {}), doc.get('password_hash'))
                        break
            except Exception:
                pass
        if not user:
            with memory_users_lock:
                for uid, u in memory_users.items():
                    if u.username == username and check_password_hash(u.password_hash or '', password):
                        user = u
                        break
        if user:
            login_user(user)
            return redirect('/dashboard')
        return render_template_string(LOGIN_PAGE, error="Invalid credentials")
    return render_template_string(LOGIN_PAGE)


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            return render_template_string(REGISTER_PAGE, error="Username and password required")
        exists = False
        if state['couchdb_available']:
            try:
                resp = requests.get(COUCHDB_URL + "users/_all_docs?include_docs=true", timeout=5)
                rows = resp.json().get('rows', [])
                for row in rows:
                    if row.get('doc', {}).get('username') == username:
                        exists = True
                        break
            except Exception:
                pass
        if not exists:
            with memory_users_lock:
                for u in memory_users.values():
                    if u.username == username:
                        exists = True
                        break
        if exists:
            return render_template_string(REGISTER_PAGE, error="Username already taken")
        import uuid
        user_id = str(uuid.uuid4())
        new_user = User(user_id, username, {}, generate_password_hash(password))
        new_user.save()
        login_user(new_user)
        return redirect('/dashboard')
    return render_template_string(REGISTER_PAGE)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')


@app.route('/api/portfolio')
@login_required
def get_portfolio():
    holdings = []
    total_value = 0.0
    with cache_lock:
        current_data = {item.get('ticker'): item for item in local_cache}
    for symbol, qty in current_user.portfolio.items():
        price = current_data.get(symbol, {}).get('price', 0.0)
        if price <= 0 and state['couchdb_available']:
            try:
                resp = requests.get(COUCHDB_URL + DB_NAME + "/" + symbol, timeout=3)
                if resp.status_code == 200:
                    price = resp.json().get('price', 0.0)
            except Exception:
                pass
        value = price * qty
        total_value += value
        holdings.append({'symbol': symbol, 'quantity': qty, 'price': price, 'value': value})
    return jsonify({'holdings': holdings, 'total_value': total_value})


@app.route('/api/buy', methods=['POST'])
@login_required
def buy_stock():
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid request body'}), 400
    symbol = data.get('symbol', '').upper().strip()
    try:
        quantity = int(data.get('quantity', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Quantity must be an integer'}), 400
    if quantity <= 0:
        return jsonify({'error': 'Invalid quantity'}), 400
    with cache_lock:
        current_data = {item.get('ticker'): item for item in local_cache}
    price = current_data.get(symbol, {}).get('price', 0.0)
    if price <= 0 and state['couchdb_available']:
        try:
            resp = requests.get(COUCHDB_URL + DB_NAME + "/" + symbol, timeout=3)
            if resp.status_code == 200:
                price = resp.json().get('price', 0.0)
        except Exception:
            pass
    if price <= 0:
        return jsonify({'error': 'Stock data temporarily unavailable. Worker syncing.'}), 404
    current_user.portfolio[symbol] = current_user.portfolio.get(symbol, 0) + quantity
    current_user.save()
    return jsonify({'success': True, 'symbol': symbol, 'quantity': quantity, 'price': price})


@app.route('/api/sell', methods=['POST'])
@login_required
def sell_stock():
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid request body'}), 400
    symbol = data.get('symbol', '').upper().strip()
    try:
        quantity = int(data.get('quantity', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Quantity must be an integer'}), 400
    if quantity <= 0:
        return jsonify({'error': 'Invalid quantity'}), 400
    if current_user.portfolio.get(symbol, 0) < quantity:
        return jsonify({'error': 'Insufficient shares in portfolio'}), 400
    with cache_lock:
        current_data = {item.get('ticker'): item for item in local_cache}
    price = current_data.get(symbol, {}).get('price', 0.0)
    if price <= 0 and state['couchdb_available']:
        try:
            resp = requests.get(COUCHDB_URL + DB_NAME + "/" + symbol, timeout=3)
            if resp.status_code == 200:
                price = resp.json().get('price', 0.0)
        except Exception:
            pass
    if price <= 0:
        return jsonify({'error': 'Stock data temporarily unavailable. Worker syncing.'}), 404
    current_user.portfolio[symbol] -= quantity
    if current_user.portfolio[symbol] == 0:
        del current_user.portfolio[symbol]
    current_user.save()
    return jsonify({'success': True, 'symbol': symbol, 'quantity': quantity, 'price': price})


@app.route('/api/data')
def get_data():
    with cache_lock:
        current_data = list(local_cache)
    if not current_data and state['couchdb_available']:
        try:
            resp = requests.get(COUCHDB_URL + DB_NAME + "/_all_docs?include_docs=true", timeout=3)
            current_data = [row['doc'] for row in resp.json().get('rows', []) if not row['id'].startswith('_design')]
        except Exception:
            pass
    try:
        mr_response = requests.get(COUCHDB_URL + DB_NAME + "/_design/analytics/_view/portfolio_value", timeout=2)
        mr_rows = mr_response.json().get('rows', [])
        total_value = mr_rows[0]['value'] if mr_rows else 0.0
    except Exception:
        total_value = sum(item.get('price', 0) for item in current_data)
    signals = {}
    exchanges = {}
    for item in current_data:
        sig = item.get('signal', 'HOLD')
        exc = item.get('exchange', 'Unknown')
        signals[sig] = signals.get(sig, 0) + 1
        exchanges[exc] = exchanges.get(exc, 0) + 1
    return jsonify({
        "server_id": socket.gethostname(),
        "stock_data": current_data,
        "mapreduce_total": round(total_value, 2),
        "total_stocks": len(current_data),
        "signal_distribution": [{"key": k, "value": v} for k, v in signals.items()],
        "exchange_distribution": [{"key": k, "value": v} for k, v in exchanges.items()],
        "market_status": get_market_status()
    })


@app.route('/api/history/<ticker>')
def get_history(ticker):
    try:
        with cache_lock:
            current_data = {item.get('ticker'): item for item in local_cache}
        symbol = current_data.get(ticker, {}).get('symbol', ticker)
        stock = yf.Ticker(symbol)
        hist = stock.history(period="7d", interval="1h")
        if hist.empty:
            raise Exception("Empty history")
        dates = hist.index.strftime('%Y-%m-%d %H:%M').tolist()  # type: ignore
        prices = hist['Close'].round(2).tolist()
        return jsonify({"ticker": ticker, "dates": dates, "prices": prices})
    except Exception:
        dates = [(datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M') for i in range(50, 0, -1)]
        prices = [round(1000 + random.uniform(-50, 50), 2) for _ in range(50)]
        return jsonify({"ticker": ticker, "dates": dates, "prices": prices})


@app.route('/api/news/<ticker>')
def get_news(ticker):
    try:
        if ticker in news_cache and (datetime.now() - news_cache_time.get(ticker, datetime.min)).seconds < CACHE_DURATION:
            return jsonify({"ticker": ticker, "news": news_cache[ticker]})
        clean_ticker = ticker.replace('.NS', '').replace('-USD', '')
        news_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(clean_ticker)}+stock+OR+{urllib.parse.quote(clean_ticker)}+share&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(news_url)
        news_items = []
        for entry in feed.entries[:8]:
            news_items.append({
                'title': entry.title,
                'link': entry.link,
                'published': entry.published,
                'source': entry.source[0].get('title', 'Google News') if hasattr(entry, 'source') and entry.source else 'Google News'
            })
        if not news_items:
            raise Exception("No news found")
        news_cache[ticker] = news_items
        news_cache_time[ticker] = datetime.now()
        return jsonify({"ticker": ticker, "news": news_items})
    except Exception:
        return jsonify({"ticker": ticker, "news": [{'title': f'{ticker} market analysis updated.', 'link': '#', 'published': datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT'), 'source': 'Quantum System'}]})


@app.route('/api/predict/<ticker>')
def predict_stock(ticker):
    """
    ML prediction endpoint. The ml_model import happens HERE, not at startup.
    If TensorFlow/CUDA/model loading fails, only this endpoint returns an error —
    the rest of the app keeps running normally.
    """
    ticker = ticker.upper().strip()
    try:
        from ml_model import predict_next_day
        result = predict_next_day(ticker)
        if result is None:
            return jsonify({'error': 'Could not generate AI prediction for this ticker.'}), 400
        return jsonify(result)
    except ImportError as e:
        logger.error(f"ml_model import failed: {e}")
        return jsonify({'error': 'ML engine unavailable on this deployment.'}), 503
    except Exception as e:
        logger.error(f"Prediction error for {ticker}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect('/dashboard')
    return render_template_string(LANDING_HTML)


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template_string(DASHBOARD_HTML, username=current_user.username)


# ==========================================
# HTML TEMPLATES
# ==========================================

LOGIN_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Login - Quantum Trading</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { background: #0a0f1c; color: #e2e8f0; font-family: 'Inter', sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin:0; }
        .login-box { background: rgba(18,25,45,0.9); padding: 40px; border-radius: 16px; border: 1px solid rgba(0, 242, 254, 0.3); box-shadow: 0 0 40px rgba(0, 242, 254, 0.1); width: 350px; backdrop-filter: blur(10px); }
        h1 { color: #00f2fe; text-align: center; margin-bottom: 30px; font-weight: 700; letter-spacing: -1px; }
        input { width: 100%; padding: 14px; margin: 10px 0; background: #0a0f1c; border: 1px solid #2a3441; color: white; border-radius: 8px; box-sizing: border-box; font-family: 'Inter'; transition: 0.3s; }
        input:focus { border-color: #00f2fe; outline: none; box-shadow: 0 0 10px rgba(0, 242, 254, 0.2); }
        button { width: 100%; padding: 14px; background: linear-gradient(135deg, #00f2fe, #4facfe); border: none; color: #0a0f1c; font-weight: 600; font-size: 16px; border-radius: 8px; cursor: pointer; transition: 0.3s; margin-top: 10px; }
        button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(0, 242, 254, 0.3); }
        .error { background: rgba(255, 51, 102, 0.1); color: #ff3366; text-align: center; padding: 10px; border-radius: 8px; border: 1px solid rgba(255, 51, 102, 0.3); margin-bottom: 15px; font-size: 14px; }
        a { color: #00ff87; text-decoration: none; font-weight: 500; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="login-box">
        <h1>Quantum Node</h1>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
        <form method="post">
            <input type="text" name="username" placeholder="Operator ID" required autocomplete="off">
            <input type="password" name="password" placeholder="Access Key" required>
            <button type="submit">Establish Uplink</button>
        </form>
        <p style="text-align:center; margin-top:20px; font-size: 14px; color: #94a3b8;">Unregistered? <a href="/register">Provision Node</a></p>
    </div>
</body>
</html>
'''

REGISTER_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Register - Quantum Trading</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { background: #0a0f1c; color: #e2e8f0; font-family: 'Inter', sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin:0; }
        .login-box { background: rgba(18,25,45,0.9); padding: 40px; border-radius: 16px; border: 1px solid rgba(0, 255, 135, 0.3); box-shadow: 0 0 40px rgba(0, 255, 135, 0.1); width: 350px; backdrop-filter: blur(10px); }
        h1 { color: #00ff87; text-align: center; margin-bottom: 30px; font-weight: 700; letter-spacing: -1px; }
        input { width: 100%; padding: 14px; margin: 10px 0; background: #0a0f1c; border: 1px solid #2a3441; color: white; border-radius: 8px; box-sizing: border-box; font-family: 'Inter'; transition: 0.3s; }
        input:focus { border-color: #00ff87; outline: none; box-shadow: 0 0 10px rgba(0, 255, 135, 0.2); }
        button { width: 100%; padding: 14px; background: linear-gradient(135deg, #00ff87, #60efff); border: none; color: #0a0f1c; font-weight: 600; font-size: 16px; border-radius: 8px; cursor: pointer; transition: 0.3s; margin-top: 10px; }
        button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(0, 255, 135, 0.3); }
        .error { background: rgba(255, 51, 102, 0.1); color: #ff3366; text-align: center; padding: 10px; border-radius: 8px; border: 1px solid rgba(255, 51, 102, 0.3); margin-bottom: 15px; font-size: 14px; }
        a { color: #00f2fe; text-decoration: none; font-weight: 500; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="login-box">
        <h1>Provision Node</h1>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
        <form method="post">
            <input type="text" name="username" placeholder="Desired Operator ID" required autocomplete="off">
            <input type="password" name="password" placeholder="Access Key" required>
            <button type="submit">Initialize Identity</button>
        </form>
        <p style="text-align:center; margin-top:20px; font-size: 14px; color: #94a3b8;">Existing Operator? <a href="/login">Login Here</a></p>
    </div>
</body>
</html>
'''

LANDING_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Trading Terminal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap" rel="stylesheet">
    <style>
        body { background: #0a0f1c; color: #e2e8f0; font-family: 'Inter', sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin:0; text-align:center; overflow: hidden; }
        .bg-glow { position: absolute; width: 600px; height: 600px; background: radial-gradient(circle, rgba(0,242,254,0.1) 0%, rgba(10,15,28,0) 70%); top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: -1; }
        .landing { max-width: 800px; z-index: 1; }
        h1 { color: #ffffff; font-size: 4rem; font-weight: 800; letter-spacing: -2px; margin-bottom: 20px; line-height: 1.1; }
        .gradient-text { background: linear-gradient(135deg, #00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        p { font-size: 1.25rem; color: #94a3b8; margin-bottom: 40px; font-weight: 400; }
        .btn-container { display: flex; gap: 20px; justify-content: center; }
        .btn { padding: 16px 40px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 1.1rem; transition: 0.3s; display: inline-block; }
        .btn-primary { background: white; color: #0a0f1c; box-shadow: 0 10px 30px rgba(255,255,255,0.1); }
        .btn-primary:hover { background: #e2e8f0; transform: translateY(-3px); }
        .btn-secondary { background: transparent; color: white; border: 1px solid rgba(255,255,255,0.2); }
        .btn-secondary:hover { background: rgba(255,255,255,0.05); border-color: white; transform: translateY(-3px); }
    </style>
</head>
<body>
    <div class="bg-glow"></div>
    <div class="landing">
        <h1>High-Frequency Intelligence.<br><span class="gradient-text">Zero Latency Execution.</span></h1>
        <p>A decoupled, event-driven algorithmic trading engine powered by Redis Pub/Sub, CouchDB, and Machine Learning inference.</p>
        <div class="btn-container">
            <a href="/login" class="btn btn-primary">Launch Terminal</a>
            <a href="/register" class="btn btn-secondary">Provision Node</a>
        </div>
    </div>
</body>
</html>
'''

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Terminal | Active</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0a0f1c;
            --bg-panel: rgba(18, 25, 45, 0.6);
            --border-light: rgba(255, 255, 255, 0.08);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --brand: #00f2fe;
            --brand-gradient: linear-gradient(135deg, #00f2fe, #4facfe);
            --up: #00ff87;
            --down: #ff3366;
            --warn: #ffd966;
            --font-sans: 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }
        * { margin:0; padding:0; box-sizing:border-box; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-base); }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--brand); }
        body { background-color: var(--bg-base); color: var(--text-main); font-family: var(--font-sans); padding: 20px; min-height: 100vh; }
        .top-bar { display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; background: var(--bg-panel); backdrop-filter: blur(12px); border: 1px solid var(--border-light); border-radius: 16px; margin-bottom: 24px; }
        .logo { font-size: 1.5rem; font-weight: 700; background: var(--brand-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px; }
        .sys-status { display: flex; gap: 16px; align-items: center; font-size: 0.875rem; font-weight: 500; }
        .status-badge { display: flex; align-items: center; gap: 8px; padding: 6px 12px; background: rgba(0,0,0,0.3); border-radius: 20px; border: 1px solid var(--border-light); }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--up); box-shadow: 0 0 10px var(--up); animation: pulse 2s infinite; }
        .op-tag { color: var(--text-muted); }
        .logout { color: var(--text-main); text-decoration: none; padding: 6px 16px; background: rgba(255, 51, 102, 0.1); border: 1px solid rgba(255, 51, 102, 0.3); border-radius: 20px; transition: 0.2s; }
        .logout:hover { background: rgba(255, 51, 102, 0.2); }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-bottom: 24px; }
        .kpi-card { background: var(--bg-panel); backdrop-filter: blur(12px); border: 1px solid var(--border-light); border-radius: 16px; padding: 20px; display: flex; flex-direction: column; justify-content: space-between; }
        .kpi-title { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 8px; }
        .kpi-value { font-size: 2rem; font-weight: 700; font-family: var(--font-mono); }
        .kpi-value.brand { color: var(--brand); }
        .kpi-value.up { color: var(--up); }
        .kpi-value.warn { color: var(--warn); }
        .dashboard-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 24px; margin-bottom: 24px; }
        .grid-3-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; margin-bottom: 24px; }
        .panel { background: var(--bg-panel); backdrop-filter: blur(12px); border: 1px solid var(--border-light); border-radius: 16px; display: flex; flex-direction: column; overflow: hidden; }
        .panel-head { padding: 16px 20px; border-bottom: 1px solid var(--border-light); font-weight: 600; display: flex; justify-content: space-between; align-items: center; }
        .panel-body { padding: 20px; flex: 1; overflow-y: auto; }
        .panel-body.no-pad { padding: 0; }
        .chart-box { height: 300px; width: 100%; position: relative; }
        table { width: 100%; border-collapse: collapse; text-align: left; }
        th { padding: 12px 20px; background: rgba(0,0,0,0.2); font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; position: sticky; top: 0; backdrop-filter: blur(5px); }
        td { padding: 12px 20px; border-bottom: 1px solid rgba(255,255,255,0.03); font-size: 0.875rem; transition: background 0.2s; }
        tr:hover td { background: rgba(255,255,255,0.02); }
        .mono { font-family: var(--font-mono); }
        .up { color: var(--up); }
        .down { color: var(--down); }
        .tag { padding: 4px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; background: rgba(255,255,255,0.05); }
        .tag.up { color: #000; background: var(--up); }
        .tag.down { color: #fff; background: var(--down); }
        .trade-form { display: flex; gap: 12px; }
        select, input { flex: 1; background: rgba(0,0,0,0.3); border: 1px solid var(--border-light); color: white; padding: 12px; border-radius: 8px; font-family: var(--font-sans); outline: none; transition: 0.2s; }
        select:focus, input:focus { border-color: var(--brand); }
        .btn { padding: 12px 24px; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; transition: 0.2s; font-family: var(--font-sans); }
        .btn-buy { background: var(--up); color: #000; }
        .btn-buy:hover { box-shadow: 0 0 15px rgba(0,255,135,0.4); }
        .btn-sell { background: var(--down); color: #fff; }
        .btn-sell:hover { box-shadow: 0 0 15px rgba(255,51,102,0.4); }
        .btn-ai { background: transparent; border: 1px solid var(--brand); color: var(--brand); padding: 6px 12px; font-size: 0.75rem; border-radius: 4px; cursor: pointer; }
        .btn-ai:hover { background: var(--brand); color: #000; }
        .news-item { padding: 16px 20px; border-bottom: 1px solid rgba(255,255,255,0.03); transition: 0.2s; display: block; text-decoration: none; }
        .news-item:hover { background: rgba(255,255,255,0.02); transform: translateX(5px); }
        .news-title { color: var(--text-main); font-weight: 500; font-size: 0.875rem; margin-bottom: 6px; line-height: 1.4; }
        .news-meta { color: var(--text-muted); font-size: 0.75rem; }
        .flash-up { animation: flash-green 0.8s ease-out; }
        .flash-down { animation: flash-red 0.8s ease-out; }
        @keyframes flash-green { 0% { background-color: rgba(0, 255, 135, 0.2); } 100% { background-color: transparent; } }
        @keyframes flash-red { 0% { background-color: rgba(255, 51, 102, 0.2); } 100% { background-color: transparent; } }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
        .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(10, 15, 28, 0.85); backdrop-filter: blur(8px); z-index: 1000; align-items: center; justify-content: center; }
        .modal { background: #12192d; border: 1px solid var(--brand); border-radius: 16px; width: 100%; max-width: 450px; box-shadow: 0 20px 50px rgba(0,242,254,0.2); }
        .modal-head { padding: 20px; border-bottom: 1px solid var(--border-light); display: flex; justify-content: space-between; align-items: center; }
        .modal-title { font-weight: 700; font-size: 1.25rem; color: var(--brand); }
        .modal-close { cursor: pointer; color: var(--text-muted); font-size: 1.5rem; transition: 0.2s; }
        .modal-close:hover { color: white; }
        .modal-body { padding: 24px 20px; }
        .m-row { display: flex; justify-content: space-between; margin-bottom: 16px; font-size: 0.95rem; }
        .m-label { color: var(--text-muted); }
        .m-val { font-family: var(--font-mono); font-size: 1.1rem; font-weight: 600; }
        .conf-bar-bg { height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; margin-top: 20px; overflow: hidden; }
        .conf-bar-fill { height: 100%; background: var(--brand-gradient); width: 0%; transition: width 1s cubic-bezier(0.16, 1, 0.3, 1); }
    </style>
</head>
<body>
    <div class="top-bar">
        <div class="logo">QUANTUM ENGINE</div>
        <div class="sys-status">
            <div class="status-badge" id="market-badge"><div class="dot"></div> <span id="market-status">SYNCING</span></div>
            <div class="op-tag">NODE: {{ username }}</div>
            <a href="/logout" class="logout">Disconnect</a>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Net Portfolio Value</div>
            <div class="kpi-value brand mono" id="port-total">&#8377;0.00</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Global Market Cap</div>
            <div class="kpi-value up mono" id="sys-cap">&#8377;0.00</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">NSE Assets</div>
            <div class="kpi-value mono" id="kpi-nse">0</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">NASDAQ Assets</div>
            <div class="kpi-value mono" id="kpi-us">0</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Crypto Assets</div>
            <div class="kpi-value warn mono" id="kpi-crypto">0</div>
        </div>
    </div>

    <div class="dashboard-grid" style="grid-template-columns: 1fr 2fr;">
        <div class="panel">
            <div class="panel-head">Order Execution</div>
            <div class="panel-body">
                <div class="trade-form" style="margin-bottom: 16px;">
                    <select id="trade-sym"></select>
                    <input type="number" id="trade-qty" placeholder="Quantity">
                </div>
                <div class="trade-form">
                    <button class="btn btn-buy" onclick="execTrade(\'buy\')" style="flex:1">Buy Limit</button>
                    <button class="btn btn-sell" onclick="execTrade(\'sell\')" style="flex:1">Sell Limit</button>
                </div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-head">Active Positions</div>
            <div class="panel-body no-pad">
                <div style="max-height: 200px; overflow-y: auto;">
                    <table>
                        <thead><tr><th>Symbol</th><th>Quantity</th><th>Unit Price</th><th>Total Value</th></tr></thead>
                        <tbody id="pos-body"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <div class="dashboard-grid">
        <div class="panel">
            <div class="panel-head">
                <span>Technical Analysis (7D)</span>
                <span id="chart-lbl" style="color: var(--brand); font-family: var(--font-mono);">RELIANCE</span>
            </div>
            <div class="panel-body">
                <div class="chart-box"><canvas id="mainChart"></canvas></div>
            </div>
        </div>
        <div class="panel" style="display: flex; flex-direction: column;">
            <div class="panel-head">System Distribution</div>
            <div class="panel-body" style="display: flex; flex-direction: column; gap: 20px;">
                <div style="flex: 1; display: flex; align-items: center; justify-content: center;">
                    <div style="width: 150px; height: 150px;"><canvas id="exchangeChart"></canvas></div>
                    <div style="margin-left: 20px; font-size: 0.8rem; color: var(--text-muted);">
                        <div><span style="color:#00ff87">&#9632;</span> NSE</div>
                        <div><span style="color:#00f2fe">&#9632;</span> NASDAQ</div>
                        <div><span style="color:#b026ff">&#9632;</span> Crypto</div>
                    </div>
                </div>
                <div style="flex: 1; display: flex; align-items: center; justify-content: center; border-top: 1px solid var(--border-light); padding-top: 20px;">
                    <div style="width: 150px; height: 150px;"><canvas id="signalChart"></canvas></div>
                    <div style="margin-left: 20px; font-size: 0.8rem; color: var(--text-muted);">
                        <div><span style="color:#00ff87">&#9632;</span> BUY</div>
                        <div><span style="color:#ff3366">&#9632;</span> SELL</div>
                        <div><span style="color:#64748b">&#9632;</span> HOLD</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="grid-3-col">
        <div class="panel">
            <div class="panel-head">Real-Time Tape</div>
            <div class="panel-body no-pad" style="max-height: 400px;">
                <table>
                    <thead><tr><th>Asset</th><th style="text-align:right">Price</th><th style="text-align:right">24h%</th></tr></thead>
                    <tbody id="prices-body"></tbody>
                </table>
            </div>
        </div>
        <div class="panel">
            <div class="panel-head">Technical Indicators</div>
            <div class="panel-body no-pad" style="max-height: 400px;">
                <table>
                    <thead><tr><th>Asset</th><th>RSI (14)</th><th>Algorithm</th></tr></thead>
                    <tbody id="rsi-body"></tbody>
                </table>
            </div>
        </div>
        <div class="panel">
            <div class="panel-head">Intelligence Feed</div>
            <div class="panel-body no-pad" id="news-container" style="max-height: 400px;">
                <div style="padding: 20px; color: var(--text-muted);">Awaiting ticker selection...</div>
            </div>
        </div>
    </div>

    <div class="panel" style="margin-bottom: 40px;">
        <div class="panel-head">Full Institutional Ledger & Inference</div>
        <div class="panel-body no-pad" style="max-height: 400px;">
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th><th>Exchange</th><th style="text-align:right">Last</th>
                        <th style="text-align:right">Net Chg</th><th style="text-align:center">RSI</th>
                        <th style="text-align:right">Volume</th><th style="text-align:center">Action</th>
                    </tr>
                </thead>
                <tbody id="detailed-body"></tbody>
            </table>
        </div>
    </div>

    <div class="modal-overlay" id="ai-modal">
        <div class="modal">
            <div class="modal-head">
                <span class="modal-title" id="m-title">Matrix Analysis</span>
                <span class="modal-close" onclick="closeModal()">&times;</span>
            </div>
            <div class="modal-body">
                <div class="m-row"><span class="m-label">Current Execution</span><span class="m-val" id="m-cur">--</span></div>
                <div class="m-row"><span class="m-label">T+1 Projection</span><span class="m-val" id="m-pred">--</span></div>
                <div class="m-row" style="margin-top:24px;"><span class="m-label">Model Confidence</span><span class="m-val" id="m-conf">--</span></div>
                <div class="conf-bar-bg"><div class="conf-bar-fill" id="m-bar"></div></div>
                <div style="margin-top: 16px; font-size: 0.75rem; color: var(--text-muted); text-align: right;" id="m-engine">--</div>
            </div>
        </div>
    </div>

    <script>
        let mainChart, exChart, sigChart;
        let curSym = 'RELIANCE';
        let prevPrices = {};

        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'JetBrains Mono', monospace";

        function initCharts() {
            mainChart = new Chart(document.getElementById('mainChart').getContext('2d'), {
                type: 'line',
                data: { labels: [], datasets: [{ data: [], borderColor: '#00f2fe', backgroundColor: 'rgba(0, 242, 254, 0.1)', borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3 }]},
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { position: 'right', grid: { color: 'rgba(255,255,255,0.05)' }, border: { display: false } } } }
            });
            exChart = new Chart(document.getElementById('exchangeChart').getContext('2d'), {
                type: 'doughnut',
                data: { labels: ['NSE', 'NASDAQ', 'Crypto'], datasets: [{ data: [0,0,0], backgroundColor: ['#00ff87', '#00f2fe', '#b026ff'], borderWidth: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, cutout: '75%', plugins: { legend: { display: false } } }
            });
            sigChart = new Chart(document.getElementById('signalChart').getContext('2d'), {
                type: 'doughnut',
                data: { labels: ['BUY', 'SELL', 'HOLD'], datasets: [{ data: [0,0,0], backgroundColor: ['#00ff87', '#ff3366', '#64748b'], borderWidth: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, cutout: '75%', plugins: { legend: { display: false } } }
            });
        }

        function loadPort() {
            fetch('/api/portfolio').then(r=>r.json()).then(d=>{
                document.getElementById('port-total').innerText = '\u20b9' + d.total_value.toLocaleString('en-IN', {minimumFractionDigits:2});
                document.getElementById('pos-body').innerHTML = d.holdings.map(h=>`
                    <tr>
                        <td class="mono" style="font-weight:600;">${h.symbol}</td>
                        <td class="mono">${h.quantity}</td>
                        <td class="mono">${h.price.toFixed(2)}</td>
                        <td class="mono" style="color:var(--brand)">\u20b9${h.value.toLocaleString('en-IN')}</td>
                    </tr>
                `).join('');
            }).catch(()=>{});
        }

        function execTrade(type) {
            const sym = document.getElementById('trade-sym').value;
            const qty = document.getElementById('trade-qty').value;
            if(!qty || qty<=0) return;
            fetch(`/api/${type}`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({symbol:sym, quantity:parseInt(qty)}) })
            .then(r=>r.json()).then(d=>{
                if(d.success) { loadPort(); document.getElementById('trade-qty').value=''; }
                else alert(d.error);
            }).catch(()=>{});
        }

        function loadHist(sym) {
            curSym = sym; document.getElementById('chart-lbl').innerText = sym;
            fetch(`/api/history/${sym}`).then(r=>r.json()).then(d=>{
                if(d.dates) { mainChart.data.labels = d.dates; mainChart.data.datasets[0].data = d.prices; mainChart.update('none'); }
            }).catch(()=>{});
        }

        function loadNewsFeed(sym) {
            fetch(`/api/news/${sym}`).then(r=>r.json()).then(d=>{
                const c = document.getElementById('news-container');
                if(d.news) {
                    c.innerHTML = d.news.map(n=>`
                        <a href="${n.link}" target="_blank" class="news-item">
                            <div class="news-title">${n.title}</div>
                            <div class="news-meta">${n.source} \u2022 ${new Date(n.published).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}</div>
                        </a>
                    `).join('');
                }
            }).catch(()=>{});
        }

        function clickRow(sym) {
            if(curSym !== sym) { loadHist(sym); loadNewsFeed(sym); }
        }

        function runInference(sym, e) {
            e.stopPropagation();
            const btn = e.target; const orig = btn.innerText; btn.innerText = 'WAIT...';
            fetch(`/api/predict/${sym}`).then(r=>r.json()).then(d=>{
                btn.innerText = orig;
                if(d.error) return alert(d.error);
                document.getElementById('m-title').innerText = `${sym} Matrix`;
                document.getElementById('m-cur').innerText = d.current_price.toFixed(2);
                const p = document.getElementById('m-pred');
                p.innerText = d.predicted_price.toFixed(2);
                p.className = 'm-val ' + (d.predicted_price > d.current_price ? 'up' : 'down');
                const conf = (d.confidence * 100).toFixed(1);
                document.getElementById('m-conf').innerText = conf + '%';
                document.getElementById('m-bar').style.width = '0%';
                document.getElementById('m-engine').innerText = 'Engine: ' + d.model_used;
                document.getElementById('ai-modal').style.display = 'flex';
                setTimeout(() => document.getElementById('m-bar').style.width = conf + '%', 50);
            }).catch(err => { btn.innerText = orig; alert('ML engine error: ' + err); });
        }

        function closeModal() { document.getElementById('ai-modal').style.display = 'none'; }

        function formatPx(val, ex) { return (ex==='NASDAQ (US)'?'$':'\u20b9') + parseFloat(val).toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2}); }

        function pollData() {
            fetch('/api/data').then(r=>r.json()).then(d=>{
                document.getElementById('sys-cap').innerText = '\u20b9' + (d.mapreduce_total/10000000).toFixed(2) + ' Cr';
                const mkt = d.market_status;
                const mktBadge = document.getElementById('market-badge');
                mktBadge.innerHTML = `<div class="dot" style="background:${mkt.includes('Open')?'var(--up)':'var(--text-muted)'}; box-shadow: 0 0 10px ${mkt.includes('Open')?'var(--up)':'transparent'};"></div> <span>${mkt.toUpperCase()}</span>`;
                let nse=0, us=0, crypto=0, b=0, s=0, h=0;
                let selOpts = '';
                let htmlPrices = '', htmlRsi = '', htmlDetail = '';
                d.stock_data.forEach(stk => {
                    if(stk.exchange==='NSE (India)') nse++; else if(stk.exchange==='NASDAQ (US)') us++; else crypto++;
                    if(stk.signal==='BUY') b++; else if(stk.signal==='SELL') s++; else h++;
                    selOpts += `<option value="${stk.ticker}">${stk.ticker}</option>`;
                    const isUp = stk.change >= 0;
                    const cCls = isUp ? 'up' : 'down';
                    const sign = isUp ? '+' : '';
                    let sigCls = 'tag ';
                    if(stk.signal==='BUY') sigCls += 'up'; else if(stk.signal==='SELL') sigCls += 'down'; else sigCls += 'tag';
                    const oldPx = prevPrices[stk.ticker];
                    let flash = '';
                    if (oldPx && oldPx !== stk.price) flash = stk.price > oldPx ? 'flash-up' : 'flash-down';
                    prevPrices[stk.ticker] = stk.price;
                    const pFmt = formatPx(stk.price, stk.exchange);
                    htmlPrices += `<tr onclick="clickRow('${stk.ticker}')"><td class="mono" style="font-weight:600;">${stk.ticker}</td><td class="mono ${flash}" style="text-align:right">${pFmt}</td><td class="mono ${cCls}" style="text-align:right">${sign}${stk.change_percent.toFixed(2)}%</td></tr>`;
                    htmlRsi += `<tr onclick="clickRow('${stk.ticker}')"><td class="mono" style="font-weight:600;">${stk.ticker}</td><td class="mono" style="color:${stk.rsi<30?'var(--up)':stk.rsi>70?'var(--down)':'var(--text-main)'}">${stk.rsi.toFixed(1)}</td><td style="text-align:left"><span class="${sigCls}">${stk.signal}</span></td></tr>`;
                    htmlDetail += `<tr onclick="clickRow('${stk.ticker}')"><td class="mono" style="font-weight:600; color:var(--brand)">${stk.ticker}</td><td style="font-size:0.75rem; color:var(--text-muted)">${stk.exchange}</td><td class="mono ${flash}" style="text-align:right">${pFmt}</td><td class="mono ${cCls}" style="text-align:right">${sign}${stk.change.toFixed(2)}</td><td class="mono" style="text-align:center">${stk.rsi.toFixed(1)}</td><td class="mono" style="text-align:right; color:var(--text-muted)">${(stk.volume||0).toLocaleString()}</td><td style="text-align:center"><button class="btn-ai" onclick="runInference('${stk.ticker}', event)">Execute ML</button></td></tr>`;
                });
                document.getElementById('kpi-nse').innerText = nse;
                document.getElementById('kpi-us').innerText = us;
                document.getElementById('kpi-crypto').innerText = crypto;
                exChart.data.datasets[0].data = [nse, us, crypto]; exChart.update();
                sigChart.data.datasets[0].data = [b, s, h]; sigChart.update();
                document.getElementById('prices-body').innerHTML = htmlPrices;
                document.getElementById('rsi-body').innerHTML = htmlRsi;
                document.getElementById('detailed-body').innerHTML = htmlDetail;
                const sel = document.getElementById('trade-sym');
                if(sel.options.length === 0) sel.innerHTML = selOpts;
            }).catch(()=>{});
        }

        initCharts(); pollData(); loadPort(); loadHist(curSym); loadNewsFeed(curSym);
        setInterval(() => { pollData(); loadPort(); }, 5000);
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"APP STARTUP: Starting Flask dev server on port {port}", flush=True, file=sys.stderr)
    app.run(host='0.0.0.0', port=port, debug=False)