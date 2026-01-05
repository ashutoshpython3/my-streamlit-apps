import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
import json
import yfinance as yf
import re
from bs4 import BeautifulSoup
import os

# Set page configuration
st.set_page_config(
    page_title="Indian Stock Portfolio Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Function to read the portfolio file
def read_portfolio():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    portfolio_path = os.path.join(script_dir, "portfolio.csv")
    
    try:
        portfolio = pd.read_csv(portfolio_path)
        return portfolio
    except FileNotFoundError:
        st.error(f"Portfolio file not found at {portfolio_path}")
        return pd.DataFrame()

# File paths for data persistence
PORTFOLIO_FILE = "portfolio_data.json"
TRANSACTIONS_FILE = "transactions_data.json"

# Initialize session state variables
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
    
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
    
if 'stock_data_cache' not in st.session_state:
    st.session_state.stock_data_cache = {}

# Upstox API Configuration
# Replace with your actual API key and access token
UPSTOX_API_KEY = "9a56569a-142f-4247-a863-f4e663fb03f1"
UPSTOX_ACCESS_TOKEN = "smgyr2big7"

# Base URL for Upstox API
BASE_URL = "https://api.upstox.com/v2"

# Headers for API requests
headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'
}

# Predefined instrument keys for popular stocks - Reduced to 5 stocks
INSTRUMENT_KEYS = {
    "RELIANCE": "NSE_EQ|INE528G01035",
    "TCS": "NSE_EQ|INE461B01029",
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "INFY": "NSE_EQ|INE009A01021",
    "HINDUNILVR": "NSE_EQ|INE029A01027"
}

# Nifty 50 instrument key
NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# Function to save portfolio data to file
def save_portfolio_data():
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(st.session_state.portfolio, f)
    except Exception as e:
        st.error(f"Error saving portfolio data: {str(e)}")

# Function to load portfolio data from file
def load_portfolio_data():
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r') as f:
                st.session_state.portfolio = json.load(f)
    except Exception as e:
        st.error(f"Error loading portfolio data: {str(e)}")
        st.session_state.portfolio = []

# Function to save transactions data to file
def save_transactions_data():
    try:
        with open(TRANSACTIONS_FILE, 'w') as f:
            json.dump(st.session_state.transactions, f)
    except Exception as e:
        st.error(f"Error saving transactions data: {str(e)}")

# Function to load transactions data from file
def load_transactions_data():
    try:
        if os.path.exists(TRANSACTIONS_FILE):
            with open(TRANSACTIONS_FILE, 'r') as f:
                st.session_state.transactions = json.load(f)
    except Exception as e:
        st.error(f"Error loading transactions data: {str(e)}")
        st.session_state.transactions = []

# Load data at the start of the app
load_portfolio_data()
load_transactions_data()

# Function to get historical data from Upstox (fixed based on official documentation)
def get_historical_data_upstox(instrument_key, interval='day', from_date=None, to_date=None):
    try:
        if from_date is None:
            from_date = datetime.now() - timedelta(days=1460)  # 4 years
        
        if to_date is None:
            to_date = datetime.now()
        
        # Format dates to YYYY-MM-DD
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')
        
        # Build URL according to Upstox API documentation
        # Fixed: changed '1day' to 'day' as per API requirements
        url = f"{BASE_URL}/historical-candle/{instrument_key}/{interval}/{to_date_str}/{from_date_str}"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            st.error(f"Failed to get historical data from Upstox: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        
        # Check if we have candles data
        if 'data' not in data or 'candles' not in data['data']:
            st.error(f"Invalid data format from Upstox API: {data}")
            return None
        
        # Convert to DataFrame
        candles = data['data']['candles']
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Convert timezone to Asia/Kolkata if not already in that timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('Asia/Kolkata')
        else:
            df.index = df.index.tz_convert('Asia/Kolkata')
        
        # Rename columns to match our format (capitalized)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
        
        return df
    except Exception as e:
        st.error(f"Error getting historical data from Upstox: {str(e)}")
        return None

# Function to get quote data from Upstox (fixed based on official documentation)
def get_quote_data_upstox(instrument_key):
    try:
        url = f"{BASE_URL}/quote/{instrument_key}"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            st.error(f"Failed to get quote data from Upstox: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        
        # Check if we have valid data
        if 'data' not in data:
            st.error(f"Invalid data format from Upstox API: {data}")
            return None
            
        return data['data']
    except Exception as e:
        st.error(f"Error getting quote data from Upstox: {str(e)}")
        return None

# Function to get stock data from Yahoo Finance with retry mechanism
def get_stock_data_yahoo(symbol, period="4y", max_retries=3):
    for attempt in range(max_retries):
        try:
            # Add .NS for NSE stocks
            ticker = yf.Ticker(f"{symbol}.NS")
            hist_data = ticker.history(period=period)
            
            # Get current info
            info = ticker.info
            
            # Make sure we have data
            if hist_data.empty:
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                st.error(f"No historical data found for {symbol}")
                return None
            
            # Rename columns to match our format (capitalized)
            hist_data = hist_data.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # Create a quote-like structure
            quote_data = {
                'last_price': hist_data['Close'].iloc[-1],
                'open_price': hist_data['Open'].iloc[-1],
                'high_price': hist_data['High'].iloc[-1],
                'low_price': hist_data['Low'].iloc[-1],
                'volume': hist_data['Volume'].iloc[-1]
            }
            
            return {
                'history': hist_data,
                'quote': quote_data,
                'info': info
            }
        except Exception as e:
            if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                # Exponential backoff for rate limiting
                wait_time = 2 ** attempt
                st.warning(f"Rate limited by Yahoo Finance. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            elif attempt < max_retries - 1:
                time.sleep(1)  # Brief wait before retry
                continue
            else:
                st.error(f"Error fetching data from Yahoo Finance for {symbol}: {str(e)}")
                return None

# Function to get fundamental data from Yahoo Finance (replacing MoneyControl)
def get_fundamental_data_yahoo(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        
        # Extract fundamental data
        fundamental_data = {}
        
        # Market Cap
        if 'marketCap' in info:
            fundamental_data['market_cap'] = info['marketCap']
        
        # Revenue
        if 'totalRevenue' in info:
            fundamental_data['revenue'] = info['totalRevenue']
        
        # PEG Ratio
        if 'pegRatio' in info:
            fundamental_data['peg_ratio'] = info['pegRatio']
        
        # Revenue Growth
        if 'revenueGrowth' in info:
            fundamental_data['revenue_growth'] = info['revenueGrowth'] * 100  # Convert to percentage
        
        # EPS Growth
        if 'earningsGrowth' in info:
            fundamental_data['eps_growth'] = info['earningsGrowth'] * 100  # Convert to percentage
        
        # Quarterly Profit
        if 'netIncomeToCommon' in info:
            fundamental_data['quarterly_profit'] = info['netIncomeToCommon']
        
        # Previous Quarterly Profit (approximation using trailing twelve months)
        if 'netIncomeToCommon' in info and 'earningsGrowth' in info:
            growth_rate = info['earningsGrowth']
            current_profit = info['netIncomeToCommon']
            # Approximate previous quarter's profit
            fundamental_data['prev_quarterly_profit'] = current_profit / (1 + growth_rate)
        
        return fundamental_data
    except Exception as e:
        st.error(f"Error getting fundamental data from Yahoo Finance for {symbol}: {str(e)}")
        return None

# Function to get stock list (reduced to 5 stocks)
def get_stock_list():
    # For demo purposes, we'll use a predefined list of 5 popular Indian stocks
    stocks = [
        {"symbol": "RELIANCE", "name": "Reliance Industries", "exchange": "NSE"},
        {"symbol": "TCS", "name": "Tata Consultancy Services", "exchange": "NSE"},
        {"symbol": "HDFCBANK", "name": "HDFC Bank", "exchange": "NSE"},
        {"symbol": "INFY", "name": "Infosys", "exchange": "NSE"},
        {"symbol": "HINDUNILVR", "name": "Hindustan Unilever", "exchange": "NSE"}
    ]
    return stocks

# Function to get stock data using multiple sources
def get_stock_data(symbol, period="4y"):
    if symbol in st.session_state.stock_data_cache:
        return st.session_state.stock_data_cache[symbol]
    
    try:
        # First try to get data from Upstox if we have the instrument key
        if symbol in INSTRUMENT_KEYS:
            instrument_key = INSTRUMENT_KEYS[symbol]
            
            # Fetch historical data
            # Fixed: changed '1day' to 'day' as per API requirements
            historical_data = get_historical_data_upstox(
                instrument_key=instrument_key,
                interval='day',  # Fixed: use 'day' as per Upstox API
                from_date=datetime.now() - timedelta(days=1460),
                to_date=datetime.now()
            )
            
            if historical_data is None or historical_data.empty:
                # Fall back to Yahoo Finance
                stock_data = get_stock_data_yahoo(symbol, period)
                if stock_data:
                    st.session_state.stock_data_cache[symbol] = stock_data
                    return stock_data
                return None
            
            # Get current quote
            quote_data = get_quote_data_upstox(instrument_key)
            
            if quote_data is None:
                # Fall back to Yahoo Finance
                stock_data = get_stock_data_yahoo(symbol, period)
                if stock_data:
                    st.session_state.stock_data_cache[symbol] = stock_data
                    return stock_data
                return None
            
            # Get fundamental data from Yahoo Finance (replacing MoneyControl)
            fundamental_data = get_fundamental_data_yahoo(symbol)
            
            # Cache the data
            st.session_state.stock_data_cache[symbol] = {
                'history': historical_data,
                'quote': quote_data,
                'info': fundamental_data
            }
            
            return st.session_state.stock_data_cache[symbol]
        else:
            # Use Yahoo Finance for stocks not in our predefined list
            stock_data = get_stock_data_yahoo(symbol, period)
            if stock_data:
                # Get fundamental data from Yahoo Finance
                fundamental_data = get_fundamental_data_yahoo(symbol)
                
                st.session_state.stock_data_cache[symbol] = {
                    'history': stock_data['history'],
                    'quote': stock_data['quote'],
                    'info': fundamental_data
                }
                return st.session_state.stock_data_cache[symbol]
            
            return None
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Function to calculate EMA
def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

# Function to calculate SMA
def calculate_sma(data, period):
    return data['Close'].rolling(window=period).mean()

# Function to detect Bull Flag pattern (Goverdhan Strategy)
def detect_bull_flag(data, ema9, ema21):
    # Check if price is above both EMAs
    if data['Close'].iloc[-1] < ema9.iloc[-1] or data['Close'].iloc[-1] < ema21.iloc[-1]:
        return False
    
    # Check for initial spike (last 5 candles)
    recent_data = data.tail(10)
    spike_high = recent_data['High'].max()
    spike_idx = recent_data['High'].idxmax()
    
    # Check for pullback (flag) after spike
    pullback_data = data.loc[spike_idx:]
    if len(pullback_data) < 3:
        return False
    
    # Check if pullback is contained within 5-10% of spike
    pullback_high = pullback_data['High'].max()
    pullback_low = pullback_data['Low'].min()
    
    if (pullback_high - pullback_low) / spike_high > 0.1:  # More than 10% range
        return False
    
    # Check if current price is breaking out of flag
    if data['Close'].iloc[-1] > pullback_high * 0.98:  # Near flag high
        return True
    
    return False

# Function to detect EMA Kiss & Fly pattern (Goverdhan Strategy)
def detect_ema_kiss_fly(data, ema9, ema21):
    # Check if EMAs are close to each other
    ema_diff = abs(ema9.iloc[-1] - ema21.iloc[-1]) / ema21.iloc[-1]
    if ema_diff > 0.02:  # More than 2% difference
        return False
    
    # Check if price is bouncing off EMAs
    recent_data = data.tail(10)
    low_point = recent_data['Low'].min()
    low_idx = recent_data['Low'].idxmin()
    
    # Check if low point is near EMAs
    ema_at_low = (ema9.loc[low_idx] + ema21.loc[low_idx]) / 2
    if abs(low_point - ema_at_low) / ema_at_low > 0.03:  # More than 3% away
        return False
    
    # Check if price is moving up after bounce
    if data['Close'].iloc[-1] > data['Close'].iloc[-5]:
        return True
    
    return False

# Function to detect Volatility Contraction Pattern (Goverdhan Strategy)
def detect_vcp(data, ema21):
    # Get last 30 days of data
    recent_data = data.tail(30)
    
    # Calculate range (high - low) for each day
    ranges = recent_data['High'] - recent_data['Low']
    
    # Check if ranges are contracting
    if not (ranges.iloc[-1] < ranges.iloc[-5] < ranges.iloc[-10]):
        return False
    
    # Check if price is hugging EMA21
    ema_close_diff = abs(recent_data['Close'] - ema21.tail(30)) / ema21.tail(30)
    if ema_close_diff.mean() > 0.05:  # More than 5% average difference
        return False
    
    # Check for breakout
    if data['Close'].iloc[-1] > data['High'].tail(10).max() * 0.98:
        return True
    
    return False

# Function to analyze Goverdhan Strategy (adapted for daily)
def analyze_goverdhan_strategy(data):
    if data is None or len(data) < 50:
        return {"signal": "hold", "pattern": "Insufficient data", "confidence": 0}
    
    # Calculate 9-day and 21-day EMAs
    ema9 = calculate_ema(data, 9)
    ema21 = calculate_ema(data, 21)
    
    # Calculate volume averages
    volume_ma = data['Volume'].rolling(window=20).mean()
    
    # Current values
    current_price = data['Close'].iloc[-1]
    current_volume = data['Volume'].iloc[-1]
    
    # Check for Bull Flag Breakout
    if detect_bull_flag(data, ema9, ema21) and current_volume > volume_ma.iloc[-1] * 1.5:
        return {"signal": "buy", "pattern": "Bull Flag Breakout", "confidence": 85}
    
    # Check for EMA Kiss & Fly
    if detect_ema_kiss_fly(data, ema9, ema21):
        return {"signal": "buy", "pattern": "EMA Kiss & Fly", "confidence": 80}
    
    # Check for Volatility Contraction Pattern
    if detect_vcp(data, ema21):
        return {"signal": "buy", "pattern": "Volatility Contraction", "confidence": 75}
    
    # Check for breakdown below EMAs
    if current_price < ema9.iloc[-1] and current_price < ema21.iloc[-1]:
        return {"signal": "sell", "pattern": "Below EMAs", "confidence": 70}
    
    return {"signal": "hold", "pattern": "No clear pattern", "confidence": 50}

# Function to detect Base N Break pattern (Oliver Kell)
def detect_base_n_break(data, ema50):
    # Check if price is above EMA50
    if data['Close'].iloc[-1] < ema50.iloc[-1]:
        return False
    
    # Look for consolidation (base) - last 50 days
    recent_data = data.tail(50)
    
    # Calculate range of consolidation
    high = recent_data['High'].max()
    low = recent_data['Low'].min()
    range_pct = (high - low) / low
    
    # Range should be less than 20% for a proper base
    if range_pct > 0.2:
        return False
    
    # Check for breakout - current price near high of base
    if data['Close'].iloc[-1] > high * 0.98:
        # Check volume on breakout
        volume_ma = data['Volume'].rolling(window=20).mean()
        if data['Volume'].iloc[-1] > volume_ma.iloc[-1] * 1.5:
            return True
    
    return False

# Function to detect Wedge Pop pattern (Oliver Kell)
def detect_wedge_pop(data, ema50):
    # Check if price is above EMA50
    if data['Close'].iloc[-1] < ema50.iloc[-1]:
        return False
    
    # Look for wedge pattern - last 30 days
    recent_data = data.tail(30)
    
    # Calculate trendlines
    highs = recent_data['High']
    lows = recent_data['Low']
    
    # Simple wedge detection: converging highs and lows
    high_slope = (highs.iloc[-1] - highs.iloc[0]) / len(highs)
    low_slope = (lows.iloc[-1] - lows.iloc[0]) / len(lows)
    
    # For a wedge, highs should be decreasing and lows increasing
    if high_slope < 0 and low_slope > 0:
        # Check for breakout
        if data['Close'].iloc[-1] > highs.iloc[-1] * 0.98:
            # Check volume
            volume_ma = data['Volume'].rolling(window=20).mean()
            if data['Volume'].iloc[-1] > volume_ma.iloc[-1] * 1.5:
                return True
    
    return False

# Function to detect Exhaustion Extension pattern (Oliver Kell)
def detect_exhaustion_extension(data, ema50):
    # Check if price is far above EMA50
    if data['Close'].iloc[-1] < ema50.iloc[-1] * 1.3:
        return False
    
    # Look for parabolic move - last 10 days
    recent_data = data.tail(10)
    
    # Calculate percentage change
    pct_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
    
    # If price moved up more than 20% in 10 days
    if pct_change > 0.2:
        # Check for reversal signs - last 3 candles
        last_3 = data.tail(3)
        if last_3['Close'].iloc[-1] < last_3['High'].iloc[-2]:
            return True
    
    return False

# Function to analyze Oliver Kell Strategy
def analyze_oliver_kell_strategy(data):
    if data is None or len(data) < 100:
        return {"signal": "hold", "pattern": "Insufficient data", "confidence": 0}
    
    # Calculate 50-day EMA
    ema50 = calculate_ema(data, 50)
    
    # Current values
    current_price = data['Close'].iloc[-1]
    
    # Check for Base N Break
    if detect_base_n_break(data, ema50):
        return {"signal": "buy", "pattern": "Base N Break", "confidence": 85}
    
    # Check for Wedge Pop
    if detect_wedge_pop(data, ema50):
        return {"signal": "buy", "pattern": "Wedge Pop", "confidence": 80}
    
    # Check for Exhaustion Extension
    if detect_exhaustion_extension(data, ema50):
        return {"signal": "sell", "pattern": "Exhaustion Extension", "confidence": 90}
    
    return {"signal": "hold", "pattern": "No clear pattern", "confidence": 50}

# Function to analyze Relative Strength Strategy
def analyze_relative_strength(stock_data, nifty_data, period=55):
    if stock_data is None or nifty_data is None or len(stock_data) < period or len(nifty_data) < period:
        return {"signal": "hold", "value": 0, "trend": "neutral"}
    
    # Calculate relative strength
    stock_change = stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-period]
    nifty_change = nifty_data['Close'].iloc[-1] / nifty_data['Close'].iloc[-period]
    
    rs = (stock_change / nifty_change) - 1
    
    # Calculate RS trend
    rs_history = []
    for i in range(period, len(stock_data)):
        # Check if index is within bounds
        if i - period >= 0 and i < len(nifty_data) and i < len(stock_data):
            stock_ch = stock_data['Close'].iloc[i] / stock_data['Close'].iloc[i-period]
            nifty_ch = nifty_data['Close'].iloc[i] / nifty_data['Close'].iloc[i-period]
            rs_history.append((stock_ch / nifty_ch) - 1)
    
    if len(rs_history) < 20:
        return {"signal": "hold", "value": rs, "trend": "insufficient data"}
    
    # Calculate trend
    recent_rs = rs_history[-20:]
    if recent_rs[-1] > recent_rs[0] and rs > 0:
        return {"signal": "buy", "value": rs, "trend": "strengthening"}
    elif recent_rs[-1] < recent_rs[0] and rs < 0:
        return {"signal": "sell", "value": rs, "trend": "weakening"}
    
    return {"signal": "hold", "value": rs, "trend": "neutral"}

# Function to analyze GARP Strategy
def analyze_garp_strategy(info):
    if info is None:
        return {"signal": "hold", "score": 0, "criteria": {}}
    
    # Extract fundamental data
    market_cap = info.get('market_cap', 0) or info.get('marketCap', 0)
    revenue = info.get('revenue', 0) or info.get('totalRevenue', 0)
    peg_ratio = info.get('peg_ratio', 0) or info.get('pegRatio', 0)
    revenue_growth = info.get('revenue_growth', 0) or info.get('revenueGrowth', 0)
    eps_growth = info.get('eps_growth', 0) or info.get('earningsGrowth', 0)
    quarterly_profit = info.get('quarterly_profit', 0) or info.get('netIncomeToCommon', 0)
    prev_quarterly_profit = info.get('prev_quarterly_profit', 0) or info.get('prevQuarterNetProfit', 0)
    
    criteria = {
        "Market Cap > 1000 crores": market_cap > 10000000000,
        "Sales Revenue > 1000 crores": revenue > 10000000000,
        "PEG Ratio between 0 and 2": 0 < peg_ratio < 2,
        "Sales Growth > 15% (1Y)": revenue_growth > 15,
        "EPS Growth > 15% (1Y)": eps_growth > 15,
        "Quarterly Profit > 2Q Ago": quarterly_profit > prev_quarterly_profit
    }
    
    score = sum(criteria.values())
    
    return {
        "signal": "buy" if score >= 5 else "sell" if score <= 2 else "hold",
        "score": score,
        "criteria": criteria
    }

# Function to generate overall recommendation
def generate_recommendation(goverdhan, oliver, relative_strength, garp):
    signals = [
        goverdhan.get("signal", "hold"),
        oliver.get("signal", "hold"),
        relative_strength.get("signal", "hold"),
        garp.get("signal", "hold")
    ]
    
    buy_count = signals.count("buy")
    sell_count = signals.count("sell")
    
    if buy_count >= 3:
        return "buy"
    elif sell_count >= 2:
        return "sell"
    else:
        return "hold"

# Function to add stock to portfolio
def add_to_portfolio(symbol, quantity, price, purchase_date=None):
    # Set default purchase date if not provided
    if purchase_date is None:
        purchase_date = datetime.now().strftime('%Y-%m-%d')
    
    # Check if stock already exists in portfolio
    for i, stock in enumerate(st.session_state.portfolio):
        if stock['symbol'] == symbol:
            # Update existing position
            total_quantity = stock['quantity'] + quantity
            total_value = (stock['quantity'] * stock['avg_price']) + (quantity * price)
            avg_price = total_value / total_quantity
            
            st.session_state.portfolio[i] = {
                'symbol': symbol,
                'quantity': total_quantity,
                'avg_price': avg_price,
                'purchase_date': purchase_date
            }
            # Save portfolio data
            save_portfolio_data()
            return
    
    # Add new stock to portfolio
    st.session_state.portfolio.append({
        'symbol': symbol,
        'quantity': quantity,
        'avg_price': price,
        'purchase_date': purchase_date
    })
    # Save portfolio data
    save_portfolio_data()

# Function to sell stock from portfolio
def sell_from_portfolio(symbol, quantity, price):
    for i, stock in enumerate(st.session_state.portfolio):
        if stock['symbol'] == symbol:
            if quantity > stock['quantity']:
                st.error(f"Cannot sell more than you own. You have {stock['quantity']} shares.")
                return 0
            
            # Calculate profit
            profit = (price - stock['avg_price']) * quantity
            
            # Record transaction
            st.session_state.transactions.append({
                'symbol': symbol,
                'quantity': quantity,
                'sell_price': price,
                'buy_price': stock['avg_price'],
                'profit': profit,
                'date': datetime.now().strftime('%Y-%m-%d')
            })
            
            # Update or remove from portfolio
            if quantity == stock['quantity']:
                st.session_state.portfolio.pop(i)
            else:
                st.session_state.portfolio[i]['quantity'] -= quantity
            
            # Save data
            save_portfolio_data()
            save_transactions_data()
            
            return profit
    
    return 0

# Function to get Nifty data
def get_nifty_data(period="4y"):
    if '^NSEI' in st.session_state.stock_data_cache:
        return st.session_state.stock_data_cache['^NSEI']['history']
    
    try:
        # Try to get from Upstox first
        historical_data = get_historical_data_upstox(
            instrument_key=NIFTY_INSTRUMENT_KEY,
            interval='day',  # Fixed: use 'day' as per Upstox API
            from_date=datetime.now() - timedelta(days=1460),
            to_date=datetime.now()
        )
        
        if historical_data is None or historical_data.empty:
            # Fall back to Yahoo Finance
            nifty_data = yf.Ticker("^NSEI").history(period=period)
            
            # Make sure we have data
            if nifty_data.empty:
                st.error("No historical data found for Nifty 50")
                return None
                
            # Rename columns to match our format (capitalized)
            nifty_data = nifty_data.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            st.session_state.stock_data_cache['^NSEI'] = {
                'history': nifty_data,
                'quote': {'last_price': nifty_data['Close'].iloc[-1]},
                'info': {}
            }
            return nifty_data
        
        # Get quote data
        quote_data = get_quote_data_upstox(NIFTY_INSTRUMENT_KEY)
        
        if quote_data is None:
            # Fall back to Yahoo Finance
            nifty_data = yf.Ticker("^NSEI").history(period=period)
            
            # Make sure we have data
            if nifty_data.empty:
                st.error("No historical data found for Nifty 50")
                return None
                
            # Rename columns to match our format (capitalized)
            nifty_data = nifty_data.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            st.session_state.stock_data_cache['^NSEI'] = {
                'history': nifty_data,
                'quote': {'last_price': nifty_data['Close'].iloc[-1]},
                'info': {}
            }
            return nifty_data
        
        st.session_state.stock_data_cache['^NSEI'] = {
            'history': historical_data,
            'quote': quote_data,
            'info': {}
        }
        
        return historical_data
    except Exception as e:
        st.error(f"Error fetching Nifty data: {str(e)}")
        return None

# Sidebar navigation
st.sidebar.title("Portfolio Manager")
page = st.sidebar.selectbox("Select Page", ["Portfolio", "Add Stock", "Sell Stock", "Analysis", "Transactions"])

# Portfolio Page
if page == "Portfolio":
    st.title("Your Stock Portfolio")
    
    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add stocks using the 'Add Stock' page.")
    else:
        # Create a dataframe for portfolio
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        
        # Get current prices
        for i, row in portfolio_df.iterrows():
            stock_data = get_stock_data(row['symbol'])
            if stock_data and 'quote' in stock_data:
                current_price = stock_data['quote']['last_price']
                portfolio_df.at[i, 'current_price'] = round(current_price, 2)  # Round to 2 decimal places
                portfolio_df.at[i, 'value'] = round(row['quantity'] * current_price, 2)
                portfolio_df.at[i, 'profit'] = round((current_price - row['avg_price']) * row['quantity'], 2)
                portfolio_df.at[i, 'profit_pct'] = round((current_price - row['avg_price']) / row['avg_price'] * 100, 2)
            else:
                portfolio_df.at[i, 'current_price'] = 0
                portfolio_df.at[i, 'value'] = 0
                portfolio_df.at[i, 'profit'] = 0
                portfolio_df.at[i, 'profit_pct'] = 0
        
        # Display portfolio
        st.subheader("Current Holdings")
        st.dataframe(
            portfolio_df[['symbol', 'quantity', 'avg_price', 'current_price', 'value', 'profit', 'profit_pct']],
            use_container_width=True,
            column_config={
                "symbol": "Stock",
                "quantity": "Quantity",
                "avg_price": st.column_config.NumberColumn("Avg. Price", format="₹%.2f"),
                "current_price": st.column_config.NumberColumn("Current Price", format="₹%.2f"),
                "value": st.column_config.NumberColumn("Value", format="₹%.2f"),
                "profit": st.column_config.NumberColumn("Profit/Loss", format="₹%.2f"),
                "profit_pct": st.column_config.NumberColumn("Profit/Loss %", format="%.2f%%")
            }
        )
        
        # Portfolio summary
        total_investment = (portfolio_df['avg_price'] * portfolio_df['quantity']).sum()
        current_value = portfolio_df['value'].sum()
        total_profit = portfolio_df['profit'].sum()
        total_profit_pct = (total_profit / total_investment) * 100 if total_investment > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Investment", f"₹{total_investment:.2f}")
        col2.metric("Current Value", f"₹{current_value:.2f}")
        col3.metric("Profit/Loss", f"₹{total_profit:.2f}", f"{total_profit_pct:.2f}%")
        col4.metric("No. of Stocks", len(portfolio_df))
        
        # Strategy Analysis Summary
        st.subheader("Strategy Analysis Summary")
        
        # Get Nifty data for relative strength
        nifty_data = get_nifty_data()
        
        # Analyze each stock
        analysis_results = []
        buy_count = 0
        sell_count = 0
        hold_count = 0
        
        for i, row in portfolio_df.iterrows():
            stock_data = get_stock_data(row['symbol'])
            if stock_data and 'history' in stock_data and stock_data['history'] is not None and not stock_data['history'].empty:
                # Analyze strategies
                goverdhan = analyze_goverdhan_strategy(stock_data['history'])
                oliver = analyze_oliver_kell_strategy(stock_data['history'])
                relative_strength = analyze_relative_strength(stock_data['history'], nifty_data)
                garp = analyze_garp_strategy(stock_data['info'])
                
                # Generate recommendation
                recommendation = generate_recommendation(goverdhan, oliver, relative_strength, garp)
                
                if recommendation == "buy":
                    buy_count += 1
                elif recommendation == "sell":
                    sell_count += 1
                else:
                    hold_count += 1
                
                analysis_results.append({
                    'symbol': row['symbol'],
                    'goverdhan': goverdhan['signal'],
                    'oliver': oliver['signal'],
                    'relative_strength': relative_strength['signal'],
                    'garp': garp['signal'],
                    'recommendation': recommendation
                })
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Buy Signals", buy_count)
        col2.metric("Sell Signals", sell_count)
        col3.metric("Hold Signals", hold_count)
        
        # Display detailed analysis
        if analysis_results:
            st.subheader("Detailed Strategy Analysis")
            analysis_df = pd.DataFrame(analysis_results)
            
            st.dataframe(
                analysis_df,
                use_container_width=True,
                column_config={
                    "symbol": "Stock",
                    "goverdhan": "Goverdhan",
                    "oliver": "Oliver Kell",
                    "relative_strength": "Relative Strength",
                    "garp": "GARP",
                    "recommendation": "Recommendation"
                }
            )

# Add Stock Page
elif page == "Add Stock":
    st.title("Add Stock to Portfolio")
    
    # Tab for single stock vs multiple stocks
    tab1, tab2 = st.tabs(["Add Single Stock", "Add Multiple Stocks"])
    
    with tab1:
        # Get stock list
        stock_list = get_stock_list()
        
        # Search box
        search_term = st.text_input("Search for a stock", "")
        
        # Filter stocks based on search
        filtered_stocks = [stock for stock in stock_list if search_term.lower() in stock['name'].lower() or search_term.upper() in stock['symbol']]
        
        # Display filtered stocks
        if filtered_stocks:
            st.write(f"Found {len(filtered_stocks)} stocks:")
            
            # Create a dataframe for display
            stock_df = pd.DataFrame(filtered_stocks)
            st.dataframe(
                stock_df[['symbol', 'name', 'exchange']],
                use_container_width=True,
                hide_index=True
            )
            
            # Select stock
            selected_stock = st.selectbox("Select a stock to add", [stock['symbol'] for stock in filtered_stocks])
            
            # Get stock details
            stock_data = get_stock_data(selected_stock)
            if stock_data and 'quote' in stock_data and 'history' in stock_data and stock_data['history'] is not None and not stock_data['history'].empty:
                current_price = stock_data['quote']['last_price']
                
                # Display stock chart
                st.subheader(f"{selected_stock} - Price Chart")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=stock_data['history'].index,
                    open=stock_data['history']['Open'],
                    high=stock_data['history']['High'],
                    low=stock_data['history']['Low'],
                    close=stock_data['history']['Close'],
                    name=selected_stock
                ))
                fig.update_layout(
                    title=f"{selected_stock} Price History",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add stock form
                st.subheader("Add to Portfolio")
                with st.form("add_stock_form"):
                    quantity = st.number_input("Quantity", min_value=1, value=1)
                    # Format the current price to 2 decimal places for the input field
                    price = st.number_input("Price (₹)", min_value=0.01, value=float(round(current_price, 2)), format="%.2f")
                    purchase_date = st.date_input("Purchase Date", value=datetime.now().date())
                    submit_button = st.form_submit_button("Add to Portfolio")
                    
                    if submit_button:
                        add_to_portfolio(selected_stock, quantity, price, purchase_date.strftime('%Y-%m-%d'))
                        st.success(f"Added {quantity} shares of {selected_stock} at ₹{price:.2f} to your portfolio.")
            else:
                st.error("Could not fetch stock data. Please try another stock.")
        else:
            # If no stocks found in predefined list, try to get from Yahoo Finance
            if search_term:
                st.write(f"Stock '{search_term}' not found in predefined list. Trying to fetch from Yahoo Finance...")
                
                # Try to get stock data from Yahoo Finance
                stock_data = get_stock_data_yahoo(search_term)
                
                if stock_data and 'history' in stock_data and stock_data['history'] is not None and not stock_data['history'].empty:
                    st.success(f"Found data for {search_term} from Yahoo Finance")
                    
                    # Display stock chart
                    st.subheader(f"{search_term} - Price Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=stock_data['history'].index,
                        open=stock_data['history']['Open'],
                        high=stock_data['history']['High'],
                        low=stock_data['history']['Low'],
                        close=stock_data['history']['Close'],
                        name=search_term
                    ))
                    fig.update_layout(
                        title=f"{search_term} Price History",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add stock form
                    st.subheader("Add to Portfolio")
                    with st.form("add_stock_form_yahoo"):
                        quantity = st.number_input("Quantity", min_value=1, value=1)
                        # Format the current price to 2 decimal places for the input field
                        price = st.number_input("Price (₹)", min_value=0.01, value=float(round(stock_data['quote']['last_price'], 2)), format="%.2f")
                        purchase_date = st.date_input("Purchase Date", value=datetime.now().date())
                        submit_button = st.form_submit_button("Add to Portfolio")
                        
                        if submit_button:
                            add_to_portfolio(search_term, quantity, price, purchase_date.strftime('%Y-%m-%d'))
                            st.success(f"Added {quantity} shares of {search_term} at ₹{price:.2f} to your portfolio.")
                else:
                    st.error(f"Could not fetch data for '{search_term}'. Please check the stock symbol and try again.")
            else:
                st.info("No stocks found. Try a different search term.")
    
    with tab2:
        st.subheader("Add Multiple Stocks")
        st.write("Paste multiple stock symbols (one per line) and we'll fetch their data:")
        
        # Text area for multiple stock symbols
        stock_symbols = st.text_area("Stock Symbols", height=150, 
                                     help="Enter one stock symbol per line (e.g., RELIANCE, TCS, HDFCBANK)")
        
        if stock_symbols:
            # Parse symbols
            symbols = [symbol.strip().upper() for symbol in stock_symbols.split('\n') if symbol.strip()]
            
            if symbols:
                st.write(f"Found {len(symbols)} stock symbols:")
                
                # Create a dataframe to store stock data
                stocks_data = []
                
                # Fetch data for each symbol with delay to avoid rate limiting
                for i, symbol in enumerate(symbols):
                    with st.spinner(f"Fetching data for {symbol}..."):
                        # Add delay between requests to avoid rate limiting
                        if i > 0:
                            time.sleep(1)  # Wait 1 second between requests
                            
                        stock_data = get_stock_data(symbol)
                        
                        if stock_data and 'quote' in stock_data and 'history' in stock_data and stock_data['history'] is not None and not stock_data['history'].empty:
                            current_price = stock_data['quote']['last_price']
                            stocks_data.append({
                                'symbol': symbol,
                                'current_price': round(current_price, 2),
                                'data': stock_data
                            })
                        else:
                            st.warning(f"Could not fetch data for {symbol}")
                
                if stocks_data:
                    # Display stocks in a dataframe
                    stocks_df = pd.DataFrame(stocks_data)
                    st.dataframe(
                        stocks_df[['symbol', 'current_price']],
                        use_container_width=True,
                        column_config={
                            "symbol": "Stock Symbol",
                            "current_price": st.column_config.NumberColumn("Current Price (₹)", format="₹%.2f")
                        }
                    )
                    
                    # Add stocks form
                    st.subheader("Add to Portfolio")
                    with st.form("add_multiple_stocks_form"):
                        # Default purchase date is today
                        purchase_date = st.date_input("Purchase Date", value=datetime.now().date())
                        
                        st.write("Enter quantity for each stock:")
                        
                        # Create input fields for each stock
                        quantities = {}
                        for stock in stocks_data:
                            quantities[stock['symbol']] = st.number_input(
                                f"Quantity for {stock['symbol']} (Current Price: ₹{stock['current_price']})",
                                min_value=1, value=1, key=f"qty_{stock['symbol']}"
                            )
                        
                        submit_button = st.form_submit_button("Add All Stocks to Portfolio")
                        
                        if submit_button:
                            success_count = 0
                            for stock in stocks_data:
                                symbol = stock['symbol']
                                quantity = quantities[symbol]
                                price = stock['current_price']
                                
                                add_to_portfolio(symbol, quantity, price, purchase_date.strftime('%Y-%m-%d'))
                                success_count += 1
                            
                            st.success(f"Successfully added {success_count} stocks to your portfolio!")
                else:
                    st.error("Could not fetch data for any of the provided stock symbols.")

# Sell Stock Page (NEW)
elif page == "Sell Stock":
    st.title("Sell Stock from Portfolio")
    
    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add stocks using the 'Add Stock' page.")
    else:
        # Create a dataframe for portfolio
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        
        # Get current prices
        for i, row in portfolio_df.iterrows():
            stock_data = get_stock_data(row['symbol'])
            if stock_data and 'quote' in stock_data:
                current_price = stock_data['quote']['last_price']
                portfolio_df.at[i, 'current_price'] = round(current_price, 2)
                portfolio_df.at[i, 'value'] = round(row['quantity'] * current_price, 2)
                portfolio_df.at[i, 'profit'] = round((current_price - row['avg_price']) * row['quantity'], 2)
                portfolio_df.at[i, 'profit_pct'] = round((current_price - row['avg_price']) / row['avg_price'] * 100, 2)
            else:
                portfolio_df.at[i, 'current_price'] = 0
                portfolio_df.at[i, 'value'] = 0
                portfolio_df.at[i, 'profit'] = 0
                portfolio_df.at[i, 'profit_pct'] = 0
        
        # Select stock to sell
        selected_stock = st.selectbox("Select a stock to sell", portfolio_df['symbol'])
        
        # Get selected stock data
        stock_data = get_stock_data(selected_stock)
        if stock_data and 'quote' in stock_data and 'history' in stock_data and stock_data['history'] is not None and not stock_data['history'].empty:
            current_price = stock_data['quote']['last_price']
            
            # Get stock details
            stock_row = portfolio_df[portfolio_df['symbol'] == selected_stock].iloc[0]
            
            # Display stock chart
            st.subheader(f"{selected_stock} - Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_data['history'].index,
                open=stock_data['history']['Open'],
                high=stock_data['history']['High'],
                low=stock_data['history']['Low'],
                close=stock_data['history']['Close'],
                name=selected_stock
            ))
            fig.update_layout(
                title=f"{selected_stock} Price History",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display stock details
            st.subheader(f"Current Holdings: {selected_stock}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Quantity", stock_row['quantity'])
            col2.metric("Avg. Price", f"₹{stock_row['avg_price']:.2f}")
            col3.metric("Current Price", f"₹{current_price:.2f}")
            
            # Sell stock form
            st.subheader("Sell Stock")
            with st.form("sell_stock_form"):
                quantity = st.number_input("Quantity to sell", min_value=1, max_value=int(stock_row['quantity']), value=1)
                # Format the current price to 2 decimal places for the input field
                price = st.number_input("Price (₹)", min_value=0.01, value=float(round(current_price, 2)), format="%.2f")
                submit_button = st.form_submit_button("Sell Stock")
                
                if submit_button:
                    profit = sell_from_portfolio(selected_stock, quantity, price)
                    if profit != 0:
                        if profit > 0:
                            st.success(f"Sold {quantity} shares of {selected_stock} at ₹{price:.2f}. Profit: ₹{profit:.2f}")
                        else:
                            st.success(f"Sold {quantity} shares of {selected_stock} at ₹{price:.2f}. Loss: ₹{abs(profit):.2f}")
                    else:
                        st.error("Failed to sell stock. Please try again.")
        else:
            st.error("Could not fetch stock data. Please try again later.")

# Analysis Page
elif page == "Analysis":
    st.title("Strategy Analysis")
    
    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add stocks using the 'Add Stock' page.")
    else:
        # Get Nifty data for relative strength
        nifty_data = get_nifty_data()
        
        # Select stock for detailed analysis
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        selected_stock = st.selectbox("Select a stock for detailed analysis", portfolio_df['symbol'])
        
        # Get stock data
        stock_data = get_stock_data(selected_stock)
        if stock_data and 'history' in stock_data and stock_data['history'] is not None and not stock_data['history'].empty:
            # Analyze strategies
            goverdhan = analyze_goverdhan_strategy(stock_data['history'])
            oliver = analyze_oliver_kell_strategy(stock_data['history'])
            relative_strength = analyze_relative_strength(stock_data['history'], nifty_data)
            garp = analyze_garp_strategy(stock_data['info'])
            
            # Generate recommendation
            recommendation = generate_recommendation(goverdhan, oliver, relative_strength, garp)
            
            # Display recommendation
            st.subheader(f"Investment Recommendation: {recommendation.upper()}")
            
            # Display strategy results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Goverdhan Strategy")
                st.write(f"Signal: {goverdhan['signal'].upper()}")
                st.write(f"Pattern: {goverdhan['pattern']}")
                st.write(f"Confidence: {goverdhan['confidence']}%")
                
                st.subheader("Oliver Kell Strategy")
                st.write(f"Signal: {oliver['signal'].upper()}")
                st.write(f"Pattern: {oliver['pattern']}")
                st.write(f"Confidence: {oliver['confidence']}%")
            
            with col2:
                st.subheader("Relative Strength")
                st.write(f"Signal: {relative_strength['signal'].upper()}")
                st.write(f"Value: {relative_strength['value']:.4f}")
                st.write(f"Trend: {relative_strength['trend']}")
                
                st.subheader("GARP Strategy")
                st.write(f"Signal: {garp['signal'].upper()}")
                st.write(f"Score: {garp['score']}/7")
                
                with st.expander("GARP Criteria Details"):
                    for criterion, met in garp['criteria'].items():
                        st.write(f"{'✅' if met else '❌'} {criterion}")
            
            # Display stock chart with indicators
            st.subheader(f"{selected_stock} - Technical Analysis")
            
            # Prepare data for chart
            chart_data = stock_data['history'].copy()
            chart_data['EMA9'] = calculate_ema(chart_data, 9)
            chart_data['EMA21'] = calculate_ema(chart_data, 21)
            chart_data['EMA50'] = calculate_ema(chart_data, 50)
            
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name="Price"
            ))
            
            # EMAs
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['EMA9'],
                mode='lines',
                name='EMA 9',
                line=dict(width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['EMA21'],
                mode='lines',
                name='EMA 21',
                line=dict(width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['EMA50'],
                mode='lines',
                name='EMA 50',
                line=dict(width=1)
            ))
            
            fig.update_layout(
                title=f"{selected_stock} Technical Analysis",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Action buttons
            st.subheader("Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Buy More"):
                    st.info("This would open a form to buy more shares of this stock.")
            
            with col2:
                if st.button("Sell Some"):
                    st.info("This would open a form to sell some shares of this stock.")
            
            with col3:
                if st.button("Sell All"):
                    st.info("This would sell all shares of this stock.")
        else:
            st.error("Could not fetch stock data. Please try again later.")

# Transactions Page
elif page == "Transactions":
    st.title("Transaction History")
    
    if not st.session_state.transactions:
        st.info("No transactions recorded yet.")
    else:
        # Display transactions with delete buttons
        st.subheader("Your Transactions")
        
        # Add headers
        header_col1, header_col2, header_col3, header_col4, header_col5, header_col6, header_col7 = st.columns([1, 1, 1, 1, 1, 1, 1])
        
        with header_col1:
            st.markdown("**Stock**")
        
        with header_col2:
            st.markdown("**Quantity**")
        
        with header_col3:
            st.markdown("**Buy Price**")
        
        with header_col4:
            st.markdown("**Sell Price**")
        
        with header_col5:
            st.markdown("**Profit/Loss**")
        
        with header_col6:
            st.markdown("**Date**")
        
        with header_col7:
            st.markdown("**Action**")
        
        # Add a divider
        st.divider()
        
        # Display each transaction
        for i, transaction in enumerate(st.session_state.transactions):
            col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])
            
            with col1:
                st.write(transaction['symbol'])
            
            with col2:
                st.write(transaction['quantity'])
            
            with col3:
                st.write(f"₹{transaction['buy_price']:.2f}")
            
            with col4:
                st.write(f"₹{transaction['sell_price']:.2f}")
            
            with col5:
                profit_color = "green" if transaction['profit'] >= 0 else "red"
                st.markdown(f"<span style='color:{profit_color}'>₹{transaction['profit']:.2f}</span>", unsafe_allow_html=True)
            
            with col6:
                st.write(transaction['date'])
            
            with col7:
                if st.button("Delete", key=f"del_{i}"):
                    st.session_state.transactions.pop(i)
                    save_transactions_data()
                    st.success("Transaction deleted successfully!")
                    st.rerun()
        
        # Transaction summary
        total_profit = sum(t['profit'] for t in st.session_state.transactions)
        profitable_transactions = sum(1 for t in st.session_state.transactions if t['profit'] > 0)
        loss_transactions = sum(1 for t in st.session_state.transactions if t['profit'] <= 0)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Profit/Loss", f"₹{total_profit:.2f}")
        col2.metric("Profitable Trades", profitable_transactions)
        col3.metric("Loss Trades", loss_transactions)
        
        # Profit/Loss chart
        st.subheader("Profit/Loss History")
        fig = px.bar(
            pd.DataFrame(st.session_state.transactions),
            x='date',
            y='profit',
            color='profit',
            color_continuous_scale=['red', 'green'],
            title="Profit/Loss per Transaction"
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made By Ashutosh© 2025 Indian Stock Portfolio Manager")
