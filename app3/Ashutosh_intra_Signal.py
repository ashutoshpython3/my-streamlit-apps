import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta, time
import pytz
import os
import argparse
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import time as tm
import warnings
import pygame
import threading
import yfinance as yf
from bs4 import BeautifulSoup
import re
from io import StringIO
import base64
from concurrent.futures import ThreadPoolExecutor
import websocket
import threading

warnings.filterwarnings('ignore', category=FutureWarning)

# Initialize pygame mixer for sound with error handling
try:
    pygame.mixer.init()
    SOUND_AVAILABLE = True
except:
    SOUND_AVAILABLE = False
    print("Warning: Audio device not available. Sound alerts will be disabled.")

# Sound alert function with error handling - updated to use pygame
def play_alert_sound(alert_type="alert"):
    if not SOUND_AVAILABLE:
        return
    try:
        if alert_type == "alert":
            try:
                root_dir = os.path.dirname(os.path.abspath(__file__))
                alert_path = os.path.join(root_dir, "assets", "alert.mp3")
                pygame.mixer.music.load(alert_path)
                pygame.mixer.music.play()
            except:
                pass
        elif alert_type == "warning":
            try:
                root_dir = os.path.dirname(os.path.abspath(__file__))
                warning_path = os.path.join(root_dir, "assets", "warning.mp3")
                pygame.mixer.music.load(warning_path)
                pygame.mixer.music.play()
            except:
                pass
    except Exception as e:
        pass

# Upstox API base URL
BASE_URL = "https://api.upstox.com/v2"
# Integrated API credentials (replace with your actual credentials)
API_KEY = "9a56569a-142f-4247-a863-f4e663fb03f1"
ACCESS_TOKEN = "smgyr2big7"

# Map symbols to Upstox instrument keys
instrument_map = {
    '^NSEI': 'NSE_INDEX|Nifty 50',
    '^BSESN': 'BSE_INDEX|Sensex',
    '^INDIAVIX': 'NSE_INDEX|India VIX'
}

# Map stock symbols to Upstox instrument keys format
def get_stock_instrument_key(symbol):
    return f"NSE_EQ|{symbol.upper()}"

# Convert 1-minute data to desired timeframe (5 or 15 minutes)
def convert_timeframe(df, timeframe_minutes=15):
    if df.empty:
        return df
    df = df.copy()
    df['time_group'] = df.index.floor(f'{timeframe_minutes}min')
    resampled = df.groupby('time_group').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'OI': 'last',
        'VIX': 'last'
    })
    return resampled

# Generate sample data for testing when API is not available
def generate_sample_data(symbol='^NSEI', days=30):
    st.info(f"Generating sample data for {symbol} for testing purposes...")
    end_date = datetime.now(pytz.timezone('Asia/Kolkata'))
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    np.random.seed(42)
    base_price = 19500
    daily_data = []
    for date in dates:
        change = np.random.normal(0, 0.01)
        open_price = base_price * (1 + change * 0.5)
        close_price = base_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = int(np.random.normal(100000, 20000))
        daily_data.append({
            'timestamp': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'oi': np.random.randint(10000, 50000)
        })
        base_price = close_price
    df = pd.DataFrame(daily_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
    df['VIX'] = np.random.uniform(12, 18, size=len(df))
    return df

# Generate sample intraday data
def generate_sample_intraday_data(symbol='^NSEI', timeframe_minutes=15, days=15):
    st.info(f"Generating sample intraday data for {symbol} for testing purposes...")
    end_date = datetime.now(pytz.timezone('Asia/Kolkata'))
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    all_data = []
    for date in dates:
        start_time = datetime.combine(date, time(9, 15)).astimezone(pytz.timezone('Asia/Kolkata'))
        end_time = datetime.combine(date, time(15, 30)).astimezone(pytz.timezone('Asia/Kolkata'))
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
        np.random.seed(42)
        base_price = 19500
        intraday_data = []
        for timestamp in timestamps:
            change = np.random.normal(0, 0.0005)
            open_price = base_price * (1 + change * 0.5)
            close_price = base_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0002)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0002)))
            volume = int(np.random.normal(5000, 1000))
            intraday_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'oi': np.random.randint(1000, 5000)
            })
            base_price = close_price
        all_data.extend(intraday_data)
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
    df['VIX'] = 15.0
    df = convert_timeframe(df, timeframe_minutes=timeframe_minutes)
    df = df.between_time('09:15', '15:30')
    return df

# Fetch Historical Daily Data
def fetch_historical_data(index_symbols=['^NSEI', '^BSESN'], api_key=None, access_token=None):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365)
    historical_data = {}
    if not api_key:
        api_key = API_KEY
    if not access_token:
        access_token = ACCESS_TOKEN
    if not api_key or not access_token:
        st.error("API key and access token are required")
        return historical_data
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}',
        'x-api-key': api_key
    }
    for symbol in index_symbols:
        try:
            st.write(f"Fetching historical data for {symbol}")
            instrument_key = instrument_map.get(symbol)
            if not instrument_key:
                st.error(f"No instrument key found for {symbol}")
                continue
            to_date = end_date.strftime('%Y-%m-%d')
            from_date = start_date.strftime('%Y-%m-%d')
            url = f"{BASE_URL}/historical-candle/{instrument_key}/day/{to_date}/{from_date}"
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    st.error(f"API returned status code {response.status_code} for {symbol}")
                    df = generate_sample_data(symbol)
                    historical_data[symbol] = df
                    continue
                data = response.json()
                if not data or 'data' not in data or 'candles' not in data['data']:
                    st.error(f"No data returned for {symbol}")
                    df = generate_sample_data(symbol)
                    historical_data[symbol] = df
                    continue
                candles = data['data']['candles']
                if not candles:
                    st.error(f"No candles returned for {symbol}")
                    df = generate_sample_data(symbol)
                    historical_data[symbol] = df
                    continue
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('Asia/Kolkata')
                else:
                    df.index = df.index.tz_convert('Asia/Kolkata')
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
                vix_instrument = instrument_map.get('^INDIAVIX')
                if vix_instrument:
                    vix_url = f"{BASE_URL}/historical-candle/{vix_instrument}/day/{to_date}/{from_date}"
                    try:
                        vix_response = requests.get(vix_url, headers=headers, timeout=10)
                        if vix_response.status_code == 200:
                            vix_data = vix_response.json()
                            if vix_data and 'data' in vix_data and 'candles' in vix_data['data'] and vix_data['data']['candles']:
                                vix_df = pd.DataFrame(vix_data['data']['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                                vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp'])
                                vix_df.set_index('timestamp', inplace=True)
                                if vix_df.index.tz is None:
                                    vix_df.index = vix_df.index.tz_localize('Asia/Kolkata')
                                else:
                                    vix_df.index = vix_df.index.tz_convert('Asia/Kolkata')
                                df['VIX'] = vix_df['close'].reindex(df.index, method='ffill')
                            else:
                                df['VIX'] = 15.0
                        else:
                            df['VIX'] = 15.0
                    except:
                        df['VIX'] = 15.0
                else:
                    df['VIX'] = 15.0
                df = df.dropna(subset=['VIX'])
                if df.empty:
                    st.warning(f"No data after VIX alignment for {symbol}. Trying without alignment.")
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('Asia/Kolkata')
                    else:
                        df.index = df.index.tz_convert('Asia/Kolkata')
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
                    df['VIX'] = 15.0
                df_pre_covid = df[(df.index < '2020-01-01') | (df.index > '2021-12-31')]
                if not df_pre_covid.empty:
                    df = df_pre_covid
                else:
                    st.warning(f"No data after COVID exclusion for {symbol}. Using full dataset.")
                st.success(f"Fetched {len(df)} rows of historical data for {symbol}")
                historical_data[symbol] = df
            except requests.exceptions.ConnectionError:
                st.error(f"Connection error when fetching data for {symbol}. Using sample data as fallback...")
                df = generate_sample_data(symbol)
                historical_data[symbol] = df
                continue
        except Exception as e:
            st.error(f"Error fetching historical data for {symbol}: {e}")
            st.info("Using sample data as fallback...")
            df = generate_sample_data(symbol)
            historical_data[symbol] = df
            continue
    return historical_data

# Fetch Intraday Data using Upstox API - updated to handle empty responses
def fetch_intraday_data(index_symbols=['^NSEI', '^BSESN'], sim_date=None, api_key=None, access_token=None, timeframe_minutes=15, days=15):
    ist = pytz.timezone('Asia/Kolkata')
    intraday_data = {}
    if not api_key:
        api_key = API_KEY
    if not access_token:
        access_token = ACCESS_TOKEN
    if not api_key or not access_token:
        st.error("API key and access token are required")
        return intraday_data
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}',
        'x-api-key': api_key
    }
    for symbol in index_symbols:
        try:
            st.write(f"Fetching intraday data for {symbol} (last {days} days)")
            if symbol.startswith('^'):
                instrument_key = instrument_map.get(symbol)
                if not instrument_key:
                    st.error(f"No instrument key found for {symbol}")
                    continue
            else:
                instrument_key = get_stock_instrument_key(symbol)
            if sim_date:
                sim_date_dt = datetime.strptime(sim_date.strip(), '%Y-%m-%d')
                if sim_date_dt.weekday() >= 5:
                    st.error(f"{sim_date} is a weekend (not a trading day).")
                    continue
                if sim_date_dt.date() > datetime.now().date():
                    st.error(f"{sim_date} is a future date, no data available.")
                    continue
                start_date = sim_date_dt - timedelta(days=days)
                while start_date.weekday() >= 5:
                    start_date = start_date - timedelta(days=1)
                start = start_date.replace(hour=9, minute=15).astimezone(ist)
                end = sim_date_dt.replace(hour=15, minute=30).astimezone(ist)
            else:
                end_date = datetime.now().astimezone(ist)
                start_date = end_date - timedelta(days=days)
                while start_date.weekday() >= 5:
                    start_date = start_date - timedelta(days=1)
                start = start_date.replace(hour=9, minute=15).astimezone(ist)
                end = end_date
            to_date = end.strftime('%Y-%m-%d')
            from_date = start.strftime('%Y-%m-%d')
            url = f"{BASE_URL}/historical-candle/intraday/{instrument_key}/1minute"
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    st.error(f"API returned status code {response.status_code} for {symbol}")
                    df = generate_sample_intraday_data(symbol, timeframe_minutes, days)
                    intraday_data[symbol] = df
                    continue
                data = response.json()
                if not data or 'data' not in data or 'candles' not in data['data']:
                    st.error(f"No data returned for {symbol}")
                    df = generate_sample_intraday_data(symbol, timeframe_minutes, days)
                    intraday_data[symbol] = df
                    continue
                candles = data['data']['candles']
                if not candles:
                    st.error(f"No candles returned for {symbol}")
                    df = generate_sample_intraday_data(symbol, timeframe_minutes, days)
                    intraday_data[symbol] = df
                    continue
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('Asia/Kolkata')
                else:
                    df.index = df.index.tz_convert('Asia/Kolkata')
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
                df = df.between_time('09:15', '15:30')
                df['VIX'] = 15.0
                df = convert_timeframe(df, timeframe_minutes=timeframe_minutes)
                df = df.between_time('09:15', '15:30')
                if sim_date:
                    sim_date_str = sim_date.strip()
                    df = df[df.index.date == datetime.strptime(sim_date_str, '%Y-%m-%d').date()]
                if df.empty:
                    st.warning(f"No data returned after processing for {symbol}. Using sample data as fallback.")
                    df = generate_sample_intraday_data(symbol, timeframe_minutes, days)
                st.success(f"Successfully fetched and converted {len(df)} rows of {timeframe_minutes}-minute intraday data for {symbol}")
                intraday_data[symbol] = df
            except requests.exceptions.ConnectionError:
                st.error(f"Connection error when fetching intraday data for {symbol}. Using sample data as fallback...")
                df = generate_sample_intraday_data(symbol, timeframe_minutes, days)
                intraday_data[symbol] = df
                continue
        except Exception as e:
            st.error(f"Error fetching intraday data for {symbol}: {e}")
            st.info("Using sample data as fallback...")
            df = generate_sample_intraday_data(symbol, timeframe_minutes, days)
            intraday_data[symbol] = df
            continue
    return intraday_data

# Fetch stock data from Yahoo Finance - updated to handle API limitations
def fetch_stock_data(symbol, period='7d', interval='5m'):
    try:
        if not symbol.endswith(('.NS', '.BO')):
            symbol = symbol + '.NS'
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            st.warning(f"No data found for {symbol}")
            return None
        data.reset_index(inplace=True)
        data['Datetime'] = pd.to_datetime(data['Datetime']).dt.tz_convert('Asia/Kolkata')
        data.set_index('Datetime', inplace=True)
        data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
        data['OI'] = 0
        data['VIX'] = 15.0
        return data
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {e}")
        return None

# Compute Indicators - updated to handle small datasets
def compute_indicators(df, intraday=True):
    if df.empty or 'Close' not in df.columns:
        st.error("DataFrame is empty or missing 'Close' column.")
        return df
    available_rows = len(df)
    if intraday:
        period_adx = min(5, available_rows - 1) if available_rows > 1 else 1
        period_ema_short = min(10, available_rows - 1) if available_rows > 1 else 1
        period_ema_long = min(20, available_rows - 1) if available_rows > 1 else 1
        period_bb = min(10, available_rows - 1) if available_rows > 1 else 1
        period_ema9 = min(9, available_rows - 1) if available_rows > 1 else 1
        period_ema21 = min(21, available_rows - 1) if available_rows > 1 else 1
    else:
        period_adx = min(14, available_rows - 1) if available_rows > 1 else 1
        period_ema_short = min(20, available_rows - 1) if available_rows > 1 else 1
        period_ema_long = min(50, available_rows - 1) if available_rows > 1 else 1
        period_bb = min(20, available_rows - 1) if available_rows > 1 else 1
        period_ema9 = min(9, available_rows - 1) if available_rows > 1 else 1
        period_ema21 = min(21, available_rows - 1) if available_rows > 1 else 1
    df = df.copy()
    try:
        if period_ema_short > 0:
            df['EMA10'] = EMAIndicator(df['Close'], window=period_ema_short).ema_indicator()
        else:
            df['EMA10'] = np.nan
        if period_ema_long > 0:
            df['EMA20'] = EMAIndicator(df['Close'], window=period_ema_long).ema_indicator()
        else:
            df['EMA20'] = np.nan
        if period_adx > 0:
            df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close'], window=period_adx).adx()
        else:
            df['ADX'] = np.nan
        if period_adx > 0:
            df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=period_adx).average_true_range()
        else:
            df['ATR'] = np.nan
        df['VOL_AVG'] = df['Volume'].rolling(window=10).mean()
        if period_ema9 > 0:
            df['EMA9'] = EMAIndicator(df['Close'], window=period_ema9).ema_indicator()
        else:
            df['EMA9'] = np.nan
        if period_ema21 > 0:
            df['EMA21'] = EMAIndicator(df['Close'], window=period_ema21).ema_indicator()
        else:
            df['EMA21'] = np.nan
        if period_bb > 0:
            bb = BollingerBands(df['Close'], window=period_bb, window_dev=2)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Mid'] = bb.bollinger_mavg()
        else:
            df['BB_Upper'] = np.nan
            df['BB_Lower'] = np.nan
            df['BB_Mid'] = np.nan
        if available_rows > 0:
            df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        else:
            df['VWAP'] = np.nan
    except Exception as e:
        st.error(f"Error computing indicators: {e}")
        return df
    return df

# Compute Oliver Kell Phases
def compute_kell_phases(df):
    if df.empty:
        return df
    df = df.copy()
    df['PHASE'] = 'None'
    df['EXHAUSTION'] = False
    df['WEDGE_POP'] = False
    df['WEDGE_DROP'] = False
    df['BASE_BREAK'] = False
    df['EMA_CROSSBACK'] = False
    df['REVERSAL_EXT'] = False
    for i in range(10, len(df)):
        close = df['Close'].iloc[i]
        prev_close = df['Close'].iloc[i-1]
        ema10 = df['EMA10'].iloc[i]
        ema20 = df['EMA20'].iloc[i]
        atr = df['ATR'].iloc[i]
        vol = df['Volume'].iloc[i]
        vol_avg = df['VOL_AVG'].iloc[i]
        prev_low = df['Low'].iloc[i-5:i].min()
        prev_high = df['High'].iloc[i-5:i].max()
        price_std = df['Close'].iloc[i-5:i].std()
        if abs(close - prev_close) > 2 * atr and vol > 1.5 * vol_avg:
            df.loc[df.index[i], 'REVERSAL_EXT'] = True
            df.loc[df.index[i], 'PHASE'] = 'Reversal Extension'
        if price_std < atr / 2 and close > prev_high and ema10 > ema20:
            df.loc[df.index[i], 'WEDGE_POP'] = True
            df.loc[df.index[i], 'PHASE'] = 'Wedge Pop'
        if price_std < atr / 2 and close < prev_low and ema10 < ema20:
            df.loc[df.index[i], 'WEDGE_DROP'] = True
            df.loc[df.index[i], 'PHASE'] = 'Wedge Drop'
        if abs(close - ema10) < atr and vol > vol_avg:
            df.loc[df.index[i], 'EMA_CROSSBACK'] = True
            df.loc[df.index[i], 'PHASE'] = 'EMA Crossback'
        base_high = df['High'].iloc[i-10:i].max()
        base_low = df['Low'].iloc[i-10:i].min()
        if (base_high - base_low) / close < 0.02 and close > base_high:
            df.loc[df.index[i], 'BASE_BREAK'] = True
            df.loc[df.index[i], 'PHASE'] = 'Base n Break (Long)'
        elif (base_high - base_low) / close < 0.02 and close < base_low:
            df.loc[df.index[i], 'BASE_BREAK'] = True
            df.loc[df.index[i], 'PHASE'] = 'Base n Break (Short)'
        air_gap = (close - ema10) / atr
        if air_gap > 1.5 and df['High'].iloc[i] > df['High'].iloc[i-1]:
            df.loc[df.index[i], 'EXHAUSTION'] = True
            df.loc[df.index[i], 'PHASE'] = 'Exhaustion Extension'
    return df

# Compute Goverdhan Gajjala's Patterns
def compute_goverdhan_patterns(df):
    if df.empty:
        return df
    df = df.copy()
    df['BULL_FLAG'] = False
    df['EMA_KISS_FLY'] = False
    df['HORIZONTAL_FADE'] = False
    df['VCP'] = False
    df['REVERSAL_SQUEEZE'] = False
    for i in range(21, len(df)):
        close = df['Close'].iloc[i]
        open_price = df['Open'].iloc[i]
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]
        volume = df['Volume'].iloc[i]
        ema9 = df['EMA9'].iloc[i]
        ema21 = df['EMA21'].iloc[i]
        atr = df['ATR'].iloc[i]
        vol_avg = df['VOL_AVG'].iloc[i]
        prev_close = df['Close'].iloc[i-1]
        spike_high = df['High'].iloc[i-5:i-3].max()
        spike_vol = df['Volume'].iloc[i-5:i-3].max()
        if (spike_vol > 2 * vol_avg and
            spike_high > df['Close'].iloc[i-6] * 1.005 and
            close > ema9 and close > ema21 and
            df['Low'].iloc[i-3:i].min() > ema9 and
            high > spike_high and volume > 1.5 * vol_avg):
            df.loc[df.index[i], 'BULL_FLAG'] = True
        if (abs(close - ema9) < 0.3 * atr and
            prev_close < ema9 and
            close > open_price and
            volume > 1.2 * vol_avg):
            df.loc[df.index[i], 'EMA_KISS_FLY'] = True
        if (abs(close - ema9) < 0.5 * atr and
            df['High'].iloc[i-3:i].max() - df['Low'].iloc[i-3:i].min() < 1.5 * atr and
            close > ema9 and close > ema21):
            df.loc[df.index[i], 'HORIZONTAL_FADE'] = True
        if (df['High'].iloc[i-5:i].max() - df['Low'].iloc[i-5:i].min() < 2 * atr and
            df['Volume'].iloc[i-5:i].max() < 1.5 * vol_avg and
            close > ema21 and
            high > df['High'].iloc[i-5:i].max()):
            df.loc[df.index[i], 'VCP'] = True
        if (df['High'].iloc[i-5:i].max() - df['Low'].iloc[i-5:i].min() < 1.5 * atr and
            df['Close'].iloc[i-5] < df['Close'].iloc[i-10] and
            close > open_price and
            volume > 1.5 * vol_avg):
            df.loc[df.index[i], 'REVERSAL_SQUEEZE'] = True
    return df

# Compute Andrea Unger's Strategy
def compute_unger_strategy(df):
    if df.empty:
        return df
    df = df.copy()
    df['UNGER_SIGNAL'] = 'None'
    for i in range(20, len(df)):
        close = df['Close'].iloc[i]
        bb_upper = df['BB_Upper'].iloc[i]
        bb_lower = df['BB_Lower'].iloc[i]
        adx = df['ADX'].iloc[i]
        volume = df['Volume'].iloc[i]
        vol_avg = df['VOL_AVG'].iloc[i]
        if (close > bb_upper and adx > 25 and volume > 1.5 * vol_avg):
            df.loc[df.index[i], 'UNGER_SIGNAL'] = 'Buy (Breakout)'
        elif (close < bb_lower and adx > 25 and volume > 1.5 * vol_avg):
            df.loc[df.index[i], 'UNGER_SIGNAL'] = 'Sell (Breakdown)'
    return df

# Compute Michael Cook's Strategy
def compute_cook_strategy(df):
    if df.empty:
        return df
    df = df.copy()
    df['COOK_SIGNAL'] = 'None'
    for i in range(10, len(df)):
        close = df['Close'].iloc[i]
        vwap = df['VWAP'].iloc[i]
        ema9 = df['EMA9'].iloc[i]
        ema21 = df['EMA21'].iloc[i]
        volume = df['Volume'].iloc[i]
        vol_avg = df['VOL_AVG'].iloc[i]
        if (close > vwap and close > ema9 and close > ema21 and
            abs(close - vwap) < 0.5 * df['ATR'].iloc[i] and volume > 1.2 * vol_avg):
            df.loc[df.index[i], 'COOK_SIGNAL'] = 'Buy (VWAP Bounce)'
        elif (close < vwap and close < ema9 and close < ema21 and
              abs(close - vwap) < 0.5 * df['ATR'].iloc[i] and volume > 1.2 * vol_avg):
            df.loc[df.index[i], 'COOK_SIGNAL'] = 'Sell (VWAP Rejection)'
    return df

# Estimate Average Sideways Band Size - Fixed to handle missing columns
def estimate_band_size(df_daily):
    if df_daily.empty:
        return 6.5, 0
    if 'ADX' not in df_daily.columns:
        df_daily = compute_indicators(df_daily, intraday=False)
    sideways_periods = []
    min_window = 10
    i = 0
    while i < len(df_daily) - min_window:
        window = df_daily.iloc[i:i+min_window]
        if 'ADX' in window.columns and (window['ADX'] < 20).all():
            full_window = window
            j = i + min_window
            while j < len(df_daily) and 'ADX' in df_daily.columns and df_daily['ADX'].iloc[j] < 20:
                full_window = df_daily.iloc[i:j+1]
                j += 1
            band_pct = (full_window['High'].max() - full_window['Low'].min()) / full_window['Low'].min() * 100
            sideways_periods.append(band_pct)
            i = j
        else:
            i += 1
    if sideways_periods:
        mean_pct = np.mean(sideways_periods)
        std_pct = np.std(sideways_periods)
        sideways_periods = [p for p in sideways_periods if p <= mean_pct + 2*std_pct]
    avg_band_pct = np.mean(sideways_periods) if sideways_periods else 6.5
    hist_atr_avg = df_daily['ATR'].mean() if 'ATR' in df_daily.columns and not df_daily['ATR'].empty else 0
    return avg_band_pct, hist_atr_avg

# Fetch Option Chain
def fetch_option_chain(current_price, simulation=False, symbol='NIFTY', is_stock=False):
    if simulation:
        return {}, False, 0, 0, "Neutral"
    try:
        if is_stock:
            return {
                current_price * 0.98: {'CE_OI': 10000, 'PE_OI': 5000},
                current_price: {'CE_OI': 8000, 'PE_OI': 12000},
                current_price * 1.02: {'CE_OI': 5000, 'PE_OI': 10000}
            }, False, 23000, 27000, "Neutral"
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            strikes_oi = {}
            total_ce_oi = 0
            total_pe_oi = 0
            for record in data['records']['data']:
                strike = record['strikePrice']
                ce_oi = record.get('CE', {}).get('openInterest', 0)
                pe_oi = record.get('PE', {}).get('openInterest', 0)
                strikes_oi[strike] = {'CE_OI': ce_oi, 'PE_OI': pe_oi}
                total_ce_oi += ce_oi
                total_pe_oi += pe_oi
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1
            if pcr > 1.5:
                sentiment = "Bullish"
            elif pcr < 0.7:
                sentiment = "Bearish"
            else:
                sentiment = "Neutral"
            return strikes_oi, total_pe_oi > total_ce_oi, total_ce_oi, total_pe_oi, sentiment
    except Exception as e:
        st.error(f"Error fetching option chain for {symbol}: {e}")
    return {}, False, 0, 0, "Neutral"

# Detect OI Buildup
def detect_oi_buildup(current_strikes_oi, prev_strikes_oi, current_price, prev_price, vix, simulation=False):
    if simulation:
        return {'bullish': None, 'bearish': None}
    buildups = {'bullish': None, 'bearish': None}
    if not prev_strikes_oi:
        return buildups
    threshold_rel = 0.25 if vix <= 20 else 0.15
    threshold_abs = 50000
    min_distance_pct = 0.01 if vix <= 20 else 0.015
    for strike in current_strikes_oi:
        if strike in prev_strikes_oi:
            ce_curr = current_strikes_oi[strike]['CE_OI']
            pe_curr = current_strikes_oi[strike]['PE_OI']
            ce_prev = prev_strikes_oi[strike]['CE_OI']
            pe_prev = prev_strikes_oi[strike]['PE_OI']
            if strike > current_price * (1 + min_distance_pct):
                ce_inc_rel = (ce_curr - ce_prev) / ce_prev if ce_prev > 0 else float('inf')
                ce_inc_abs = ce_curr - ce_prev
                price_move_up = (current_price - prev_price) / prev_price
                if (ce_inc_rel > threshold_rel or ce_inc_abs > threshold_abs) and price_move_up < 0.002:
                    buildups['bullish'] = strike
            if strike < current_price * (1 - min_distance_pct):
                pe_inc_rel = (pe_curr - pe_prev) / pe_prev if pe_prev > 0 else float('inf')
                pe_inc_abs = pe_curr - pe_prev
                price_move_down = (prev_price - current_price) / prev_price
                if (pe_inc_rel > threshold_rel or pe_inc_abs > threshold_abs) and price_move_down < 0.002:
                    buildups['bearish'] = strike
    return buildups

# Generate Oliver Kell Signals
def generate_kell_signals(df, vix, total_ce, total_pe):
    latest = df.iloc[-1]
    kell_signal = "No Kell Signal"
    if latest['REVERSAL_EXT'] or latest['EMA_CROSSBACK']:
        if vix < 20 and total_ce > total_pe:
            kell_signal = "Oliver Kell: Buy CE (Reversal/Consolidation Entry)"
    elif latest['WEDGE_POP'] or (latest['BASE_BREAK'] and latest['Close'] > latest['EMA10']):
        kell_signal = "Oliver Kell: Buy CE (Breakout Long)"
    elif latest['WEDGE_DROP'] or (latest['BASE_BREAK'] and latest['Close'] < latest['EMA10']):
        kell_signal = "Oliver Kell: Buy PE (Breakdown Short)"
    elif latest['EXHAUSTION']:
        if vix > 15:
            kell_signal = "Oliver Kell: Buy PE (Sell into Strength - Exhaustion)"
    return kell_signal

# Generate Goverdhan Gajjala Signals
def generate_goverdhan_signals(df, vix, total_ce, total_pe):
    latest = df.iloc[-1]
    goverdhan_signal = "No Goverdhan Signal"
    if latest['BULL_FLAG']:
        if vix < 20 and total_ce > total_pe:
            goverdhan_signal = "Goverdhan: Buy CE (Bull Flag Breakout)"
    elif latest['EMA_KISS_FLY']:
        if vix < 20 and total_ce > total_pe:
            goverdhan_signal = "Goverdhan: Buy CE (EMA Kiss & Fly)"
    elif latest['HORIZONTAL_FADE']:
        if 15 < vix < 20:
            goverdhan_signal = "Goverdhan: Buy CE/PE (Horizontal Fade)"
    elif latest['VCP']:
        if vix < 20 and total_ce > total_pe:
            goverdhan_signal = "Goverdhan: Buy CE (VCP Breakout)"
    elif latest['REVERSAL_SQUEEZE']:
        if vix < 20 and total_ce > total_pe:
            goverdhan_signal = "Goverdhan: Buy CE (Reversal Squeeze)"
    return goverdhan_signal

# Generate Andrea Unger Signals
def generate_unger_signals(df, vix, total_ce, total_pe):
    latest = df.iloc[-1]
    unger_signal = "No Unger Signal"
    if latest['UNGER_SIGNAL'] == 'Buy (Breakout)':
        if vix < 20 and total_ce > total_pe:
            unger_signal = "Andrea Unger: Buy CE (Volatility Breakout)"
    elif latest['UNGER_SIGNAL'] == 'Sell (Breakdown)':
        if vix > 15 and total_pe > total_ce:
            unger_signal = "Andrea Unger: Buy PE (Volatility Breakdown)"
    return unger_signal

# Generate Michael Cook Signals
def generate_cook_signals(df, vix, total_ce, total_pe):
    latest = df.iloc[-1]
    cook_signal = "No Cook Signal"
    if latest['COOK_SIGNAL'] == 'Buy (VWAP Bounce)':
        if vix < 20 and total_ce > total_pe:
            cook_signal = "Michael Cook: Buy CE (VWAP Bounce)"
    elif latest['COOK_SIGNAL'] == 'Sell (VWAP Rejection)':
        if vix > 15 and total_pe > total_ce:
            cook_signal = "Michael Cook: Buy PE (VWAP Rejection)"
    return cook_signal

# Monitor Trade for Reversal Warnings
def monitor_trade(df_intra, trade_type, entry_price, stop_loss, prev_strikes_oi, index_symbol, simulation=False):
    if df_intra.empty:
        return False, ""
    latest = df_intra.iloc[-1]
    prev_idx = -2
    current_price = latest['Close']
    vix = latest['VIX']
    vol = latest['Volume']
    vol_avg = latest['VOL_AVG']
    ema10 = latest['EMA10']
    option_chain_symbol = 'NIFTY' if index_symbol == '^NSEI' else 'BANKNIFTY'
    is_stock = not index_symbol.startswith('^')
    current_strikes_oi, _, total_ce, total_pe, _ = fetch_option_chain(current_price, simulation, option_chain_symbol, is_stock)
    warning = False
    warning_message = ""
    bearish_engulfing = latest['Close'] < latest['Open'] and latest['High'] > df_intra['High'].iloc[prev_idx] and latest['Close'] < df_intra['Open'].iloc[prev_idx]
    bullish_engulfing = latest['Close'] > latest['Open'] and latest['Low'] < df_intra['Low'].iloc[prev_idx] and latest['Close'] > df_intra['Open'].iloc[prev_idx]
    vix_spike = vix > df_intra['VIX'].iloc[prev_idx] * 1.1
    vol_drop = vol < vol_avg * 0.8
    vol_spike = vol > vol_avg * 2
    ema_cross = (trade_type in ["Buy CE", "Sell Straddle"] and current_price < ema10) or (trade_type == "Buy PE" and current_price > ema10)
    oi_shift = False
    if current_strikes_oi and prev_strikes_oi:
        total_ce_curr = sum(d['CE_OI'] for d in current_strikes_oi.values())
        total_pe_curr = sum(d['PE_OI'] for d in current_strikes_oi.values())
        total_ce_prev = sum(d['CE_OI'] for d in prev_strikes_oi.values())
        total_pe_prev = sum(d['PE_OI'] for d in prev_strikes_oi.values())
        if trade_type == "Buy CE" and total_pe_curr > total_pe_prev * 1.25:
            oi_shift = True
        if trade_type == "Buy PE" and total_ce_curr > total_ce_prev * 1.25:
            oi_shift = True
    reversal_indicators = sum([
        bearish_engulfing and trade_type == "Buy CE",
        bullish_engulfing and trade_type == "Buy PE",
        vix_spike, vol_drop, vol_spike, ema_cross, oi_shift,
        latest['EXHAUSTION'] and trade_type == "Buy CE",
        latest['WEDGE_DROP'] and trade_type == "Buy CE",
        latest['WEDGE_POP'] and trade_type == "Buy PE"
    ])
    if reversal_indicators >= 2:
        warning = True
        warning_message = f"Warning: Potential Reversal - Exit {trade_type}"
    return warning, warning_message

# Generate Signals with Target/Stop-Loss - updated to handle small datasets
def generate_signals(df_intra, avg_band_pct, hist_atr_avg, index_symbol, simulation=False, prev_strikes_oi={}):
    if len(df_intra) < 5:
        st.error(f"Insufficient data ({len(df_intra)} rows) for signal generation for {index_symbol}. Need at least 5 rows.")
        return "Unknown", "No Signal", 0, 0, 0, False, {}, 0, "No Kell Signal", "No Goverdhan Signal", "No Unger Signal", "No Cook Signal", 0, 0, None, "Neutral"
    if len(df_intra) < 20:
        st.warning(f"Only {len(df_intra)} rows of data available for {index_symbol}. Indicators may not be accurate.")
    df_intra = compute_indicators(df_intra)
    if df_intra.empty:
        return "Unknown", "No Signal", 0, 0, 0, False, {}, 0, "No Kell Signal", "No Goverdhan Signal", "No Unger Signal", "No Cook Signal", 0, 0, None, "Neutral"
    df_intra = compute_kell_phases(df_intra)
    df_intra = compute_goverdhan_patterns(df_intra)
    df_intra = compute_unger_strategy(df_intra)
    df_intra = compute_cook_strategy(df_intra)
    latest = df_intra.iloc[-1]
    prev_idx = -2
    current_price = latest['Close']
    prev_price = df_intra['Close'].iloc[prev_idx]
    mid = latest['BB_Mid']
    atr = latest['ATR']
    vix = latest['VIX']
    adx = latest['ADX']
    option_chain_symbol = 'NIFTY' if index_symbol == '^NSEI' else 'BANKNIFTY'
    is_stock = not index_symbol.startswith('^')
    current_strikes_oi, bearish_oi, total_ce, total_pe, option_sentiment = fetch_option_chain(current_price, simulation, option_chain_symbol, is_stock)
    buildups = detect_oi_buildup(current_strikes_oi, prev_strikes_oi, current_price, prev_price, vix, simulation)
    direction = "Unknown"
    signal = "No Signal"
    target = 0
    stop_loss = 0
    trade_type = None
    kell_signal = generate_kell_signals(df_intra, vix, total_ce, total_pe)
    goverdhan_signal = generate_goverdhan_signals(df_intra, vix, total_ce, total_pe)
    unger_signal = generate_unger_signals(df_intra, vix, total_ce, total_pe)
    cook_signal = generate_cook_signals(df_intra, vix, total_ce, total_pe)
    intra_scale = 0.4 * (atr / (hist_atr_avg / 6)) if hist_atr_avg != 0 else 0.4
    adapted_band_pct = avg_band_pct * intra_scale
    if vix > 20:
        adapted_band_pct *= 1.3
    elif vix < 15:
        adapted_band_pct *= 0.9
    upper_band = mid + (adapted_band_pct / 200 * current_price)
    lower_band = mid - (adapted_band_pct / 200 * current_price)
    target_points = 2 * atr if adx > 25 else (upper_band - lower_band) * 0.5
    if vix > 20:
        target_points *= 1.2
    elif vix < 15:
        target_points *= 0.9
    stop_points = atr * (1.1 if atr > hist_atr_avg else 1.0)
    market_sentiment_bullish = total_ce > total_pe and vix < 20
    market_sentiment_bearish = total_pe > total_ce and vix > 20
    signals = [kell_signal, goverdhan_signal, unger_signal, cook_signal]
    valid_signals = [s for s in signals if s != "No Kell Signal" and s != "No Goverdhan Signal" and s != "No Unger Signal" and s != "No Cook Signal"]
    if valid_signals:
        signal = " | ".join(valid_signals)
        if "Buy CE" in signal and not market_sentiment_bullish:
            signal = "No Signal (Market sentiment not bullish)"
        elif "Buy PE" in signal and not market_sentiment_bearish:
            signal = "No Signal (Market sentiment not bearish)"
        else:
            if "Buy CE" in signal:
                trade_type = "Buy CE"
                target = current_price + target_points
                stop_loss = current_price - stop_points
            elif "Buy PE" in signal:
                trade_type = "Buy PE"
                target = current_price - target_points
                stop_loss = current_price + stop_points
    if buildups['bullish']:
        signal += f" + Potential Upside Blast: Buy OTM CE at strike {buildups['bullish']}"
    if buildups['bearish']:
        signal += f" + Potential Downside Blast: Buy OTM PE at strike {buildups['bearish']}"
    return direction, signal, upper_band, lower_band, vix, bearish_oi, buildups, current_price, kell_signal, goverdhan_signal, unger_signal, cook_signal, target, stop_loss, trade_type, option_sentiment

# Fetch Market Sentiment News
def fetch_market_sentiment():
    try:
        url = "https://economictimes.indiatimes.com/markets/stocks/news"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = []
            for item in soup.select('.eachStory'):
                headline = item.select_one('h3')
                if headline:
                    headlines.append(headline.text.strip())
            return headlines[:5]
    except Exception as e:
        st.error(f"Error fetching market sentiment: {e}")
    return []

# Check Market Open
def is_market_open():
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    ist_time = now.time()
    return now.weekday() < 5 and time(9,15) <= ist_time <= time(15,30)

# Convert 24-hour time format to 12-hour format
def format_time_12h(time_str):
    try:
        dt = datetime.strptime(time_str, '%H:%M')
        return dt.strftime('%I:%M %p')
    except:
        return time_str

# Get color for signal type
def get_signal_color(signal):
    if "Buy CE" in signal or "Long" in signal:
        return "background-color: #90EE90"
    elif "Buy PE" in signal or "Short" in signal:
        return "background-color: #FFB6C1"
    else:
        return "background-color: #FFFFE0"

# Format signal output as a table with background colors
def format_signal_table(results):
    df = pd.DataFrame(results)
    styled_df = df.style.applymap(get_signal_color)
    return styled_df

# Consensus Signal Logic
def get_consensus_signal(signals):
    long_count = sum(1 for s in signals if "Buy CE" in s or "Long" in s)
    short_count = sum(1 for s in signals if "Buy PE" in s or "Short" in s)
    if long_count >= 2:
        return "Long"
    elif short_count >= 2:
        return "Short"
    else:
        return "Sideways"

# Simulate Day with Monitoring - updated to use 15 days of data
def simulate_day(sim_date, df_daily_dict, avg_band_pct_dict, hist_atr_avg_dict, api_key=None, access_token=None, timeframe_minutes=15):
    intraday_data_dict = fetch_intraday_data(list(df_daily_dict.keys()), sim_date, api_key, access_token, timeframe_minutes, days=15)
    results = []
    for index_symbol, df_intra in intraday_data_dict.items():
        if df_intra.empty:
            st.warning(f"No intraday data for {index_symbol} on {sim_date}.")
            continue
        if len(df_intra) < 5:
            st.warning(f"Insufficient data for {index_symbol} on {sim_date} ({len(df_intra)} rows).")
            continue
        df_intra = compute_indicators(df_intra)
        if df_intra.empty:
            st.error(f"Failed to compute indicators for {index_symbol} on {sim_date}.")
            continue
        df_intra = compute_kell_phases(df_intra)
        df_intra = compute_goverdhan_patterns(df_intra)
        df_intra = compute_unger_strategy(df_intra)
        df_intra = compute_cook_strategy(df_intra)
        st.write(f"Simulating {sim_date} for {index_symbol} with multiple strategies - {len(df_intra)} {timeframe_minutes}m bars.")
        prev_strikes_oi = {}
        for i in range(21, len(df_intra)):
            df_slice = df_intra.iloc[:i+1]
            bar_time = df_slice.index[-1].strftime('%H:%M')
            direction, signal, upper, lower, vix, bearish_oi, buildups, current_price, kell_signal, goverdhan_signal, unger_signal, cook_signal, target, stop_loss, trade_type, option_sentiment = generate_signals(
                df_slice, avg_band_pct_dict[index_symbol], hist_atr_avg_dict[index_symbol], index_symbol, simulation=True, prev_strikes_oi=prev_strikes_oi
            )
            signals = [kell_signal, goverdhan_signal, unger_signal, cook_signal]
            consensus = get_consensus_signal(signals)
            results.append({
                'Time': format_time_12h(bar_time),
                'Price': current_price,
                'Signal': signal,
                'Consensus': consensus,
                'Kell Phase': df_slice.iloc[-1]['PHASE'],
                'Goverdhan Pattern': ', '.join([p for p in ['BULL_FLAG', 'EMA_KISS_FLY', 'HORIZONTAL_FADE', 'VCP', 'REVERSAL_SQUEEZE'] if df_slice.iloc[-1][p]]),
                'Unger Signal': df_slice.iloc[-1]['UNGER_SIGNAL'],
                'Cook Signal': df_slice.iloc[-1]['COOK_SIGNAL'],
                'Option Sentiment': option_sentiment
            })
        st.dataframe(format_signal_table(results))
        st.success("Simulation completed.")

# Simulate Date Range - updated to use 15 days of data
def simulate_date_range(start_date, end_date, df_daily_dict, avg_band_pct_dict, hist_atr_avg_dict, api_key=None, access_token=None, timeframe_minutes=15):
    start_date_dt = datetime.strptime(start_date.strip(), '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date.strip(), '%Y-%m-%d')
    date_range = pd.date_range(start=start_date_dt, end=end_date_dt, freq='B')
    all_results = pd.DataFrame()
    for sim_date in date_range:
        sim_date_str = sim_date.strftime('%Y-%m-%d')
        st.write(f"Simulating {sim_date_str}...")
        intraday_data_dict = fetch_intraday_data(list(df_daily_dict.keys()), sim_date_str, api_key, access_token, timeframe_minutes, days=15)
        for index_symbol, df_intra in intraday_data_dict.items():
            if df_intra.empty:
                st.warning(f"No intraday data for {index_symbol} on {sim_date_str}.")
                continue
            if len(df_intra) < 5:
                st.warning(f"Insufficient data for {index_symbol} on {sim_date_str} ({len(df_intra)} rows).")
                continue
            df_intra = compute_indicators(df_intra)
            if df_intra.empty:
                st.error(f"Failed to compute indicators for {index_symbol} on {sim_date_str}.")
                continue
            df_intra = compute_kell_phases(df_intra)
            df_intra = compute_goverdhan_patterns(df_intra)
            df_intra = compute_unger_strategy(df_intra)
            df_intra = compute_cook_strategy(df_intra)
            prev_strikes_oi = {}
            for i in range(21, len(df_intra)):
                df_slice = df_intra.iloc[:i+1]
                bar_time = df_slice.index[-1].strftime('%H:%M')
                direction, signal, upper, lower, vix, bearish_oi, buildups, current_price, kell_signal, goverdhan_signal, unger_signal, cook_signal, target, stop_loss, trade_type, option_sentiment = generate_signals(
                    df_slice, avg_band_pct_dict[index_symbol], hist_atr_avg_dict[index_symbol], index_symbol, simulation=True, prev_strikes_oi=prev_strikes_oi
                )
                signals = [kell_signal, goverdhan_signal, unger_signal, cook_signal]
                consensus = get_consensus_signal(signals)
                result_row = {
                    'Date': sim_date_str,
                    'Time': format_time_12h(bar_time),
                    'Index': index_symbol,
                    'Price': current_price,
                    'Signal': signal,
                    'Consensus': consensus,
                    'Kell Phase': df_slice.iloc[-1]['PHASE'],
                    'Goverdhan Pattern': ', '.join([p for p in ['BULL_FLAG', 'EMA_KISS_FLY', 'HORIZONTAL_FADE', 'VCP', 'REVERSAL_SQUEEZE'] if df_slice.iloc[-1][p]]),
                    'Unger Signal': df_slice.iloc[-1]['UNGER_SIGNAL'],
                    'Cook Signal': df_slice.iloc[-1]['COOK_SIGNAL'],
                    'Option Sentiment': option_sentiment
                }
                all_results = pd.concat([all_results, pd.DataFrame([result_row])], ignore_index=True)
    if not all_results.empty:
        st.dataframe(format_signal_table(all_results))
        st.success("Simulation completed.")
    else:
        st.warning("No results found for the selected date range.")

# Thread function for live scanning - updated to use 15 days of data
def live_scanning_thread():
    symbols_to_scan = ['^NSEI', '^BSESN']
    if st.session_state.stock_symbol:
        symbols_to_scan.append(st.session_state.stock_symbol)
    df_daily_dict = fetch_historical_data(['^NSEI', '^BSESN'])
    if not df_daily_dict:
        st.error("Exiting due to empty historical data.")
        return
    avg_band_pct_dict = {}
    hist_atr_avg_dict = {}
    for index_symbol, df_daily in df_daily_dict.items():
        df_daily = compute_indicators(df_daily, intraday=False)
        df_daily = df_daily.dropna()
        avg_band_pct, hist_atr_avg = estimate_band_size(df_daily)
        avg_band_pct_dict[index_symbol] = avg_band_pct
        hist_atr_avg_dict[index_symbol] = hist_atr_avg
    symbol_states = {symbol: {'prev_strikes_oi': {}} for symbol in symbols_to_scan}
    while st.session_state.live_scanning:
        if is_market_open():
            intraday_data_dict = fetch_intraday_data(['^NSEI', '^BSESN'], timeframe_minutes=st.session_state.timeframe_minutes, days=15)
            if st.session_state.stock_symbol:
                stock_data = fetch_stock_data(st.session_state.stock_symbol, period='7d', interval='5m')
                if stock_data is not None and not stock_data.empty:
                    stock_data = convert_timeframe(stock_data, timeframe_minutes=st.session_state.timeframe_minutes)
                    intraday_data_dict[st.session_state.stock_symbol] = stock_data
            results = []
            for symbol in symbols_to_scan:
                if symbol in intraday_data_dict and len(intraday_data_dict[symbol]) >= 5:
                    state = symbol_states[symbol]
                    avg_band_pct = avg_band_pct_dict.get(symbol, 6.5)
                    hist_atr_avg = hist_atr_avg_dict.get(symbol, 0)
                    direction, signal, upper, lower, vix, bearish_oi, buildups, current_price, kell_signal, goverdhan_signal, unger_signal, cook_signal, target, stop_loss, trade_type, option_sentiment = generate_signals(
                        intraday_data_dict[symbol], avg_band_pct, hist_atr_avg, symbol, prev_strikes_oi=state['prev_strikes_oi']
                    )
                    signals = [kell_signal, goverdhan_signal, unger_signal, cook_signal]
                    consensus = get_consensus_signal(signals)
                    results.append({
                        'Symbol': symbol,
                        'Time': datetime.now().strftime('%I:%M %p'),
                        'Price': current_price,
                        'Signal': signal,
                        'Consensus': consensus,
                        'VIX': vix,
                        'Option Sentiment': option_sentiment,
                        'Kell Signal': kell_signal,
                        'Goverdhan Signal': goverdhan_signal,
                        'Unger Signal': unger_signal,
                        'Cook Signal': cook_signal,
                        'Target': target,
                        'Stop Loss': stop_loss
                    })
                    if signal != "No Signal":
                        play_alert_sound("alert")
            st.session_state.live_results = results
            tm.sleep(st.session_state.timeframe_minutes * 60)
        else:
            st.session_state.market_closed = True
            tm.sleep(60)

# Main Streamlit App
def main():
    st.set_page_config(
        page_title="Intraday Trading Signal Scanner made By Ashutosh",
        page_icon="📈",
        layout="wide"
    )
    st.title("📈 Intraday Trading Signal Scanner made By Ashutosh")
    st.markdown("Scan for trading signals using multiple strategies: Oliver Kell, Goverdhan Gajjala, Andrea Unger, and Michael Cook")
    if 'live_scanning' not in st.session_state:
        st.session_state.live_scanning = False
    if 'stock_symbol' not in st.session_state:
        st.session_state.stock_symbol = ""
    if 'simulation_date' not in st.session_state:
        st.session_state.simulation_date = ""
    if 'timeframe_minutes' not in st.session_state:
        st.session_state.timeframe_minutes = 15
    if 'live_results' not in st.session_state:
        st.session_state.live_results = []
    if 'new_trades' not in st.session_state:
        st.session_state.new_trades = []
    if 'market_closed' not in st.session_state:
        st.session_state.market_closed = False
    if 'live_thread' not in st.session_state:
        st.session_state.live_thread = None
    option = st.sidebar.selectbox("Select an option", ["Live Scanning", "Simulation"])
    timeframe_minutes = st.sidebar.selectbox("Timeframe (minutes)", [5, 15], index=1)
    st.session_state.timeframe_minutes = timeframe_minutes
    if option == "Live Scanning":
        st.subheader("Live Scanning")
        stock_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS) - Leave empty to scan only indices", value=st.session_state.stock_symbol)
        st.session_state.stock_symbol = stock_symbol
        if st.button("Start Live Scanning" if not st.session_state.live_scanning else "Stop Live Scanning"):
            st.session_state.live_scanning = not st.session_state.live_scanning
            if st.session_state.live_scanning:
                st.session_state.live_thread = threading.Thread(target=live_scanning_thread, daemon=True)
                st.session_state.live_thread.start()
                st.success("Live scanning started in background!")
            else:
                st.success("Live scanning stopped!")
        if st.session_state.live_scanning:
            st.subheader("Live Signal Results")
            display_live_prices()
            results_placeholder = st.empty()
            if st.session_state.live_results:
                st.dataframe(format_signal_table(st.session_state.live_results))
            if st.session_state.market_closed:
                st.warning("Market is closed. Waiting for market to open...")
                st.session_state.market_closed = False
        st.subheader("Market Sentiment")
        headlines = fetch_market_sentiment()
        if headlines:
            for headline in headlines:
                st.write(f"- {headline}")
        else:
            st.write("No market sentiment data available.")
    elif option == "Simulation":
        st.subheader("Simulation Mode")
        sim_type = st.radio("Select simulation type", ["Single Date", "Date Range"])
        if sim_type == "Single Date":
            sim_date = st.date_input("Select a date to simulate", value=datetime.now().date())
            st.session_state.simulation_date = sim_date.strftime('%Y-%m-%d')
            selection_type = st.radio("Select simulation target", ["Indices", "Individual Stock"])
            if selection_type == "Indices":
                index_options = ['Nifty (^NSEI)', 'Sensex (^BSESN)']
                selected_index = st.selectbox("Select an index", index_options)
                index_symbol = '^NSEI' if selected_index == 'Nifty (^NSEI)' else '^BSESN'
                if st.button("Run Simulation"):
                    df_daily_dict = fetch_historical_data([index_symbol])
                    if not df_daily_dict:
                        st.error("Exiting due to empty historical data.")
                        return
                    avg_band_pct_dict = {}
                    hist_atr_avg_dict = {}
                    for index_symbol, df_daily in df_daily_dict.items():
                        df_daily = compute_indicators(df_daily, intraday=False)
                        df_daily = df_daily.dropna()
                        avg_band_pct, hist_atr_avg = estimate_band_size(df_daily)
                        avg_band_pct_dict[index_symbol] = avg_band_pct
                        hist_atr_avg_dict[index_symbol] = hist_atr_avg
                        st.write(f"{index_symbol} - Estimated Avg Sideways Band Size: {avg_band_pct:.2f}%")
                        st.write(f"{index_symbol} - Historical Avg Daily ATR: {hist_atr_avg:.2f}")
                    simulate_day(st.session_state.simulation_date, df_daily_dict, avg_band_pct_dict, hist_atr_avg_dict, timeframe_minutes=timeframe_minutes)
            else:
                stock_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS)", value="")
                if stock_symbol and st.button("Run Simulation"):
                    stock_data = fetch_stock_data(stock_symbol, period='7d', interval='5m')
                    if stock_data is not None and not stock_data.empty:
                        stock_data = compute_indicators(stock_data)
                        stock_data = compute_kell_phases(stock_data)
                        stock_data = compute_goverdhan_patterns(stock_data)
                        stock_data = compute_unger_strategy(stock_data)
                        stock_data = compute_cook_strategy(stock_data)
                        results = []
                        for i in range(21, len(stock_data)):
                            df_slice = stock_data.iloc[:i+1]
                            bar_time = df_slice.index[-1].strftime('%H:%M')
                            direction, signal, upper, lower, vix, bearish_oi, buildups, current_price, kell_signal, goverdhan_signal, unger_signal, cook_signal, target, stop_loss, trade_type, option_sentiment = generate_signals(
                                df_slice, 6.5, 0, stock_symbol, simulation=True
                            )
                            signals = [kell_signal, goverdhan_signal, unger_signal, cook_signal]
                            consensus = get_consensus_signal(signals)
                            results.append({
                                'Time': format_time_12h(bar_time),
                                'Price': current_price,
                                'Signal': signal,
                                'Consensus': consensus,
                                'Kell Phase': df_slice.iloc[-1]['PHASE'],
                                'Goverdhan Pattern': ', '.join([p for p in ['BULL_FLAG', 'EMA_KISS_FLY', 'HORIZONTAL_FADE', 'VCP', 'REVERSAL_SQUEEZE'] if df_slice.iloc[-1][p]]),
                                'Unger Signal': df_slice.iloc[-1]['UNGER_SIGNAL'],
                                'Cook Signal': df_slice.iloc[-1]['COOK_SIGNAL'],
                                'Option Sentiment': option_sentiment
                            })
                        st.dataframe(format_signal_table(results))
                        st.success("Simulation completed.")
                    else:
                        st.error(f"No data found for {stock_symbol}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date", value=datetime.now().date() - timedelta(days=7))
            with col2:
                end_date = st.date_input("End date", value=datetime.now().date())
            selection_type = st.radio("Select simulation target", ["Indices", "Individual Stock"], key="date_range_selection")
            if selection_type == "Indices":
                index_options = ['Nifty (^NSEI)', 'Sensex (^BSESN)']
                selected_index = st.selectbox("Select an index", index_options, key="date_range_index")
                index_symbol = '^NSEI' if selected_index == 'Nifty (^NSEI)' else '^BSESN'
                if st.button("Run Simulation", key="run_date_range"):
                    df_daily_dict = fetch_historical_data([index_symbol])
                    if not df_daily_dict:
                        st.error("Exiting due to empty historical data.")
                        return
                    avg_band_pct_dict = {}
                    hist_atr_avg_dict = {}
                    for index_symbol, df_daily in df_daily_dict.items():
                        df_daily = compute_indicators(df_daily, intraday=False)
                        df_daily = df_daily.dropna()
                        avg_band_pct, hist_atr_avg = estimate_band_size(df_daily)
                        avg_band_pct_dict[index_symbol] = avg_band_pct
                        hist_atr_avg_dict[index_symbol] = hist_atr_avg
                        st.write(f"{index_symbol} - Estimated Avg Sideways Band Size: {avg_band_pct:.2f}%")
                        st.write(f"{index_symbol} - Historical Avg Daily ATR: {hist_atr_avg:.2f}")
                    simulate_date_range(
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        df_daily_dict,
                        avg_band_pct_dict,
                        hist_atr_avg_dict,
                        timeframe_minutes=timeframe_minutes
                    )
            else:
                stock_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS)", value="", key="date_range_stock")
                if stock_symbol and st.button("Run Simulation", key="run_date_range_stock"):
                    start_date_dt = datetime.strptime(start_date.strftime('%Y-%m-%d'), '%Y-%m-%d')
                    end_date_dt = datetime.strptime(end_date.strftime('%Y-%m-%d'), '%Y-%m-%d')
                    date_range = pd.date_range(start=start_date_dt, end=end_date_dt, freq='B')
                    all_results = pd.DataFrame()
                    for sim_date in date_range:
                        sim_date_str = sim_date.strftime('%Y-%m-%d')
                        st.write(f"Processing {sim_date_str}...")
                        stock_data = fetch_stock_data(stock_symbol, period='7d', interval='5m')
                        if stock_data is not None and not stock_data.empty:
                            stock_data = compute_indicators(stock_data)
                            stock_data = compute_kell_phases(stock_data)
                            stock_data = compute_goverdhan_patterns(stock_data)
                            stock_data = compute_unger_strategy(stock_data)
                            stock_data = compute_cook_strategy(stock_data)
                            for i in range(21, len(stock_data)):
                                df_slice = stock_data.iloc[:i+1]
                                bar_time = df_slice.index[-1].strftime('%H:%M')
                                direction, signal, upper, lower, vix, bearish_oi, buildups, current_price, kell_signal, goverdhan_signal, unger_signal, cook_signal, target, stop_loss, trade_type, option_sentiment = generate_signals(
                                    df_slice, 6.5, 0, stock_symbol, simulation=True
                                )
                                signals = [kell_signal, goverdhan_signal, unger_signal, cook_signal]
                                consensus = get_consensus_signal(signals)
                                result_row = {
                                    'Date': sim_date_str,
                                    'Time': format_time_12h(bar_time),
                                    'Stock': stock_symbol,
                                    'Price': current_price,
                                    'Signal': signal,
                                    'Consensus': consensus,
                                    'Kell Phase': df_slice.iloc[-1]['PHASE'],
                                    'Goverdhan Pattern': ', '.join([p for p in ['BULL_FLAG', 'EMA_KISS_FLY', 'HORIZONTAL_FADE', 'VCP', 'REVERSAL_SQUEEZE'] if df_slice.iloc[-1][p]]),
                                    'Unger Signal': df_slice.iloc[-1]['UNGER_SIGNAL'],
                                    'Cook Signal': df_slice.iloc[-1]['COOK_SIGNAL'],
                                    'Option Sentiment': option_sentiment
                                }
                                all_results = pd.concat([all_results, pd.DataFrame([result_row])], ignore_index=True)
                    if not all_results.empty:
                        st.dataframe(format_signal_table(all_results))
                        st.success("Simulation completed.")
                    else:
                        st.warning("No results found for the selected date range.")

def display_live_prices():
    nifty_price = fetch_live_price('^NSEI')
    sensex_price = fetch_live_price('^BSESN')
    stock_price = fetch_live_price(st.session_state.stock_symbol) if st.session_state.stock_symbol else None
    st.write(f"**Nifty:** {nifty_price} | **Sensex:** {sensex_price} | **Stock:** {stock_price}")

def fetch_live_price(symbol):
    try:
        if symbol.startswith('^'):
            instrument_key = instrument_map.get(symbol)
            if not instrument_key:
                return "N/A"
            url = f"{BASE_URL}/market-quote/quotes?symbol={instrument_key}"
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {ACCESS_TOKEN}',
                'x-api-key': API_KEY
            }
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data and len(data['data']) > 0:
                    return data['data'][0]['last_price']
        else:
            ticker = yf.Ticker(symbol + '.NS')
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return data.iloc[-1]['Close']
    except Exception as e:
        st.error(f"Error fetching live price for {symbol}: {e}")
    return "N/A"

if __name__ == "__main__":
    main()
