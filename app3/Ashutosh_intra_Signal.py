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

# Sound alert function with error handling
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

# API credentials
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

# Convert 1-minute data to desired timeframe
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

# Generate sample data for testing
def generate_sample_data(symbol='^NSEI', days=30, base_price=None):
    st.info(f"Generating sample data for {symbol} for testing purposes...")
    
    end_date = datetime.now(pytz.timezone('Asia/Kolkata'))
    start_date = end_date - timedelta(days=days)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Use different base prices for different symbols
    if base_price is None:
        if symbol == '^NSEI':
            base_price = 19500
        elif symbol == '^BSESN':
            base_price = 65000
        else:
            base_price = 1000
    
    # Set a seed based on the symbol to ensure different data for different symbols
    np.random.seed(hash(symbol) % 1000)
    
    daily_data = []
    current_price = base_price
    
    for date in dates:
        # Daily price change with some randomness
        change = np.random.normal(0, 0.01)
        open_price = current_price * (1 + change * 0.5)
        close_price = current_price * (1 + change)
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
        
        # Update base price for next day
        current_price = close_price
    
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

# Generate sample intraday data for a specific date
def generate_sample_intraday_data_for_date(symbol, date, timeframe_minutes=15):
    st.info(f"Generating sample intraday data for {symbol} for {date}...")
    
    # Create timestamps for market hours (9:15 to 15:30)
    start_time = datetime.combine(date, time(9, 15)).astimezone(pytz.timezone('Asia/Kolkata'))
    end_time = datetime.combine(date, time(15, 30)).astimezone(pytz.timezone('Asia/Kolkata'))
    
    # Create 1-minute timestamps
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Use different base prices for different symbols
    if symbol == '^NSEI':
        base_price = 19500
    elif symbol == '^BSESN':
        base_price = 65000
    else:
        base_price = 1000
    
    # Set a seed based on the symbol and date to ensure different data
    seed = hash(symbol + str(date)) % 1000
    np.random.seed(seed)
    
    intraday_data = []
    current_price = base_price
    
    for timestamp in timestamps:
        # 1-minute price change with some randomness
        change = np.random.normal(0, 0.0005)
        open_price = current_price * (1 + change * 0.5)
        close_price = current_price * (1 + change)
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
        
        # Update base price for next minute
        current_price = close_price
    
    df = pd.DataFrame(intraday_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    if df.index.tz is None:
        df.index = df.index.tz_localize('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')
    
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
    df['VIX'] = 15.0
    
    # Convert to desired timeframe
    df = convert_timeframe(df, timeframe_minutes=timeframe_minutes)
    
    # Ensure we have data for the full market hours
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
                    st.info("Using sample data as fallback...")
                    df = generate_sample_data(symbol)
                    historical_data[symbol] = df
                    continue
                    
                data = response.json()
                
                if not data or 'data' not in data or 'candles' not in data['data']:
                    st.error(f"No data returned for {symbol}")
                    st.info("Using sample data as fallback...")
                    df = generate_sample_data(symbol)
                    historical_data[symbol] = df
                    continue
                    
                candles = data['data']['candles']
                if not candles:
                    st.error(f"No candles returned for {symbol}")
                    st.info("Using sample data as fallback...")
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

# Fetch Intraday Data for a specific date
def fetch_intraday_data_for_date(index_symbols, sim_date, api_key=None, access_token=None, timeframe_minutes=15):
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
    
    # Parse the simulation date
    sim_date_dt = datetime.strptime(sim_date.strip(), '%Y-%m-%d')
    
    for symbol in index_symbols:
        try:
            st.write(f"Fetching intraday data for {symbol} on {sim_date}")
            
            if symbol.startswith('^'):
                instrument_key = instrument_map.get(symbol)
                if not instrument_key:
                    st.error(f"No instrument key found for {symbol}")
                    continue
            else:
                instrument_key = get_stock_instrument_key(symbol)
            
            # Format dates correctly for Upstox API
            to_date = sim_date_dt.strftime('%Y-%m-%d')
            from_date = sim_date_dt.strftime('%Y-%m-%d')
            
            # Try to fetch intraday data using the new endpoint
            url = f"{BASE_URL}/historical-candle/intraday/{instrument_key}/1minute"
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    st.error(f"API returned status code {response.status_code} for {symbol}")
                    st.info("Using sample data as fallback...")
                    df = generate_sample_intraday_data_for_date(symbol, sim_date_dt, timeframe_minutes)
                    intraday_data[symbol] = df
                    continue
                    
                data = response.json()
                
                if not data or 'data' not in data or 'candles' not in data['data']:
                    st.error(f"No data returned for {symbol}")
                    st.info("Using sample data as fallback...")
                    df = generate_sample_intraday_data_for_date(symbol, sim_date_dt, timeframe_minutes)
                    intraday_data[symbol] = df
                    continue
                    
                candles = data['data']['candles']
                if not candles:
                    st.error(f"No candles returned for {symbol}")
                    st.info("Using sample data as fallback...")
                    df = generate_sample_intraday_data_for_date(symbol, sim_date_dt, timeframe_minutes)
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
                
                # Filter to market hours
                df = df.between_time('09:15', '15:30')
                
                # Add VIX (use default for intraday)
                df['VIX'] = 15.0
                
                # Convert 1-minute data to desired timeframe
                df = convert_timeframe(df, timeframe_minutes=timeframe_minutes)
                
                # Ensure we have data for the full market hours
                df = df.between_time('09:15', '15:30')
                
                # Filter to the simulation date
                df = df[df.index.date == sim_date_dt.date()]
                
                # Check if we got any data
                if df.empty:
                    st.warning(f"No data returned after processing for {symbol}. Using sample data as fallback.")
                    df = generate_sample_intraday_data_for_date(symbol, sim_date_dt, timeframe_minutes)
                
                st.success(f"Successfully fetched and converted {len(df)} rows of {timeframe_minutes}-minute intraday data for {symbol}")
                intraday_data[symbol] = df
                
            except requests.exceptions.ConnectionError:
                st.error(f"Connection error when fetching intraday data for {symbol}. Using sample data as fallback...")
                df = generate_sample_intraday_data_for_date(symbol, sim_date_dt, timeframe_minutes)
                intraday_data[symbol] = df
                continue
                
        except Exception as e:
            st.error(f"Error fetching intraday data for {symbol}: {e}")
            st.info("Using sample data as fallback...")
            df = generate_sample_intraday_data_for_date(symbol, sim_date_dt, timeframe_minutes)
            intraday_data[symbol] = df
            continue
    
    return intraday_data

# Fetch stock data from Yahoo Finance for a specific date
def fetch_stock_data_for_date(symbol, date, timeframe_minutes=15):
    try:
        if not symbol.endswith(('.NS', '.BO')):
            symbol = symbol + '.NS'
            
        ticker = yf.Ticker(symbol)
        
        # Fetch data for a wider period to ensure we get the specific date
        end_date = date + timedelta(days=1)
        start_date = date - timedelta(days=7)
        
        data = ticker.history(start=start_date, end=end_date, interval='5m')
        
        if data.empty:
            st.warning(f"No data found for {symbol}")
            return None
            
        # Reset index to make timestamp a column
        data.reset_index(inplace=True)
        
        # Convert timezone to Asia/Kolkata
        if 'Datetime' in data.columns:
            data['Datetime'] = pd.to_datetime(data['Datetime']).dt.tz_convert('Asia/Kolkata')
            data.set_index('Datetime', inplace=True)
        elif 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_convert('Asia/Kolkata')
            data.set_index('Date', inplace=True)
        
        # Rename columns to match our format
        data.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }, inplace=True)
        
        # Add OI column (not available in Yahoo Finance)
        data['OI'] = 0
        
        # Add VIX column (not available for individual stocks)
        data['VIX'] = 15.0
        
        # Filter to the specific date
        data = data[data.index.date == date.date()]
        
        # Convert to desired timeframe
        data = convert_timeframe(data, timeframe_minutes=timeframe_minutes)
        
        # Ensure we have data for the full market hours
        data = data.between_time('09:15', '15:30')
        
        return data
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {e}")
        return None

# Compute Indicators
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

# Estimate Average Sideways Band Size
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

# Function to determine combined signals from multiple strategies
def determine_combined_signal(kell_signal, goverdhan_signal, unger_signal, cook_signal, option_sentiment):
    long_signals = 0
    short_signals = 0
    sideways_signals = 0
    
    if "Buy CE" in kell_signal or "Long" in kell_signal:
        long_signals += 1
    elif "Buy PE" in kell_signal or "Short" in kell_signal:
        short_signals += 1
    else:
        sideways_signals += 1
    
    if "Buy CE" in goverdhan_signal or "Long" in goverdhan_signal:
        long_signals += 1
    elif "Buy PE" in goverdhan_signal or "Short" in goverdhan_signal:
        short_signals += 1
    else:
        sideways_signals += 1
    
    if "Buy CE" in unger_signal or "Long" in unger_signal:
        long_signals += 1
    elif "Buy PE" in unger_signal or "Short" in unger_signal:
        short_signals += 1
    else:
        sideways_signals += 1
    
    if "Buy CE" in cook_signal or "Long" in cook_signal:
        long_signals += 1
    elif "Buy PE" in cook_signal or "Short" in cook_signal:
        short_signals += 1
    else:
        sideways_signals += 1
    
    if "Bullish" in option_sentiment:
        long_signals += 1
    elif "Bearish" in option_sentiment:
        short_signals += 1
    else:
        sideways_signals += 1
    
    if long_signals >= 2:
        return "Long (Buy CE)"
    elif short_signals >= 2:
        return "Short (Buy PE)"
    elif sideways_signals >= 2:
        return "Sideways"
    else:
        return "No Clear Signal"

# Get color for signal type
def get_signal_color(signal):
    if isinstance(signal, str):
        if "Buy CE" in signal or "Long" in signal:
            return "#90EE90"  # Light forest green
        elif "Buy PE" in signal or "Short" in signal:
            return "#FFB6C1"  # Light red
        elif "Sideways" in signal:
            return "#FFFFE0"  # Light yellow
        else:
            return "#FFFFFF"  # White for no signal
    return "#FFFFFF"  # Default white

# Generate Signals with Target/Stop-Loss
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
    
    combined_signal = determine_combined_signal(kell_signal, goverdhan_signal, unger_signal, cook_signal, option_sentiment)
    
    if combined_signal == "Long (Buy CE)":
        signal = "Long (Buy CE)"
        trade_type = "Buy CE"
        target = current_price + target_points
        stop_loss = current_price - stop_points
    elif combined_signal == "Short (Buy PE)":
        signal = "Short (Buy PE)"
        trade_type = "Buy PE"
        target = current_price - target_points
        stop_loss = current_price + stop_points
    elif combined_signal == "Sideways":
        signal = "Sideways (No Trade)"
    else:
        signal = "No Clear Signal"
    
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

# Fix simulate_day function to show results in table format for all time periods
def simulate_day(sim_date, df_daily_dict, avg_band_pct_dict, hist_atr_avg_dict, api_key=None, access_token=None, timeframe_minutes=15):
    # Parse the simulation date
    sim_date_dt = datetime.strptime(sim_date.strip(), '%Y-%m-%d')
    
    # Fetch intraday data for the specific date
    intraday_data_dict = fetch_intraday_data_for_date(list(df_daily_dict.keys()), sim_date, api_key, access_token, timeframe_minutes)
    
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
        
        # Create a DataFrame to store the results
        results = []
        
        prev_strikes_oi = {}
        active_trade = None
        trade_entry_price = 0
        trade_type = None
        stop_loss = 0
        avg_band_pct = avg_band_pct_dict.get(index_symbol, 6.5)
        hist_atr_avg = hist_atr_avg_dict.get(index_symbol, 0)
        
        # Process all data points for full market hours (9:15 AM to 3:30 PM)
        for i in range(21, len(df_intra)):
            df_slice = df_intra.iloc[:i+1]
            bar_time = df_slice.index[-1].strftime('%H:%M')
            
            direction, signal, upper, lower, vix, bearish_oi, buildups, current_price, kell_signal, goverdhan_signal, unger_signal, cook_signal, target, stop_loss, trade_type, option_sentiment = generate_signals(
                df_slice, avg_band_pct, hist_atr_avg, index_symbol, simulation=True, prev_strikes_oi=prev_strikes_oi
            )
            
            # Get the latest patterns for display
            kell_phase = df_slice.iloc[-1]['PHASE']
            goverdhan_patterns = ', '.join([p for p in ['BULL_FLAG', 'EMA_KISS_FLY', 'HORIZONTAL_FADE', 'VCP', 'REVERSAL_SQUEEZE'] if df_slice.iloc[-1][p]])
            unger_pattern = df_slice.iloc[-1]['UNGER_SIGNAL']
            cook_pattern = df_slice.iloc[-1]['COOK_SIGNAL']
            
            # Add to results - convert time to 12-hour format
            results.append({
                'Time': format_time_12h(bar_time),
                'Price': current_price,
                'Signal': signal,
                'Kell Phase': kell_phase,
                'Goverdhan Pattern': goverdhan_patterns,
                'Unger Signal': unger_pattern,
                'Cook Signal': cook_pattern,
                'Option Sentiment': option_sentiment
            })
            
            warning, warning_message = False, ""
            if active_trade:
                warning, warning_message = monitor_trade(df_slice, active_trade, trade_entry_price, stop_loss, prev_strikes_oi, index_symbol, simulation=True)
                if warning:
                    st.warning(f"Sim Time: {bar_time} | Warning: {warning_message}")
                    active_trade = None
            
            if signal != "No Signal" and not active_trade:
                active_trade = trade_type
                trade_entry_price = current_price
                
                # Only show new trade notification in the sidebar, not in the main results
                with st.sidebar:
                    st.subheader(f"*** NEW TRADE ***")
                    st.write(f"Time: {format_time_12h(bar_time)} [{index_symbol}]")
                    st.write(f"Current Price: {current_price:.2f}")
                    st.write(f"Direction: {direction}")
                    
                    # Display signal with color
                    signal_color = get_signal_color(signal)
                    st.markdown(f"Signal: <span style='color:{signal_color}'>{signal}</span>", unsafe_allow_html=True)
                    
                    st.write(f"Adapted Bands: Lower={lower:.2f}, Upper={upper:.2f}")
                    st.write(f"VIX: {vix:.2f} (Bearish OI: {bearish_oi})")
                    st.write(f"Option Sentiment: {option_sentiment}")
                    st.write(f"Kell Phase: {kell_phase}")
                    st.write(f"Goverdhan Pattern: {goverdhan_patterns}")
                    st.write(f"Unger Signal: {unger_pattern}")
                    st.write(f"Cook Signal: {cook_pattern}")
                    
                    if buildups['bullish'] or buildups['bearish']:
                        st.write(f"OI Buildups: Bullish={buildups['bullish']}, Bearish={buildups['bearish']}")
                    
                    st.write(f"Target: {target:.2f}, Stop Loss: {stop_loss:.2f}")
                    st.write("****************")
                
                # Play sound alert
                play_alert_sound("alert")
            
            prev_strikes_oi = {}
        
        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)
        
        # Apply color coding to the DataFrame
        def highlight_signals(val):
            color = get_signal_color(val)
            return f'background-color: {color}'
        
        # Apply the styling
        styled_results = results_df.style.applymap(highlight_signals, subset=['Signal'])
        
        # Display the styled DataFrame
        st.dataframe(styled_results)
        st.success("Simulation completed.")

# Fix individual stock simulation
def simulate_stock(stock_symbol, sim_date=None):
    # Parse the simulation date
    sim_date_dt = datetime.strptime(sim_date.strip(), '%Y-%m-%d')
    
    # Fetch stock data for the specific date
    stock_data = fetch_stock_data_for_date(stock_symbol, sim_date_dt)
    
    if stock_data is None or stock_data.empty:
        st.error(f"No data found for {stock_symbol}")
        return
    
    # Compute indicators
    stock_data = compute_indicators(stock_data)
    if stock_data.empty:
        st.error(f"Failed to compute indicators for {stock_symbol}")
        return
    
    stock_data = compute_kell_phases(stock_data)
    stock_data = compute_goverdhan_patterns(stock_data)
    stock_data = compute_unger_strategy(stock_data)
    stock_data = compute_cook_strategy(stock_data)
    
    # Create a DataFrame to store the results
    results = []
    
    # Process all data points for full market hours (9:15 AM to 3:30 PM)
    for i in range(21, len(stock_data)):
        df_slice = stock_data.iloc[:i+1]
        bar_time = df_slice.index[-1].strftime('%H:%M')
        
        direction, signal, upper, lower, vix, bearish_oi, buildups, current_price, kell_signal, goverdhan_signal, unger_signal, cook_signal, target, stop_loss, trade_type, option_sentiment = generate_signals(
            df_slice, 6.5, 0, stock_symbol, simulation=True
        )
        
        # Get the latest patterns for display
        kell_phase = df_slice.iloc[-1]['PHASE']
        goverdhan_patterns = ', '.join([p for p in ['BULL_FLAG', 'EMA_KISS_FLY', 'HORIZONTAL_FADE', 'VCP', 'REVERSAL_SQUEEZE'] if df_slice.iloc[-1][p]])
        unger_pattern = df_slice.iloc[-1]['UNGER_SIGNAL']
        cook_pattern = df_slice.iloc[-1]['COOK_SIGNAL']
        
        # Add to results - convert time to 12-hour format
        results.append({
            'Time': format_time_12h(bar_time),
            'Price': current_price,
            'Signal': signal,
            'Kell Phase': kell_phase,
            'Goverdhan Pattern': goverdhan_patterns,
            'Unger Signal': unger_pattern,
            'Cook Signal': cook_pattern,
            'Option Sentiment': option_sentiment
        })
        
        if signal != "No Signal":
            # Only show new trade notification in the sidebar, not in the main results
            with st.sidebar:
                st.subheader(f"*** NEW TRADE ***")
                st.write(f"Time: {format_time_12h(bar_time)} [{stock_symbol}]")
                st.write(f"Current Price: {current_price:.2f}")
                st.write(f"Direction: {direction}")
                
                # Display signal with color
                signal_color = get_signal_color(signal)
                st.markdown(f"Signal: <span style='color:{signal_color}'>{signal}</span>", unsafe_allow_html=True)
                
                st.write(f"Adapted Bands: Lower={lower:.2f}, Upper={upper:.2f}")
                st.write(f"VIX: {vix:.2f}")
                st.write(f"Option Sentiment: {option_sentiment}")
                st.write(f"Kell Phase: {kell_phase}")
                st.write(f"Goverdhan Pattern: {goverdhan_patterns}")
                st.write(f"Unger Signal: {unger_pattern}")
                st.write(f"Cook Signal: {cook_pattern}")
                
                if buildups['bullish'] or buildups['bearish']:
                    st.write(f"OI Buildups: Bullish={buildups['bullish']}, Bearish={buildups['bearish']}")
                
                st.write(f"Target: {target:.2f}, Stop Loss: {stop_loss:.2f}")
                st.write("****************")
            
            # Play sound alert
            play_alert_sound("alert")
    
    # Create a DataFrame from the results
    if results:
        results_df = pd.DataFrame(results)
        
        # Apply color coding to the DataFrame
        def highlight_signals(val):
            color = get_signal_color(val)
            return f'background-color: {color}'
        
        # Apply the styling
        styled_results = results_df.style.applymap(highlight_signals, subset=['Signal'])
        
        # Display the styled DataFrame
        st.dataframe(styled_results)
        st.success("Simulation completed.")
    else:
        st.warning("No results to display for the simulation.")

# Simulate Date Range
def simulate_date_range(start_date, end_date, df_daily_dict, avg_band_pct_dict, hist_atr_avg_dict, api_key=None, access_token=None, timeframe_minutes=15):
    start_date_dt = datetime.strptime(start_date.strip(), '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date.strip(), '%Y-%m-%d')
    
    date_range = pd.date_range(start=start_date_dt, end=end_date_dt, freq='B')
    
    all_results = pd.DataFrame()
    
    for sim_date in date_range:
        sim_date_str = sim_date.strftime('%Y-%m-%d')
        st.write(f"Simulating {sim_date_str}...")
        
        intraday_data_dict = fetch_intraday_data_for_date(list(df_daily_dict.keys()), sim_date_str, api_key, access_token, timeframe_minutes)
        
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
            active_trade = None
            trade_entry_price = 0
            trade_type = None
            stop_loss = 0
            avg_band_pct = avg_band_pct_dict.get(index_symbol, 6.5)
            hist_atr_avg = hist_atr_avg_dict.get(index_symbol, 0)
            
            for i in range(21, len(df_intra)):
                df_slice = df_intra.iloc[:i+1]
                bar_time = df_slice.index[-1].strftime('%H:%M')
                
                direction, signal, upper, lower, vix, bearish_oi, buildups, current_price, kell_signal, goverdhan_signal, unger_signal, cook_signal, target, stop_loss, trade_type, option_sentiment = generate_signals(
                    df_slice, avg_band_pct, hist_atr_avg, index_symbol, simulation=True, prev_strikes_oi=prev_strikes_oi
                )
                
                kell_phase = df_slice.iloc[-1]['PHASE']
                goverdhan_patterns = ', '.join([p for p in ['BULL_FLAG', 'EMA_KISS_FLY', 'HORIZONTAL_FADE', 'VCP', 'REVERSAL_SQUEEZE'] if df_slice.iloc[-1][p]])
                unger_pattern = df_slice.iloc[-1]['UNGER_SIGNAL']
                cook_pattern = df_slice.iloc[-1]['COOK_SIGNAL']
                
                result_row = {
                    'Date': sim_date_str,
                    'Time': format_time_12h(bar_time),
                    'Index': index_symbol,
                    'Price': current_price,
                    'Signal': signal,
                    'Kell Phase': kell_phase,
                    'Goverdhan Pattern': goverdhan_patterns,
                    'Unger Signal': unger_pattern,
                    'Cook Signal': cook_pattern,
                    'Option Sentiment': option_sentiment
                }
                
                all_results = pd.concat([all_results, pd.DataFrame([result_row])], ignore_index=True)
                
                warning, warning_message = False, ""
                if active_trade:
                    warning, warning_message = monitor_trade(df_slice, active_trade, trade_entry_price, stop_loss, prev_strikes_oi, index_symbol, simulation=True)
                    if warning:
                        st.warning(f"Sim Time: {bar_time} | Warning: {warning_message}")
                        active_trade = None
                
                if signal != "No Signal" and not active_trade:
                    active_trade = trade_type
                    trade_entry_price = current_price
                    
                    # Only show new trade notification in the sidebar, not in the main results
                    with st.sidebar:
                        st.subheader(f"*** NEW TRADE ***")
                        st.write(f"Date: {sim_date_str}")
                        st.write(f"Time: {format_time_12h(bar_time)} [{index_symbol}]")
                        st.write(f"Current Price: {current_price:.2f}")
                        st.write(f"Direction: {direction}")
                        
                        signal_color = get_signal_color(signal)
                        st.markdown(f"Signal: <span style='color:{signal_color}'>{signal}</span>", unsafe_allow_html=True)
                        
                        st.write(f"Adapted Bands: Lower={lower:.2f}, Upper={upper:.2f}")
                        st.write(f"VIX: {vix:.2f} (Bearish OI: {bearish_oi})")
                        st.write(f"Option Sentiment: {option_sentiment}")
                        st.write(f"Kell Phase: {kell_phase}")
                        st.write(f"Goverdhan Pattern: {goverdhan_patterns}")
                        st.write(f"Unger Signal: {unger_pattern}")
                        st.write(f"Cook Signal: {cook_pattern}")
                        
                        if buildups['bullish'] or buildups['bearish']:
                            st.write(f"OI Buildups: Bullish={buildups['bullish']}, Bearish={buildups['bearish']}")
                        
                        st.write(f"Target: {target:.2f}, Stop Loss: {stop_loss:.2f}")
                        st.write("****************")
                    
                    play_alert_sound("alert")
                
                prev_strikes_oi = {}
    
    if not all_results.empty:
        def highlight_signals(val):
            if isinstance(val, str):
                color = get_signal_color(val)
                return f'background-color: {color}'
            return ''
        
        styled_results = all_results.style.applymap(highlight_signals, subset=['Signal'])
        st.dataframe(styled_results)
        st.success("Simulation completed.")
    else:
        st.warning("No results found for the selected date range.")

# Update live_scanning_thread function
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
    
    symbol_states = {}
    for symbol in symbols_to_scan:
        symbol_states[symbol] = {
            'active_trade': None,
            'trade_entry_price': 0,
            'trade_type': None,
            'stop_loss': 0,
            'prev_strikes_oi': {},
            'current_price': 0
        }
    
    while st.session_state.live_scanning:
        if is_market_open():
            # Get current date for live scanning
            current_date = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d')
            
            intraday_data_dict = fetch_intraday_data_for_date(['^NSEI', '^BSESN'], current_date, timeframe_minutes=st.session_state.timeframe_minutes)
            
            if st.session_state.stock_symbol:
                stock_data = fetch_stock_data_for_date(st.session_state.stock_symbol, datetime.now(pytz.timezone('Asia/Kolkata')), st.session_state.timeframe_minutes)
                if stock_data is not None and not stock_data.empty:
                    intraday_data_dict[st.session_state.stock_symbol] = stock_data
            
            results = []
            current_prices = {}
            
            for symbol in symbols_to_scan:
                if symbol in intraday_data_dict and len(intraday_data_dict[symbol]) >= 5:
                    state = symbol_states[symbol]
                    avg_band_pct = avg_band_pct_dict.get(symbol, 6.5)
                    hist_atr_avg = hist_atr_avg_dict.get(symbol, 0)
                    
                    direction, signal, upper, lower, vix, bearish_oi, buildups, current_price, kell_signal, goverdhan_signal, unger_signal, cook_signal, target, stop_loss, trade_type, option_sentiment = generate_signals(
                        intraday_data_dict[symbol], avg_band_pct, hist_atr_avg, symbol, prev_strikes_oi=state['prev_strikes_oi']
                    )
                    
                    state['current_price'] = current_price
                    current_prices[symbol] = current_price
                    
                    results.append({
                        'Symbol': symbol,
                        'Time': datetime.now().strftime('%I:%M:%S %p'),
                        'Price': current_price,
                        'Signal': signal,
                        'VIX': vix,
                        'Option Sentiment': option_sentiment,
                        'Kell Signal': kell_signal,
                        'Goverdhan Signal': goverdhan_signal,
                        'Unger Signal': unger_signal,
                        'Cook Signal': cook_signal,
                        'Target': target,
                        'Stop Loss': stop_loss
                    })
                    
                    if state['active_trade']:
                        warning, warning_message = monitor_trade(intraday_data_dict[symbol], state['active_trade'], state['trade_entry_price'], state['stop_loss'], state['prev_strikes_oi'], symbol)
                        if warning:
                            st.warning(f"Time: {datetime.now().strftime('%I:%M:%S %p')} | {warning_message}")
                            play_alert_sound("warning")
                            state['active_trade'] = None
                    
                    if signal != "No Signal" and not state['active_trade']:
                        state['active_trade'] = trade_type
                        state['trade_entry_price'] = current_price
                        state['stop_loss'] = stop_loss
                        
                        new_trade = {
                            'time': datetime.now().strftime('%I:%M:%S %p'),
                            'symbol': symbol,
                            'current_price': current_price,
                            'direction': direction,
                            'signal': signal,
                            'upper': upper,
                            'lower': lower,
                            'vix': vix,
                            'bearish_oi': bearish_oi,
                            'option_sentiment': option_sentiment,
                            'kell_phase': intraday_data_dict[symbol].iloc[-1]['PHASE'],
                            'goverdhan_patterns': ', '.join([p for p in ['BULL_FLAG', 'EMA_KISS_FLY', 'HORIZONTAL_FADE', 'VCP', 'REVERSAL_SQUEEZE'] if intraday_data_dict[symbol].iloc[-1][p]]),
                            'unger_signal': intraday_data_dict[symbol].iloc[-1]['UNGER_SIGNAL'],
                            'cook_signal': intraday_data_dict[symbol].iloc[-1]['COOK_SIGNAL'],
                            'buildups': buildups,
                            'target': target,
                            'stop_loss': stop_loss,
                            'kell_signal': kell_signal,
                            'goverdhan_signal': goverdhan_signal,
                            'unger_signal': unger_signal,
                            'cook_signal': cook_signal
                        }
                        
                        if 'new_trades' not in st.session_state:
                            st.session_state.new_trades = []
                        st.session_state.new_trades.append(new_trade)
                        
                        play_alert_sound("alert")
                    
                    state['prev_strikes_oi'] = {}
            
            st.session_state.live_results = results
            st.session_state.current_prices = current_prices
            
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
    if 'current_prices' not in st.session_state:
        st.session_state.current_prices = {}
    
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        "Select an option",
        ["Live Scanning", "Simulation"]
    )
    
    timeframe_minutes = st.sidebar.selectbox(
        "Timeframe (minutes)",
        [5, 15],
        index=1
    )
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
            st.subheader("Live Prices")
            if st.session_state.current_prices:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if '^NSEI' in st.session_state.current_prices:
                        st.write(f"Nifty: {st.session_state.current_prices['^NSEI']:.2f}")
                with col2:
                    if '^BSESN' in st.session_state.current_prices:
                        st.write(f"Sensex: {st.session_state.current_prices['^BSESN']:.2f}")
                with col3:
                    if stock_symbol and stock_symbol in st.session_state.current_prices:
                        st.write(f"{stock_symbol}: {st.session_state.current_prices[stock_symbol]:.2f}")
            
            st.subheader("Live Signal Results")
            
            if st.session_state.live_results:
                table_data = []
                
                symbols = ['^NSEI', '^BSESN']
                if stock_symbol:
                    symbols.append(stock_symbol)
                
                for symbol in symbols:
                    symbol_results = [r for r in st.session_state.live_results if r['Symbol'] == symbol]
                    if symbol_results:
                        latest_result = symbol_results[-1]
                        
                        table_data.append({
                            'Symbol': symbol,
                            'Time': latest_result['Time'],
                            'Price': latest_result['Price'],
                            'Signal': latest_result['Signal'],
                            'Kell Signal': latest_result['Kell Signal'],
                            'Goverdhan Signal': latest_result['Goverdhan Signal'],
                            'Unger Signal': latest_result['Unger Signal'],
                            'Cook Signal': latest_result['Cook Signal'],
                            'Option Sentiment': latest_result['Option Sentiment'],
                            'Target': latest_result['Target'],
                            'Stop Loss': latest_result['Stop Loss']
                        })
                
                if table_data:
                    display_df = pd.DataFrame(table_data)
                    
                    def highlight_signals(val):
                        if isinstance(val, str):
                            color = get_signal_color(val)
                            return f'background-color: {color}'
                        return ''
                    
                    styled_df = display_df.style.applymap(highlight_signals, subset=['Signal', 'Kell Signal', 'Goverdhan Signal', 'Unger Signal', 'Cook Signal'])
                    
                    st.dataframe(styled_df)
                else:
                    st.write("Fetching data...")
            else:
                st.write("Fetching data...")
            
            if st.session_state.new_trades:
                st.subheader("New Trades")
                for trade in st.session_state.new_trades:
                    st.subheader(f"*** NEW TRADE ***")
                    st.write(f"Time: {trade['time']} [{trade['symbol']}]")
                    st.write(f"Current Price: {trade['current_price']:.2f}")
                    st.write(f"Direction: {trade['direction']}")
                    
                    signal_color = get_signal_color(trade['signal'])
                    st.markdown(f"Signal: <span style='color:{signal_color}'>{trade['signal']}</span>", unsafe_allow_html=True)
                    
                    st.write(f"Adapted Bands: Lower={trade['lower']:.2f}, Upper={trade['upper']:.2f}")
                    st.write(f"VIX: {trade['vix']:.2f} (Bearish OI: {trade['bearish_oi']})")
                    st.write(f"Option Sentiment: {trade['option_sentiment']}")
                    st.write(f"Kell Phase: {trade['kell_phase']}")
                    st.write(f"Goverdhan Pattern: {trade['goverdhan_patterns']}")
                    st.write(f"Unger Signal: {trade['unger_signal']}")
                    st.write(f"Cook Signal: {trade['cook_signal']}")
                    
                    if trade['buildups']['bullish'] or trade['buildups']['bearish']:
                        st.write(f"OI Buildups: Bullish={trade['buildups']['bullish']}, Bearish={trade['buildups']['bearish']}")
                    
                    st.write(f"Target: {trade['target']:.2f}, Stop Loss: {trade['stop_loss']:.2f}")
                    st.write("****************")
                
                st.session_state.new_trades = []
        
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
                    simulate_stock(stock_symbol, st.session_state.simulation_date)
        
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
                        
                        stock_data = fetch_stock_data_for_date(stock_symbol, sim_date)
                        
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
                                
                                kell_phase = df_slice.iloc[-1]['PHASE']
                                goverdhan_patterns = ', '.join([p for p in ['BULL_FLAG', 'EMA_KISS_FLY', 'HORIZONTAL_FADE', 'VCP', 'REVERSAL_SQUEEZE'] if df_slice.iloc[-1][p]])
                                unger_pattern = df_slice.iloc[-1]['UNGER_SIGNAL']
                                cook_pattern = df_slice.iloc[-1]['COOK_SIGNAL']
                                
                                result_row = {
                                    'Date': sim_date_str,
                                    'Time': format_time_12h(bar_time),
                                    'Stock': stock_symbol,
                                    'Price': current_price,
                                    'Signal': signal,
                                    'Kell Phase': kell_phase,
                                    'Goverdhan Pattern': goverdhan_patterns,
                                    'Unger Signal': unger_pattern,
                                    'Cook Signal': cook_pattern,
                                    'Option Sentiment': option_sentiment
                                }
                                
                                all_results = pd.concat([all_results, pd.DataFrame([result_row])], ignore_index=True)
                                
                                if signal != "No Signal":
                                    # Only show new trade notification in the sidebar, not in the main results
                                    with st.sidebar:
                                        st.subheader(f"*** NEW TRADE ***")
                                        st.write(f"Date: {sim_date_str}")
                                        st.write(f"Time: {format_time_12h(bar_time)} [{stock_symbol}]")
                                        st.write(f"Current Price: {current_price:.2f}")
                                        st.write(f"Direction: {direction}")
                                        
                                        signal_color = get_signal_color(signal)
                                        st.markdown(f"Signal: <span style='color:{signal_color}'>{signal}</span>", unsafe_allow_html=True)
                                        
                                        st.write(f"Adapted Bands: Lower={lower:.2f}, Upper={upper:.2f}")
                                        st.write(f"VIX: {vix:.2f}")
                                        st.write(f"Option Sentiment: {option_sentiment}")
                                        st.write(f"Kell Phase: {kell_phase}")
                                        st.write(f"Goverdhan Pattern: {goverdhan_patterns}")
                                        st.write(f"Unger Signal: {unger_pattern}")
                                        st.write(f"Cook Signal: {cook_pattern}")
                                        
                                        if buildups['bullish'] or buildups['bearish']:
                                            st.write(f"OI Buildups: Bullish={buildups['bullish']}, Bearish={buildups['bearish']}")
                                        
                                        st.write(f"Target: {target:.2f}, Stop Loss: {stop_loss:.2f}")
                                        st.write("****************")
                                    
                                    play_alert_sound("alert")
                    
                    if not all_results.empty:
                        def highlight_signals(val):
                            if isinstance(val, str):
                                color = get_signal_color(val)
                                return f'background-color: {color}'
                            return ''
                        
                        styled_results = all_results.style.applymap(highlight_signals, subset=['Signal'])
                        
                        st.dataframe(styled_results)
                        st.success("Simulation completed.")
                    else:
                        st.warning("No results found for the selected date range.")

if __name__ == "__main__":
    main()
