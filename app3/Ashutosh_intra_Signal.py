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
import pygame  # Replaced playsound with pygame
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
    # If pygame mixer initialization fails, set a flag to disable sound
    SOUND_AVAILABLE = False
    print("Warning: Audio device not available. Sound alerts will be disabled.")

# Sound alert function with error handling - updated to use pygame
def play_alert_sound(alert_type="alert"):
    # Check if sound is available before trying to play
    if not SOUND_AVAILABLE:
        return
    
    try:
        if alert_type == "alert":
            # Try to play the alert sound, but ignore if file not found
            try:
                # Get the root directory (assuming assets is in the root)
                root_dir = os.path.dirname(os.path.abspath(__file__))
                alert_path = os.path.join(root_dir, "assets", "alert.mp3")
                pygame.mixer.music.load(alert_path)
                pygame.mixer.music.play()
            except:
                pass
        elif alert_type == "warning":
            try:
                # Get the root directory (assuming assets is in the root)
                root_dir = os.path.dirname(os.path.abspath(__file__))
                warning_path = os.path.join(root_dir, "assets", "warning.mp3")
                pygame.mixer.music.load(warning_path)
                pygame.mixer.music.play()
            except:
                pass
    except Exception as e:
        pass  # Silently ignore sound errors

# Upstox API base URL
BASE_URL = "https://api.upstox.com/v2"

# Integrated API credentials (replace with your actual credentials)
API_KEY = "9a56569a-142f-4247-a863-f4e663fb03f1"  # Replace with your actual API key
ACCESS_TOKEN = "smgyr2big7"  # Replace with your actual access token

# Map symbols to Upstox instrument keys
instrument_map = {
    '^NSEI': 'NSE_INDEX|Nifty 50',
    '^BSESN': 'BSE_INDEX|Sensex',
    '^INDIAVIX': 'NSE_INDEX|India VIX'
}

# Map stock symbols to Upstox instrument keys format
def get_stock_instrument_key(symbol):
    # For NSE stocks: NSE_EQ|{ISIN}
    # For BSE stocks: BSE_EQ|{ISIN}
    # Since we don't have ISIN, we'll use a simplified approach
    return f"NSE_EQ|{symbol.upper()}"

# Convert 1-minute data to desired timeframe (5 or 15 minutes)
def convert_timeframe(df, timeframe_minutes=15):
    if df.empty:
        return df
    
    # Create a grouping key that rounds timestamps to the nearest interval
    df = df.copy()
    df['time_group'] = df.index.floor(f'{timeframe_minutes}min')
    
    # Group by the time interval and aggregate
    resampled = df.groupby('time_group').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'OI': 'last',
        'VIX': 'last'
    })
    
    # Drop the grouping column and return
    return resampled

# Generate sample data for testing when API is not available
def generate_sample_data(symbol='^NSEI', days=30):
    st.info(f"Generating sample data for {symbol} for testing purposes...")
    
    # Create date range
    end_date = datetime.now(pytz.timezone('Asia/Kolkata'))
    start_date = end_date - timedelta(days=days)
    
    # Create business day range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate sample price data
    np.random.seed(42)  # For reproducible results
    base_price = 19500  # Approximate Nifty value
    
    daily_data = []
    for date in dates:
        # Daily price change with some randomness
        change = np.random.normal(0, 0.01)  # 1% standard deviation
        open_price = base_price * (1 + change * 0.5)
        close_price = base_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        
        # Volume with some randomness
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
        base_price = close_price
    
    # Create DataFrame
    df = pd.DataFrame(daily_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Convert timezone to Asia/Kolkata
    if df.index.tz is None:
        df.index = df.index.tz_localize('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')
    
    # Rename columns to match original format
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
    
    # Add VIX data
    df['VIX'] = np.random.uniform(12, 18, size=len(df))
    
    return df

# Generate sample intraday data
def generate_sample_intraday_data(symbol='^NSEI', timeframe_minutes=15, days=15):
    st.info(f"Generating sample intraday data for {symbol} for testing purposes...")
    
    # Create date range for the last 15 days
    end_date = datetime.now(pytz.timezone('Asia/Kolkata'))
    start_date = end_date - timedelta(days=days)
    
    # Create business day range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    all_data = []
    
    for date in dates:
        # Create timestamps for market hours (9:15 to 15:30)
        start_time = datetime.combine(date, time(9, 15)).astimezone(pytz.timezone('Asia/Kolkata'))
        end_time = datetime.combine(date, time(15, 30)).astimezone(pytz.timezone('Asia/Kolkata'))
        
        # Create 1-minute timestamps
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
        
        # Generate sample price data
        np.random.seed(42)  # For reproducible results
        base_price = 19500  # Approximate Nifty value
        
        intraday_data = []
        for timestamp in timestamps:
            # 1-minute price change with some randomness
            change = np.random.normal(0, 0.0005)  # Small change for 1 minute
            open_price = base_price * (1 + change * 0.5)
            close_price = base_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0002)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0002)))
            
            # Volume with some randomness
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
            base_price = close_price
        
        all_data.extend(intraday_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Convert timezone to Asia/Kolkata
    if df.index.tz is None:
        df.index = df.index.tz_localize('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')
    
    # Rename columns to match original format
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
    
    # Add VIX data
    df['VIX'] = 15.0  # Constant VIX for intraday
    
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
    
    # Use integrated credentials if not provided
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
                
            # Format dates correctly for Upstox API
            to_date = end_date.strftime('%Y-%m-%d')
            from_date = start_date.strftime('%Y-%m-%d')
            
            # Fetch historical data - using correct interval value 'day'
            url = f"{BASE_URL}/historical-candle/{instrument_key}/day/{to_date}/{from_date}"
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    st.error(f"API returned status code {response.status_code} for {symbol}")
                    # Use sample data as fallback
                    st.info("Using sample data as fallback...")
                    df = generate_sample_data(symbol)
                    historical_data[symbol] = df
                    continue
                    
                data = response.json()
                
                if not data or 'data' not in data or 'candles' not in data['data']:
                    st.error(f"No data returned for {symbol}")
                    # Use sample data as fallback
                    st.info("Using sample data as fallback...")
                    df = generate_sample_data(symbol)
                    historical_data[symbol] = df
                    continue
                    
                # Process candles
                candles = data['data']['candles']
                if not candles:
                    st.error(f"No candles returned for {symbol}")
                    # Use sample data as fallback
                    st.info("Using sample data as fallback...")
                    df = generate_sample_data(symbol)
                    historical_data[symbol] = df
                    continue
                    
                # Create DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                
                # Fix timestamp conversion - Upstox is returning ISO format, not Unix timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Convert timezone to Asia/Kolkata if not already in that timezone
                if df.index.tz is None:
                    df.index = df.index.tz_localize('Asia/Kolkata')
                else:
                    df.index = df.index.tz_convert('Asia/Kolkata')
                
                # Rename columns to match original format
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
                
                # Fetch VIX data
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
                                
                                # Convert timezone to Asia/Kolkata if not already in that timezone
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
                
                # Drop rows with NaN VIX
                df = df.dropna(subset=['VIX'])
                if df.empty:
                    st.warning(f"No data after VIX alignment for {symbol}. Trying without alignment.")
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    
                    # Convert timezone to Asia/Kolkata if not already in that timezone
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('Asia/Kolkata')
                    else:
                        df.index = df.index.tz_convert('Asia/Kolkata')
                        
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
                    df['VIX'] = 15.0
                
                # Exclude COVID period (2020-01-01 to 2021-12-31)
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

# Fetch Intraday Data using Upstox API - updated to fetch 15 days of data
def fetch_intraday_data(index_symbols=['^NSEI', '^BSESN'], sim_date=None, api_key=None, access_token=None, timeframe_minutes=15, days=15):
    ist = pytz.timezone('Asia/Kolkata')
    intraday_data = {}
    
    # Use integrated credentials if not provided
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
            
            # Determine if it's an index or stock
            if symbol.startswith('^'):
                # It's an index
                instrument_key = instrument_map.get(symbol)
                if not instrument_key:
                    st.error(f"No instrument key found for {symbol}")
                    continue
            else:
                # It's a stock
                instrument_key = get_stock_instrument_key(symbol)
            
            # Determine date range - fetch last 15 days of data
            if sim_date:
                sim_date_dt = datetime.strptime(sim_date.strip(), '%Y-%m-%d')
                if sim_date_dt.weekday() >= 5:
                    st.error(f"{sim_date} is a weekend (not a trading day).")
                    continue
                if sim_date_dt.date() > datetime.now().date():
                    st.error(f"{sim_date} is a future date, no data available.")
                    continue
                
                # For simulation, we want data from 15 days before the simulation date
                start_date = sim_date_dt - timedelta(days=days)
                # Adjust for weekends
                while start_date.weekday() >= 5:
                    start_date = start_date - timedelta(days=1)
                
                start = start_date.replace(hour=9, minute=15).astimezone(ist)
                end = sim_date_dt.replace(hour=15, minute=30).astimezone(ist)
            else:
                # For live scanning, fetch last 15 days
                end_date = datetime.now().astimezone(ist)
                start_date = end_date - timedelta(days=days)
                # Adjust for weekends
                while start_date.weekday() >= 5:
                    start_date = start_date - timedelta(days=1)
                
                start = start_date.replace(hour=9, minute=15).astimezone(ist)
                end = end_date
            
            # Format dates correctly for Upstox API
            to_date = end.strftime('%Y-%m-%d')
            from_date = start.strftime('%Y-%m-%d')
            
            # Try to fetch intraday data using the new endpoint
            url = f"{BASE_URL}/historical-candle/intraday/{instrument_key}/1minute"
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    st.error(f"API returned status code {response.status_code} for {symbol}")
                    # Use sample data as fallback
                    st.info("Using sample data as fallback...")
                    df = generate_sample_intraday_data(symbol, timeframe_minutes, days)
                    intraday_data[symbol] = df
                    continue
                    
                data = response.json()
                
                if not data or 'data' not in data or 'candles' not in data['data']:
                    st.error(f"No data returned for {symbol}")
                    # Use sample data as fallback
                    st.info("Using sample data as fallback...")
                    df = generate_sample_intraday_data(symbol, timeframe_minutes, days)
                    intraday_data[symbol] = df
                    continue
                    
                # Process candles
                candles = data['data']['candles']
                if not candles:
                    st.error(f"No candles returned for {symbol}")
                    # Use sample data as fallback
                    st.info("Using sample data as fallback...")
                    df = generate_sample_intraday_data(symbol, timeframe_minutes, days)
                    intraday_data[symbol] = df
                    continue
                    
                # Create DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                
                # Fix timestamp conversion - Upstox is returning ISO format, not Unix timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Convert timezone to Asia/Kolkata if not already in that timezone
                if df.index.tz is None:
                    df.index = df.index.tz_localize('Asia/Kolkata')
                else:
                    df.index = df.index.tz_convert('Asia/Kolkata')
                
                # Rename columns to match original format
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
                
                # Filter to market hours
                df = df.between_time('09:15', '15:30')
                
                # Add VIX (use default for intraday)
                df['VIX'] = 15.0
                
                # Convert 1-minute data to desired timeframe
                df = convert_timeframe(df, timeframe_minutes=timeframe_minutes)
                
                # Ensure we have data for the full market hours
                df = df.between_time('09:15', '15:30')
                
                # For simulation, filter to the simulation date if provided
                if sim_date:
                    sim_date_str = sim_date.strip()
                    df = df[df.index.date == datetime.strptime(sim_date_str, '%Y-%m-%d').date()]
                
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

# Fetch stock data from Yahoo Finance - updated to fetch 15 days of data
def fetch_stock_data(symbol, period='15d', interval='1m'):
    try:
        # Append .NS suffix if not present
        if not symbol.endswith(('.NS', '.BO')):
            symbol = symbol + '.NS'  # Default to NSE
            
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            st.warning(f"No data found for {symbol}")
            return None
            
        # Reset index to make timestamp a column
        data.reset_index(inplace=True)
        
        # Convert timezone to Asia/Kolkata
        data['Datetime'] = pd.to_datetime(data['Datetime']).dt.tz_convert('Asia/Kolkata')
        data.set_index('Datetime', inplace=True)
        
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
        
        return data
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {e}")
        return None

# Compute Indicators - updated to handle small datasets
def compute_indicators(df, intraday=True):
    if df.empty or 'Close' not in df.columns:
        st.error("DataFrame is empty or missing 'Close' column.")
        return df
    
    # Adjust window sizes based on available data
    available_rows = len(df)
    
    if intraday:
        # For intraday, we want at least 20 rows, but we'll adjust if we have less
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
        # Calculate EMAs
        if period_ema_short > 0:
            df['EMA10'] = EMAIndicator(df['Close'], window=period_ema_short).ema_indicator()
        else:
            df['EMA10'] = np.nan
            
        if period_ema_long > 0:
            df['EMA20'] = EMAIndicator(df['Close'], window=period_ema_long).ema_indicator()
        else:
            df['EMA20'] = np.nan
        
        # Calculate ADX
        if period_adx > 0:
            df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close'], window=period_adx).adx()
        else:
            df['ADX'] = np.nan
            
        # Calculate ATR
        if period_adx > 0:
            df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=period_adx).average_true_range()
        else:
            df['ATR'] = np.nan
            
        # Calculate volume average
        df['VOL_AVG'] = df['Volume'].rolling(window=10).mean()
        
        # Goverdhan's EMAs
        if period_ema9 > 0:
            df['EMA9'] = EMAIndicator(df['Close'], window=period_ema9).ema_indicator()
        else:
            df['EMA9'] = np.nan
            
        if period_ema21 > 0:
            df['EMA21'] = EMAIndicator(df['Close'], window=period_ema21).ema_indicator()
        else:
            df['EMA21'] = np.nan
        
        # Andrea Unger's Bollinger Bands
        if period_bb > 0:
            bb = BollingerBands(df['Close'], window=period_bb, window_dev=2)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Mid'] = bb.bollinger_mavg()
        else:
            df['BB_Upper'] = np.nan
            df['BB_Lower'] = np.nan
            df['BB_Mid'] = np.nan
        
        # Michael Cook's VWAP
        # VWAP requires cumulative sum, so we need at least one row
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
        
        # Bull Flag Breakout
        spike_high = df['High'].iloc[i-5:i-3].max()
        spike_vol = df['Volume'].iloc[i-5:i-3].max()
        if (spike_vol > 2 * vol_avg and 
            spike_high > df['Close'].iloc[i-6] * 1.005 and  # 0.5% move
            close > ema9 and close > ema21 and
            df['Low'].iloc[i-3:i].min() > ema9 and  # Pullback stays above EMA
            high > spike_high and volume > 1.5 * vol_avg):
            df.loc[df.index[i], 'BULL_FLAG'] = True
        
        # EMA Kiss & Fly
        if (abs(close - ema9) < 0.3 * atr and  # Price near EMA9
            prev_close < ema9 and  # Previous bar below EMA
            close > open_price and  # Current bar is bullish
            volume > 1.2 * vol_avg):
            df.loc[df.index[i], 'EMA_KISS_FLY'] = True
        
        # Horizontal Fade
        if (abs(close - ema9) < 0.5 * atr and  # Price near EMA9
            df['High'].iloc[i-3:i].max() - df['Low'].iloc[i-3:i].min() < 1.5 * atr and  # Tight range
            close > ema9 and close > ema21):
            df.loc[df.index[i], 'HORIZONTAL_FADE'] = True
        
        # Intraday VCP
        if (df['High'].iloc[i-5:i].max() - df['Low'].iloc[i-5:i].min() < 2 * atr and  # Tight range
            df['Volume'].iloc[i-5:i].max() < 1.5 * vol_avg and  # Low volume
            close > ema21 and  # Above EMA21
            high > df['High'].iloc[i-5:i].max()):
            df.loc[df.index[i], 'VCP'] = True
        
        # Reversal Squeeze
        if (df['High'].iloc[i-5:i].max() - df['Low'].iloc[i-5:i].min() < 1.5 * atr and  # Tight range
            df['Close'].iloc[i-5] < df['Close'].iloc[i-10] and  # Previous down move
            close > open_price and  # Current bar is bullish
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
        
        # Volatility Expansion Breakout
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
        
        # VWAP Trend Continuation
        if (close > vwap and close > ema9 and close > ema21 and 
            abs(close - vwap) < 0.5 * df['ATR'].iloc[i] and volume > 1.2 * vol_avg):
            df.loc[df.index[i], 'COOK_SIGNAL'] = 'Buy (VWAP Bounce)'
        elif (close < vwap and close < ema9 and close < ema21 and 
              abs(close - vwap) < 0.5 * df['ATR'].iloc[i] and volume > 1.2 * vol_avg):
            df.loc[df.index[i], 'COOK_SIGNAL'] = 'Sell (VWAP Rejection)'
    
    return df

# Estimate Average Sideways Band Size
def estimate_band_size(df_daily):
    if df.empty:
        return 6.5, 0
    
    sideways_periods = []
    min_window = 10
    i = 0
    
    while i < len(df_daily) - min_window:
        window = df_daily.iloc[i:i+min_window]
        if (window['ADX'] < 20).all():
            full_window = window
            j = i + min_window
            while j < len(df_daily) and df_daily['ADX'].iloc[j] < 20:
                full_window = df_daily.iloc[i:j+1]
                j += 1
            band_pct = (full_window['High'].max() - full_window['Low'].min()) / full_window['Low'].min() * 100
            sideways_periods.append(band_pct)
            i = j
        else:
            i += 1
    
    # Remove outliers
    if sideways_periods:
        mean_pct = np.mean(sideways_periods)
        std_pct = np.std(sideways_periods)
        sideways_periods = [p for p in sideways_periods if p <= mean_pct + 2*std_pct]
    
    avg_band_pct = np.mean(sideways_periods) if sideways_periods else 6.5
    hist_atr_avg = df_daily['ATR'].mean() if not df_daily['ATR'].empty else 0
    
    return avg_band_pct, hist_atr_avg

# Fetch Option Chain
def fetch_option_chain(current_price, simulation=False, symbol='NIFTY', is_stock=False):
    if simulation:
        return {}, False, 0, 0, "Neutral"
    
    try:
        if is_stock:
            # For stocks, we'll use a simplified approach
            # In a real implementation, you would fetch from NSE or other sources
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
            
            # Calculate option sentiment based on PCR
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
    
    # Bull Flag Breakout
    if latest['BULL_FLAG']:
        if vix < 20 and total_ce > total_pe:
            goverdhan_signal = "Goverdhan: Buy CE (Bull Flag Breakout)"
    
    # EMA Kiss & Fly
    elif latest['EMA_KISS_FLY']:
        if vix < 20 and total_ce > total_pe:
            goverdhan_signal = "Goverdhan: Buy CE (EMA Kiss & Fly)"
    
    # Horizontal Fade
    elif latest['HORIZONTAL_FADE']:
        if 15 < vix < 20:
            goverdhan_signal = "Goverdhan: Buy CE/PE (Horizontal Fade)"
    
    # Intraday VCP
    elif latest['VCP']:
        if vix < 20 and total_ce > total_pe:
            goverdhan_signal = "Goverdhan: Buy CE (VCP Breakout)"
    
    # Reversal Squeeze
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
    
    # Map index symbol to option chain symbol
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
    # Check if we have enough data
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
    
    # Map index symbol to option chain symbol
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
    
    # Market sentiment conditions
    market_sentiment_bullish = total_ce > total_pe and vix < 20
    market_sentiment_bearish = total_pe > total_ce and vix > 20
    
    # Generate combined signal
    signals = [kell_signal, goverdhan_signal, unger_signal, cook_signal]
    valid_signals = [s for s in signals if s != "No Kell Signal" and s != "No Goverdhan Signal" and s != "No Unger Signal" and s != "No Cook Signal"]
    
    if valid_signals:
        signal = " | ".join(valid_signals)
        
        # Apply market sentiment conditions
        if "Buy CE" in signal and not market_sentiment_bullish:
            signal = "No Signal (Market sentiment not bullish)"
        elif "Buy PE" in signal and not market_sentiment_bearish:
            signal = "No Signal (Market sentiment not bearish)"
        else:
            # Set trade type based on signal
            if "Buy CE" in signal:
                trade_type = "Buy CE"
                target = current_price + target_points
                stop_loss = current_price - stop_points
            elif "Buy PE" in signal:
                trade_type = "Buy PE"
                target = current_price - target_points
                stop_loss = current_price + stop_points
    
    # Add OI buildup info
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
            
            return headlines[:5]  # Return top 5 headlines
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
        return "forestgreen"
    elif "Buy PE" in signal or "Short" in signal:
        return "red"
    else:
        return "yellow"

# Get background color for sentiment
def get_sentiment_bg_color(sentiment):
    if "Bullish" in sentiment or "Long" in sentiment:
        return "#90EE90"  # Light green
    elif "Bearish" in sentiment or "Short" in sentiment:
        return "#FFB6C1"  # Light red
    else:
        return "#FFFFE0"  # Light yellow

# Format signal output as a table with background colors
def format_signal_table(kell_signal, goverdhan_signal, unger_signal, cook_signal, option_sentiment, final_signal, target, stop_loss):
    # Create a DataFrame for better formatting
    data = {
        'Oliver Kell': [kell_signal],
        'Goverdhan': [goverdhan_signal],
        'Option Sentiment': [option_sentiment],
        'Andrea Unger': [unger_signal],
        'Michael Cook': [cook_signal],
        'Final Recommendation': [final_signal],
        'Target': [target],
        'Stop Loss': [stop_loss]
    }
    
    df = pd.DataFrame(data)
    
    # Apply background colors based on signal type
    styles = [
        {
            'selector': 'thead',
            'props': [('background-color', '#f0f0f0'), ('color', 'black')]
        }
    ]
    
    # Apply background colors to each cell based on content
    for i, col in enumerate(df.columns):
        cell_value = df[col].iloc[0]
        styles.append({
            'selector': f'td:nth-child({i+1})',
            'props': [('background-color', get_sentiment_bg_color(cell_value))]
        })
    
    # Style the DataFrame
    styled_df = df.style.set_table_styles(styles)
    
    return styled_df

# Simulate Day with Monitoring - updated to use 15 days of data
def simulate_day(sim_date, df_daily_dict, avg_band_pct_dict, hist_atr_avg_dict, api_key=None, access_token=None, timeframe_minutes=15):
    # Fetch 15 days of intraday data
    intraday_data_dict = fetch_intraday_data(list(df_daily_dict.keys()), sim_date, api_key, access_token, timeframe_minutes, days=15)
    
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
        
        # Display results as a table
        st.dataframe(pd.DataFrame(results))
        st.success("Simulation completed.")

# Simulate Date Range - updated to use 15 days of data
def simulate_date_range(start_date, end_date, df_daily_dict, avg_band_pct_dict, hist_atr_avg_dict, api_key=None, access_token=None, timeframe_minutes=15):
    # Convert string dates to datetime objects
    start_date_dt = datetime.strptime(start_date.strip(), '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date.strip(), '%Y-%m-%d')
    
    # Create a list of dates in the range
    date_range = pd.date_range(start=start_date_dt, end=end_date_dt, freq='B')  # Business days only
    
    # Create a DataFrame to store all results
    all_results = pd.DataFrame()
    
    for sim_date in date_range:
        sim_date_str = sim_date.strftime('%Y-%m-%d')
        st.write(f"Simulating {sim_date_str}...")
        
        # Fetch 15 days of intraday data
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
                
                # Get the latest patterns for display
                kell_phase = df_slice.iloc[-1]['PHASE']
                goverdhan_patterns = ', '.join([p for p in ['BULL_FLAG', 'EMA_KISS_FLY', 'HORIZONTAL_FADE', 'VCP', 'REVERSAL_SQUEEZE'] if df_slice.iloc[-1][p]])
                unger_pattern = df_slice.iloc[-1]['UNGER_SIGNAL']
                cook_pattern = df_slice.iloc[-1]['COOK_SIGNAL']
                
                # Add to results - convert time to 12-hour format
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
                
                # Add to all results
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
                    
                    st.subheader(f"*** NEW TRADE ***")
                    st.write(f"Date: {sim_date_str}")
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
    
    # Display all results as a table
    if not all_results.empty:
        st.dataframe(all_results)
        st.success("Simulation completed.")
    else:
        st.warning("No results found for the selected date range.")

# Thread function for live scanning - updated to use 15 days of data
def live_scanning_thread():
    # Determine which symbols to scan
    symbols_to_scan = ['^NSEI', '^BSESN']
    if st.session_state.stock_symbol:
        symbols_to_scan.append(st.session_state.stock_symbol)
    
    # Fetch historical data for indices
    df_daily_dict = fetch_historical_data(['^NSEI', '^BSESN'])
    if not df_daily_dict:
        st.error("Exiting due to empty historical data.")
        return
    
    # Compute indicators and estimate band sizes
    avg_band_pct_dict = {}
    hist_atr_avg_dict = {}
    for index_symbol, df_daily in df_daily_dict.items():
        df_daily = compute_indicators(df_daily, intraday=False)
        df_daily = df_daily.dropna()
        avg_band_pct, hist_atr_avg = estimate_band_size(df_daily)
        avg_band_pct_dict[index_symbol] = avg_band_pct
        hist_atr_avg_dict[index_symbol] = hist_atr_avg
    
    # Initialize state for each symbol
    symbol_states = {}
    for symbol in symbols_to_scan:
        symbol_states[symbol] = {
            'active_trade': None,
            'trade_entry_price': 0,
            'trade_type': None,
            'stop_loss': 0,
            'prev_strikes_oi': {}
        }
    
    # Run the live scanning loop
    while st.session_state.live_scanning:
        if is_market_open():
            # Fetch 15 days of intraday data for indices
            intraday_data_dict = fetch_intraday_data(['^NSEI', '^BSESN'], timeframe_minutes=st.session_state.timeframe_minutes, days=15)
            
            # Fetch stock data if provided
            if st.session_state.stock_symbol:
                stock_data = fetch_stock_data(st.session_state.stock_symbol, period='15d', interval='1m')
                if stock_data is not None and not stock_data.empty:
                    # Convert to desired timeframe
                    stock_data = convert_timeframe(stock_data, timeframe_minutes=st.session_state.timeframe_minutes)
                    intraday_data_dict[st.session_state.stock_symbol] = stock_data
            
            # Create a DataFrame to display the results
            results = []
            
            for symbol in symbols_to_scan:
                if symbol in intraday_data_dict and len(intraday_data_dict[symbol]) >= 5:
                    state = symbol_states[symbol]
                    avg_band_pct = avg_band_pct_dict.get(symbol, 6.5)
                    hist_atr_avg = hist_atr_avg_dict.get(symbol, 0)
                    
                    direction, signal, upper, lower, vix, bearish_oi, buildups, current_price, kell_signal, goverdhan_signal, unger_signal, cook_signal, target, stop_loss, trade_type, option_sentiment = generate_signals(
                        intraday_data_dict[symbol], avg_band_pct, hist_atr_avg, symbol, prev_strikes_oi=state['prev_strikes_oi']
                    )
                    
                    # Add to results
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
                        
                        # Store the new trade in session state for display
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
                        
                        # Add to session state
                        if 'new_trades' not in st.session_state:
                            st.session_state.new_trades = []
                        st.session_state.new_trades.append(new_trade)
                        
                        # Play sound alert
                        play_alert_sound("alert")
                    
                    state['prev_strikes_oi'] = {}
            
            # Update session state with results
            st.session_state.live_results = results
            
            # Sleep for a short interval before next update
            tm.sleep(st.session_state.timeframe_minutes * 60)
        else:
            # Update session state with market closed message
            st.session_state.market_closed = True
            tm.sleep(60)  # Check every minute if market is open

# Main Streamlit App
def main():
    st.set_page_config(
        page_title="Intraday Trading Signal Scanner made By Ashutosh",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Intraday Trading Signal Scanner made By Ashutosh")
    st.markdown("Scan for trading signals using multiple strategies: Oliver Kell, Goverdhan Gajjala, Andrea Unger, and Michael Cook")
    
    # Initialize session state variables
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
    
    # Sidebar for options
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        "Select an option",
        ["Live Scanning", "Simulation"]
    )
    
    # Timeframe selection
    timeframe_minutes = st.sidebar.selectbox(
        "Timeframe (minutes)",
        [5, 15],
        index=1  # Default to 15 minutes
    )
    st.session_state.timeframe_minutes = timeframe_minutes
    
    # Option 1: Live Scanning
    if option == "Live Scanning":
        st.subheader("Live Scanning")
        
        # Stock symbol input
        stock_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS) - Leave empty to scan only indices", value=st.session_state.stock_symbol)
        st.session_state.stock_symbol = stock_symbol
        
        # Start/Stop button
        if st.button("Start Live Scanning" if not st.session_state.live_scanning else "Stop Live Scanning"):
            st.session_state.live_scanning = not st.session_state.live_scanning
            
            if st.session_state.live_scanning:
                # Start the live scanning thread
                st.session_state.live_thread = threading.Thread(target=live_scanning_thread, daemon=True)
                st.session_state.live_thread.start()
                st.success("Live scanning started in background!")
            else:
                st.success("Live scanning stopped!")
        
        # Create a table for live results
        if st.session_state.live_scanning:
            st.subheader("Live Signal Results")
            
            # Create a placeholder for the live results table
            results_placeholder = st.empty()
            
            # Create columns for the instruments
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("Nifty")
                nifty_placeholder = st.empty()
            
            with col2:
                st.write("Sensex")
                sensex_placeholder = st.empty()
            
            with col3:
                if stock_symbol:
                    st.write(stock_symbol)
                    stock_placeholder = st.empty()
                else:
                    st.write("Stock")
                    stock_placeholder = st.empty()
            
            # Update the placeholders with the latest results
            if st.session_state.live_results:
                # Create a DataFrame for the table
                table_data = []
                
                # Get the latest result for each symbol
                symbols = ['^NSEI', '^BSESN']
                if stock_symbol:
                    symbols.append(stock_symbol)
                
                for symbol in symbols:
                    symbol_results = [r for r in st.session_state.live_results if r['Symbol'] == symbol]
                    if symbol_results:
                        latest_result = symbol_results[-1]
                        
                        # Create a formatted signal table
                        signal_table = format_signal_table(
                            latest_result['Kell Signal'],
                            latest_result['Goverdhan Signal'],
                            latest_result['Unger Signal'],
                            latest_result['Cook Signal'],
                            latest_result['Option Sentiment'],
                            latest_result['Signal'],
                            latest_result['Target'],
                            latest_result['Stop Loss']
                        )
                        
                        # Add to table data
                        table_data.append({
                            'Symbol': symbol,
                            'Time': latest_result['Time'],
                            'Price': latest_result['Price'],
                            'Signal Table': signal_table
                        })
                
                # Display the table
                if table_data:
                    # Create a DataFrame for display
                    display_df = pd.DataFrame(table_data)
                    
                    # Display the table in the appropriate column
                    for _, row in display_df.iterrows():
                        if row['Symbol'] == '^NSEI':
                            nifty_placeholder.write(f"Time: {row['Time']}")
                            nifty_placeholder.write(f"Price: {row['Price']:.2f}")
                            nifty_placeholder.dataframe(row['Signal Table'])
                        elif row['Symbol'] == '^BSESN':
                            sensex_placeholder.write(f"Time: {row['Time']}")
                            sensex_placeholder.write(f"Price: {row['Price']:.2f}")
                            sensex_placeholder.dataframe(row['Signal Table'])
                        else:
                            stock_placeholder.write(f"Time: {row['Time']}")
                            stock_placeholder.write(f"Price: {row['Price']:.2f}")
                            stock_placeholder.dataframe(row['Signal Table'])
                else:
                    # Display "Fetching data..." message
                    nifty_placeholder.write("Fetching data...")
                    sensex_placeholder.write("Fetching data...")
                    if stock_symbol:
                        stock_placeholder.write("Fetching data...")
            
            # Display new trades if available
            if st.session_state.new_trades:
                st.subheader("New Trades")
                for trade in st.session_state.new_trades:
                    st.subheader(f"*** NEW TRADE ***")
                    st.write(f"Time: {trade['time']} [{trade['symbol']}]")
                    st.write(f"Current Price: {trade['current_price']:.2f}")
                    st.write(f"Direction: {trade['direction']}")
                    
                    # Display signal with color
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
                
                # Clear new trades after displaying
                st.session_state.new_trades = []
        
        # Display market status
        if st.session_state.market_closed:
            st.warning("Market is closed. Waiting for market to open...")
            st.session_state.market_closed = False
        
        # Fetch and display market sentiment
        st.subheader("Market Sentiment")
        headlines = fetch_market_sentiment()
        if headlines:
            for headline in headlines:
                st.write(f"- {headline}")
        else:
            st.write("No market sentiment data available.")
    
    # Option 2: Simulation
    elif option == "Simulation":
        st.subheader("Simulation Mode")
        
        # Simulation type selection
        sim_type = st.radio("Select simulation type", ["Single Date", "Date Range"])
        
        if sim_type == "Single Date":
            # Date input
            sim_date = st.date_input("Select a date to simulate", value=datetime.now().date())
            st.session_state.simulation_date = sim_date.strftime('%Y-%m-%d')
            
            # Index/Stock selection
            selection_type = st.radio("Select simulation target", ["Indices", "Individual Stock"])
            
            if selection_type == "Indices":
                index_options = ['Nifty (^NSEI)', 'Sensex (^BSESN)']
                selected_index = st.selectbox("Select an index", index_options)
                index_symbol = '^NSEI' if selected_index == 'Nifty (^NSEI)' else '^BSESN'
                
                # Run simulation button
                if st.button("Run Simulation"):
                    # Fetch historical data
                    df_daily_dict = fetch_historical_data([index_symbol])
                    if not df_daily_dict:
                        st.error("Exiting due to empty historical data.")
                        return
                    
                    # Compute indicators and estimate band sizes
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
                    
                    # Run simulation
                    simulate_day(st.session_state.simulation_date, df_daily_dict, avg_band_pct_dict, hist_atr_avg_dict, timeframe_minutes=timeframe_minutes)
            
            else:  # Individual Stock
                stock_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS)", value="")
                
                if stock_symbol and st.button("Run Simulation"):
                    # Fetch 15 days of stock data
                    stock_data = fetch_stock_data(stock_symbol, period='15d', interval='1m')
                    
                    if stock_data is not None and not stock_data.empty:
                        # Compute indicators
                        stock_data = compute_indicators(stock_data)
                        stock_data = compute_kell_phases(stock_data)
                        stock_data = compute_goverdhan_patterns(stock_data)
                        stock_data = compute_unger_strategy(stock_data)
                        stock_data = compute_cook_strategy(stock_data)
                        
                        # Create a DataFrame to store the results
                        results = []
                        
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
                        
                        # Display results as a table
                        st.dataframe(pd.DataFrame(results))
                        st.success("Simulation completed.")
                    else:
                        st.error(f"No data found for {stock_symbol}")
        
        else:  # Date Range
            # Date range input
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date", value=datetime.now().date() - timedelta(days=7))
            with col2:
                end_date = st.date_input("End date", value=datetime.now().date())
            
            # Index/Stock selection
            selection_type = st.radio("Select simulation target", ["Indices", "Individual Stock"], key="date_range_selection")
            
            if selection_type == "Indices":
                index_options = ['Nifty (^NSEI)', 'Sensex (^BSESN)']
                selected_index = st.selectbox("Select an index", index_options, key="date_range_index")
                index_symbol = '^NSEI' if selected_index == 'Nifty (^NSEI)' else '^BSESN'
                
                # Run simulation button
                if st.button("Run Simulation", key="run_date_range"):
                    # Fetch historical data
                    df_daily_dict = fetch_historical_data([index_symbol])
                    if not df_daily_dict:
                        st.error("Exiting due to empty historical data.")
                        return
                    
                    # Compute indicators and estimate band sizes
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
                    
                    # Run simulation
                    simulate_date_range(
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d'), 
                        df_daily_dict, 
                        avg_band_pct_dict, 
                        hist_atr_avg_dict, 
                        timeframe_minutes=timeframe_minutes
                    )
            
            else:  # Individual Stock
                stock_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS)", value="", key="date_range_stock")
                
                if stock_symbol and st.button("Run Simulation", key="run_date_range_stock"):
                    # For individual stocks, we'll fetch data for each day in the range
                    start_date_dt = datetime.strptime(start_date.strftime('%Y-%m-%d'), '%Y-%m-%d')
                    end_date_dt = datetime.strptime(end_date.strftime('%Y-%m-%d'), '%Y-%m-%d')
                    
                    # Create a list of dates in the range
                    date_range = pd.date_range(start=start_date_dt, end=end_date_dt, freq='B')  # Business days only
                    
                    # Create a DataFrame to store all results
                    all_results = pd.DataFrame()
                    
                    for sim_date in date_range:
                        sim_date_str = sim_date.strftime('%Y-%m-%d')
                        st.write(f"Processing {sim_date_str}...")
                        
                        # Fetch 15 days of stock data
                        stock_data = fetch_stock_data(stock_symbol, period='15d', interval='1m')
                        
                        if stock_data is not None and not stock_data.empty:
                            # Compute indicators
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
                                
                                # Get the latest patterns for display
                                kell_phase = df_slice.iloc[-1]['PHASE']
                                goverdhan_patterns = ', '.join([p for p in ['BULL_FLAG', 'EMA_KISS_FLY', 'HORIZONTAL_FADE', 'VCP', 'REVERSAL_SQUEEZE'] if df_slice.iloc[-1][p]])
                                unger_pattern = df_slice.iloc[-1]['UNGER_SIGNAL']
                                cook_pattern = df_slice.iloc[-1]['COOK_SIGNAL']
                                
                                # Add to results - convert time to 12-hour format
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
                                
                                # Add to all results
                                all_results = pd.concat([all_results, pd.DataFrame([result_row])], ignore_index=True)
                                
                                if signal != "No Signal":
                                    st.subheader(f"*** NEW TRADE ***")
                                    st.write(f"Date: {sim_date_str}")
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
                    
                    # Display all results as a table
                    if not all_results.empty:
                        st.dataframe(all_results)
                        st.success("Simulation completed.")
                    else:
                        st.warning("No results found for the selected date range.")

if __name__ == "__main__":
    main()
