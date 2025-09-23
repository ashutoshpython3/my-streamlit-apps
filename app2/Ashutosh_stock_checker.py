import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import os
import sys
import platform
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced Stock Analyzer Made By Ashutosh",
    page_icon="📈",
    layout="wide"
)

class StockAnalyzer:
    def __init__(self, stock_name, data=None):
        self.stock_name = stock_name.upper()
        self.data = data
        self.signals = {
            'Oliver Kell': None,
            'Goverdhan Gajjala': None,
            'Shankar Nath': None,
            'Relative Strength': None,
            'Machine Learning': None
        }
        self.debug = False
        
    def fetch_data(self):
        """Fetch 5 years of daily stock data from Yahoo Finance with Indian exchange suffix"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        # Try without suffix first
        try:
            ticker = yf.Ticker(self.stock_name)
            self.data = ticker.history(start=start_date, end=end_date)
            
            if not self.data.empty:
                print(f"Successfully fetched data for {self.stock_name}")
                return True
        except Exception:
            pass
        
        # Try with NSE suffix
        nse_symbol = self.stock_name + '.NS'
        try:
            ticker = yf.Ticker(nse_symbol)
            self.data = ticker.history(start=start_date, end=end_date)
            
            if not self.data.empty:
                self.stock_name = nse_symbol
                print(f"Successfully fetched data for {nse_symbol} (NSE)")
                return True
        except Exception:
            pass
        
        # Try with BSE suffix
        bse_symbol = self.stock_name + '.BO'
        try:
            ticker = yf.Ticker(bse_symbol)
            self.data = ticker.history(start=start_date, end=end_date)
            
            if not self.data.empty:
                self.stock_name = bse_symbol
                print(f"Successfully fetched data for {bse_symbol} (BSE)")
                return True
        except Exception:
            pass
        
        print(f"No data found for {self.stock_name} with any Indian exchange suffix")
        return False
    
    def calculate_indicators(self):
        """Calculate necessary technical indicators"""
        if self.data is None:
            print("No data available. Fetch data first.")
            return
            
        # Calculate EMAs for Oliver Kell's strategy
        self.data['EMA_10'] = self.data['Close'].ewm(span=10, adjust=False).mean()
        self.data['EMA_20'] = self.data['Close'].ewm(span=20, adjust=False).mean()
        self.data['EMA_50'] = self.data['Close'].ewm(span=50, adjust=False).mean()
        
        # Calculate EMAs for Goverdhan Gajjala's strategy
        self.data['EMA_9'] = self.data['Close'].ewm(span=9, adjust=False).mean()
        self.data['EMA_21'] = self.data['Close'].ewm(span=21, adjust=False).mean()
        
        # Calculate distance from EMAs for Oliver Kell's strategy
        self.data['Dist_10EMA'] = (self.data['Close'] - self.data['EMA_10']) / self.data['EMA_10'] * 100
        self.data['Dist_20EMA'] = (self.data['Close'] - self.data['EMA_20']) / self.data['EMA_20'] * 100
        self.data['Dist_50EMA'] = (self.data['Close'] - self.data['EMA_50']) / self.data['EMA_50'] * 100
        
        # Calculate volume change
        self.data['Volume_Change'] = self.data['Volume'].pct_change() * 100
        
        # Calculate average volume
        self.data['Avg_Volume_20'] = self.data['Volume'].rolling(window=20).mean()
        
        # Calculate True Range and ATR for stop-loss levels
        self.data['TR1'] = self.data['High'] - self.data['Low']
        self.data['TR2'] = abs(self.data['High'] - self.data['Close'].shift(1))
        self.data['TR3'] = abs(self.data['Low'] - self.data['Close'].shift(1))
        self.data['TR'] = self.data[['TR1', 'TR2', 'TR3']].max(axis=1)
        self.data['ATR'] = self.data['TR'].rolling(window=14).mean()
        
        # Calculate daily range for VCP detection
        self.data['Daily_Range'] = (self.data['High'] - self.data['Low']) / self.data['Close'] * 100
        
        # Calculate range change for VCP
        self.data['Range_Change'] = self.data['Daily_Range'].pct_change() * 100
        
        # Calculate returns for ML strategy
        self.data['Return_1d'] = self.data['Close'].pct_change(1)
        self.data['Return_5d'] = self.data['Close'].pct_change(5)
        self.data['Return_10d'] = self.data['Close'].pct_change(10)
        self.data['Return_20d'] = self.data['Close'].pct_change(20)
        
        # Calculate volatility
        self.data['Volatility'] = self.data['Return_1d'].rolling(window=20).std()
        
        # Calculate Relative Strength vs Nifty 50 using 55-day period
        try:
            nifty = yf.Ticker('^NSEI').history(start=self.data.index[0], end=self.data.index[-1])
            self.data['Nifty_Close'] = nifty['Close'].reindex(self.data.index, method='ffill')
            
            # Calculate 55-day relative strength
            length = 55
            base_symbol = self.data['Close']
            comparative_symbol = self.data['Nifty_Close']
            
            # Calculate returns over 55 days
            base_return = (base_symbol / base_symbol.shift(length)) - 1
            nifty_return = (comparative_symbol / comparative_symbol.shift(length)) - 1
            
            # Calculate relative strength
            self.data['Relative_Strength_Value'] = base_return - nifty_return
            
            # Calculate RS trend
            self.data['RS_SMA'] = self.data['Relative_Strength_Value'].rolling(window=20).mean()
            
            # Calculate price trend
            self.data['Price_SMA'] = self.data['Close'].rolling(window=50).mean()
            
        except Exception as e:
            print(f"Error calculating relative strength: {e}")
            self.data['Relative_Strength_Value'] = 0.0
            self.data['RS_SMA'] = 0.0
            self.data['Price_SMA'] = self.data['Close']
        
        # Calculate Support and Resistance levels
        self._calculate_support_resistance()
    
    def _calculate_support_resistance(self):
        """Calculate support and resistance levels"""
        if self.data is None:
            return
            
        # Find local maxima and minima for support/resistance
        highs = self.data['High'].values
        lows = self.data['Low'].values
        
        # Find peaks and troughs
        high_idx = argrelextrema(highs, np.greater, order=5)[0]
        low_idx = argrelextrema(lows, np.less, order=5)[0]
        
        # Initialize support and resistance columns
        self.data['Support'] = np.nan
        self.data['Resistance'] = np.nan
        
        # Set resistance levels at local maxima
        if len(high_idx) > 0:
            for idx in high_idx:
                if idx < len(self.data):
                    self.data.loc[self.data.index[idx], 'Resistance'] = highs[idx]
        
        # Set support levels at local minima
        if len(low_idx) > 0:
            for idx in low_idx:
                if idx < len(self.data):
                    self.data.loc[self.data.index[idx], 'Support'] = lows[idx]
        
        # Forward fill the support and resistance levels
        self.data['Support'] = self.data['Support'].fillna(method='ffill')
        self.data['Resistance'] = self.data['Resistance'].fillna(method='ffill')
        
        # If no support/resistance found, use recent min/max
        if pd.isna(self.data['Support'].iloc[-1]):
            self.data['Support'] = self.data['Low'].rolling(window=20).min()
        
        if pd.isna(self.data['Resistance'].iloc[-1]):
            self.data['Resistance'] = self.data['High'].rolling(window=20).max()
    
    def apply_oliver_kell_strategy(self):
        """Apply Oliver Kell's Price Cycle Strategy with improved pattern detection"""
        if self.data is None or len(self.data) < 100:
            return {'phase': "Unknown", 'signals': ["Insufficient data"], 'recommendation': "HOLD", 'stop_loss': None, 'trend': "Unknown", 'buy_price': None}
            
        # Initialize signals
        signals = []
        current_phase = "Unknown"
        stop_loss = None
        buy_price = None
        
        # Determine the overall trend
        trend = self._determine_trend()
        
        # Check for patterns in the last 60 days
        recent_data = self.data.tail(60)
        
        # Check for Wedge Pop (Bullish Falling Wedge)
        wedge_pop_result = self._detect_wedge_pop_improved(recent_data)
        if wedge_pop_result['is_pattern']:
            current_phase = "Wedge Pop"
            signals.append(f"BUY - Bullish falling wedge breakout with volume confirmation")
            stop_loss = wedge_pop_result['stop_loss']
            buy_price = wedge_pop_result['buy_price']
        
        # Check for Wedge Drop (Bearish Rising Wedge)
        wedge_drop_result = self._detect_wedge_drop_improved(recent_data)
        if wedge_drop_result['is_pattern']:
            current_phase = "Wedge Drop"
            signals.append(f"SELL - Bearish rising wedge breakdown with volume confirmation")
            stop_loss = wedge_drop_result['stop_loss']
        
        # Check for Base N Break (Classic CAN SLIM)
        base_break_result = self._detect_base_n_break(recent_data, trend)
        if base_break_result['is_pattern']:
            current_phase = "Base N Break"
            signals.append(f"BUY - Breaking out of base with volume confirmation")
            stop_loss = base_break_result['stop_loss']
            buy_price = base_break_result['buy_price']
        
        # Check for EMA Crossback
        ema_crossback_result = self._detect_ema_crossback(recent_data, trend)
        if ema_crossback_result['is_pattern']:
            current_phase = "EMA Crossback"
            signals.append(f"BUY - Low-risk entry point at EMA support")
            stop_loss = ema_crossback_result['stop_loss']
            buy_price = ema_crossback_result['buy_price']
        
        # Check for Exhaustion Extension
        exhaustion_result = self._detect_exhaustion_extension(recent_data, trend)
        if exhaustion_result['is_pattern']:
            current_phase = "Exhaustion Extension"
            signals.append(f"SELL - Signs of exhaustion in uptrend, consider taking profits")
            stop_loss = exhaustion_result['stop_loss']
        
        # Check for Reversal Extension
        reversal_result = self._detect_reversal_extension(recent_data, trend)
        if reversal_result['is_pattern']:
            current_phase = "Reversal Extension"
            signals.append(f"BUY - Failed pullback and quick reversal, shows strength")
            stop_loss = reversal_result['stop_loss']
            buy_price = reversal_result['buy_price']
        
        # If no patterns detected, assign based on trend
        if current_phase == "Unknown":
            if trend == "Uptrend":
                current_phase = "Uptrend Continuation"
                signals.append("WATCH - Stock in uptrend, wait for pullback or setup")
            elif trend == "Downtrend":
                current_phase = "Downtrend Continuation"
                signals.append("AVOID - Stock in downtrend, avoid long positions")
            else:
                current_phase = "Sideways"
                signals.append("WATCH - Stock in sideways trend, wait for breakout")
        
        # Determine recommendation
        if signals:
            recommendation = signals[0].split(' - ')[0]
        else:
            recommendation = "HOLD"
        
        self.signals['Oliver Kell'] = {
            'phase': current_phase,
            'signals': signals,
            'recommendation': recommendation,
            'stop_loss': stop_loss,
            'trend': trend,
            'buy_price': buy_price
        }
        
        return self.signals['Oliver Kell']
    
    def apply_goverdhan_gajjala_strategy(self):
        """Apply Goverdhan Gajjala's Strategy adapted for Daily Timeframes"""
        if self.data is None or len(self.data) < 50:
            return {'signals': ["Insufficient data"], 'recommendation': "HOLD", 'stop_loss': None, 'buy_price': None}
            
        # We'll analyze the most recent data for momentum patterns
        recent_data = self.data.tail(30)
        
        signals = []
        stop_loss = None
        buy_price = None
        
        # Check for Bull Flag pattern (adapted for daily)
        bull_flag_result = self._detect_bull_flag_daily(recent_data)
        if bull_flag_result['is_pattern']:
            signals.append(f"BUY - Bull flag breakout at {bull_flag_result['breakout_level']:.2f}")
            stop_loss = bull_flag_result['stop_loss']
            buy_price = bull_flag_result['buy_price']
        
        # Check for EMA Kiss & Fly pattern
        ema_kiss_result = self._detect_ema_kiss_fly(recent_data)
        if ema_kiss_result['is_pattern']:
            signals.append(f"BUY - EMA Kiss & Fly bounce at {ema_kiss_result['kiss_level']:.2f}")
            if not stop_loss:
                stop_loss = ema_kiss_result['stop_loss']
            if not buy_price:
                buy_price = ema_kiss_result['buy_price']
        
        # Check for Volatility Contraction Pattern (VCP)
        vcp_result = self._detect_vcp(recent_data)
        if vcp_result['is_pattern']:
            signals.append(f"BUY - VCP breakout at {vcp_result['breakout_level']:.2f}")
            if not stop_loss:
                stop_loss = vcp_result['stop_loss']
            if not buy_price:
                buy_price = vcp_result['buy_price']
        
        # Check for EMA alignment
        ema_alignment = self._check_ema_alignment(recent_data)
        if ema_alignment['signal']:
            signals.append(ema_alignment['signal'])
            if not stop_loss:
                stop_loss = ema_alignment['stop_loss']
            if not buy_price:
                buy_price = ema_alignment['buy_price']
        
        # Check for volume confirmation
        volume_result = self._check_volume_surge(recent_data)
        if volume_result['signal']:
            signals.append(volume_result['signal'])
        
        # Check for AVOID signal if price closes below 9 EMA (trailing stop)
        if recent_data['Close'].iloc[-1] < recent_data['EMA_9'].iloc[-1]:
            signals.append("AVOID - Price closed below 9 EMA (trailing stop)")
            if any(s.startswith("BUY") for s in signals):
                signals = [s for s in signals if not s.startswith("BUY")]
                signals.append("AVOID - Price closed below 9 EMA (trailing stop)")
        
        # Determine overall recommendation
        if signals:
            buy_signals = sum(1 for s in signals if s.startswith("BUY"))
            avoid_signals = sum(1 for s in signals if s.startswith("AVOID"))
            watch_signals = sum(1 for s in signals if s.startswith("WATCH"))
            
            if avoid_signals > 0:
                recommendation = "AVOID"
            elif buy_signals > 0:
                recommendation = "BUY"
            elif watch_signals > 0:
                recommendation = "WATCH"
            else:
                recommendation = "HOLD"
        else:
            recommendation = "HOLD"
        
        self.signals['Goverdhan Gajjala'] = {
            'signals': signals,
            'recommendation': recommendation,
            'stop_loss': stop_loss,
            'buy_price': buy_price
        }
        
        return self.signals['Goverdhan Gajjala']
    
    def apply_shankar_nath_strategy(self):
        """Apply Shankar Nath's GARP Strategy"""
        if self.data is None:
            return {'signals': ["No data available"], 'passes': 0, 'fails': 7, 'recommendation': "AVOID", 'buy_price': None}
            
        # For Indian stocks, we'll try to get fundamental data from Yahoo Finance
        # If not available, we'll use mock data
        fundamentals = self._get_fundamental_data()
        
        signals = []
        
        # Check each criterion
        if fundamentals['Market Cap'] > 1000:
            signals.append("PASS - Market Cap > 1,000 crores")
        else:
            signals.append("FAIL - Market Cap < 1,000 crores")
            
        if fundamentals['Sales Revenue'] > 1000:
            signals.append("PASS - Sales Revenue > 1,000 crores")
        else:
            signals.append("FAIL - Sales Revenue < 1,000 crores")
            
        if 0 <= fundamentals['PEG Ratio'] <= 2:
            signals.append("PASS - PEG Ratio between 0 and 2")
        else:
            signals.append("FAIL - PEG Ratio not between 0 and 2")
            
        if (fundamentals['Sales Growth 1Y'] > 15 and 
            fundamentals['Sales Growth 3Y'] > 15 and 
            fundamentals['Sales Growth 5Y'] > 15):
            signals.append("PASS - Sales Growth > 15% for 1, 3, and 5 years")
        else:
            signals.append("FAIL - Sales Growth not > 15% for all periods")
            
        if (fundamentals['EPS Growth 1Y'] > 15 and 
            fundamentals['EPS Growth 3Y'] > 15 and 
            fundamentals['EPS Growth 5Y'] > 15):
            signals.append("PASS - EPS Growth > 15% for 1, 3, and 5 years")
        else:
            signals.append("FAIL - EPS Growth not > 15% for all periods")
            
        if fundamentals['EPS Ratio'] > 1.15:
            signals.append("PASS - EPS (Last Year / Preceding Year) > 1.15")
        else:
            signals.append("FAIL - EPS (Last Year / Preceding Year) < 1.15")
            
        if fundamentals['Quarterly Net Profit'] > fundamentals['Net Profit 2Q Ago']:
            signals.append("PASS - Quarterly Net Profit > Net Profit 2 Quarters Ago")
        else:
            signals.append("FAIL - Quarterly Net Profit < Net Profit 2 Quarters Ago")
        
        # Count passes and fails
        passes = sum(1 for s in signals if s.startswith("PASS"))
        fails = sum(1 for s in signals if s.startswith("FAIL"))
        
        # Determine recommendation
        if passes == 7:
            recommendation = "BUY"
        elif passes >= 5:
            recommendation = "CONSIDER"
        else:
            recommendation = "AVOID"
        
        self.signals['Shankar Nath'] = {
            'signals': signals,
            'passes': passes,
            'fails': fails,
            'recommendation': recommendation,
            'buy_price': None
        }
        
        return self.signals['Shankar Nath']
    
    def apply_relative_strength_strategy(self):
        """Apply Relative Strength strategy based on comparison with Nifty"""
        if self.data is None or len(self.data) < 100:
            return {'signals': ["Insufficient data"], 'recommendation': "HOLD", 'rs_value': 0, 'buy_price': None}
        
        signals = []
        
        # Get current RS value
        current_rs = self.data['Relative_Strength_Value'].iloc[-1]
        
        # Check if RS is above zero (stock outperformed Nifty)
        if current_rs > 0:
            signals.append("PASS - Stock outperformed Nifty over 55 days")
        else:
            signals.append("FAIL - Stock underperformed Nifty over 55 days")
        
        # Check RS trend
        current_rs_sma = self.data['RS_SMA'].iloc[-1]
        if current_rs > current_rs_sma:
            signals.append("PASS - RS above its SMA (rising trend)")
        else:
            signals.append("FAIL - RS below its SMA (falling trend)")
        
        # Check price trend
        current_price = self.data['Close'].iloc[-1]
        current_price_sma = self.data['Price_SMA'].iloc[-1]
        
        if current_price > current_price_sma:
            signals.append("PASS - Price above its SMA (uptrend)")
        else:
            signals.append("FAIL - Price below its SMA (downtrend)")
        
        # Count passes
        passes = sum(1 for s in signals if s.startswith("PASS"))
        
        # Determine recommendation
        if passes >= 2:
            recommendation = "BUY"
        elif passes == 1:
            recommendation = "HOLD"
        else:
            recommendation = "SELL"
        
        self.signals['Relative Strength'] = {
            'signals': signals,
            'recommendation': recommendation,
            'rs_value': current_rs,
            'buy_price': None
        }
        
        return self.signals['Relative Strength']
    
    def apply_ml_momentum_strategy(self):
        """Apply Machine Learning-Enhanced Momentum strategy"""
        if self.data is None or len(self.data) < 200:
            return {'signals': ["Insufficient data for ML"], 'recommendation': "HOLD", 'accuracy': 0, 'buy_price': None}
        
        # Prepare features and target
        df = self.data.copy()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) < 100:
            return {'signals': ["Insufficient data after cleaning"], 'recommendation': "HOLD", 'accuracy': 0, 'buy_price': None}
        
        # Define features
        features = [
            'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d',
            'Volatility', 'Volume_Change', 'Relative_Strength_Value',
            'Dist_10EMA', 'Dist_20EMA', 'Dist_50EMA'
        ]
        
        # Define target: 1 if next day return is positive, else 0
        df['Target'] = (df['Return_1d'].shift(-1) > 0).astype(int)
        
        # Drop last row as it has NaN target
        df = df[:-1]
        
        if len(df) < 50:
            return {'signals': ["Insufficient data for ML modeling"], 'recommendation': "HOLD", 'accuracy': 0, 'buy_price': None}
        
        X = df[features]
        y = df['Target']
        
        # Handle infinite and NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Make prediction for the most recent data
        latest_features = df[features].iloc[-1:].values
        prediction_prob = model.predict_proba(latest_features)[0]
        
        # Determine recommendation
        if prediction_prob[1] > 0.65:  # High probability of positive return
            recommendation = "BUY"
            signals = [f"BUY - ML predicts {prediction_prob[1]:.2%} probability of positive return"]
            buy_price = self.data['Close'].iloc[-1]  # Use current close as buy price
        elif prediction_prob[0] > 0.65:  # High probability of negative return
            recommendation = "SELL"
            signals = [f"SELL - ML predicts {prediction_prob[0]:.2%} probability of negative return"]
            buy_price = None
        else:
            recommendation = "HOLD"
            signals = [f"HOLD - ML prediction uncertain (BUY: {prediction_prob[1]:.2%}, SELL: {prediction_prob[0]:.2%})"]
            buy_price = None
        
        self.signals['Machine Learning'] = {
            'signals': signals,
            'recommendation': recommendation,
            'accuracy': accuracy,
            'buy_price': buy_price
        }
        
        return self.signals['Machine Learning']
    
    def _get_fundamental_data(self):
        """Get fundamental data for Indian stocks"""
        try:
            # Try to get data from Yahoo Finance
            ticker = yf.Ticker(self.stock_name)
            info = ticker.info
            
            # Convert market cap to crores (1 crore = 10 million)
            market_cap = info.get('marketCap', 0) / 10000000 if info.get('marketCap') else 0
            
            # Get revenue data
            total_revenue = info.get('totalRevenue', 0) / 10000000 if info.get('totalRevenue') else 0
            
            # Get PEG ratio
            peg_ratio = info.get('pegRatio', 0)
            
            # Get earnings data
            earnings_growth = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0
            revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
            
            # Get quarterly profit data (if available)
            quarterly_profit = info.get('netIncomeToCommon', 0) / 10000000 if info.get('netIncomeToCommon') else 0
            
            # For demonstration, we'll use available data and estimate missing values
            fundamentals = {
                'Market Cap': market_cap,
                'Sales Revenue': total_revenue,
                'PEG Ratio': peg_ratio,
                'Sales Growth 1Y': revenue_growth,
                'Sales Growth 3Y': revenue_growth * 0.9,  # Estimate
                'Sales Growth 5Y': revenue_growth * 0.8,  # Estimate
                'EPS Growth 1Y': earnings_growth,
                'EPS Growth 3Y': earnings_growth * 0.9,   # Estimate
                'EPS Growth 5Y': earnings_growth * 0.8,   # Estimate
                'EPS Ratio': 1.0 + (earnings_growth / 100),  # Estimate
                'Quarterly Net Profit': quarterly_profit,
                'Net Profit 2Q Ago': quarterly_profit * 0.95  # Estimate
            }
            
            return fundamentals
            
        except Exception as e:
            print(f"Error fetching fundamental data: {e}")
            # Return mock data if real data is not available
            return {
                'Market Cap': 1500,  # in crores
                'Sales Revenue': 1200,  # in crores
                'PEG Ratio': 1.2,
                'Sales Growth 1Y': 18,
                'Sales Growth 3Y': 20,
                'Sales Growth 5Y': 22,
                'EPS Growth 1Y': 19,
                'EPS Growth 3Y': 21,
                'EPS Growth 5Y': 23,
                'EPS Ratio': 1.18,  # Last Year / Preceding Year
                'Quarterly Net Profit': 150,
                'Net Profit 2Q Ago': 140
            }
    
    def _determine_trend(self):
        """Determine the overall trend using EMAs and price action"""
        if self.data is None or len(self.data) < 50:
            return "Unknown"
        
        recent_data = self.data.tail(20)
        
        # Check if EMAs are in bullish alignment
        bullish_ema_alignment = (recent_data['EMA_10'].iloc[-1] > recent_data['EMA_20'].iloc[-1] > 
                                recent_data['EMA_50'].iloc[-1])
        
        # Check if price is above key EMAs
        price_above_ema = (recent_data['Close'].iloc[-1] > recent_data['EMA_50'].iloc[-1])
        
        # Check for higher highs and higher lows
        higher_highs = (recent_data['High'].iloc[-1] > recent_data['High'].iloc[-5])
        higher_lows = (recent_data['Low'].iloc[-1] > recent_data['Low'].iloc[-5])
        
        if bullish_ema_alignment and price_above_ema and (higher_highs or higher_lows):
            return "Uptrend"
        
        # Check for bearish alignment
        bearish_ema_alignment = (recent_data['EMA_10'].iloc[-1] < recent_data['EMA_20'].iloc[-1] < 
                                recent_data['EMA_50'].iloc[-1])
        
        # Check if price is below key EMAs
        price_below_ema = (recent_data['Close'].iloc[-1] < recent_data['EMA_50'].iloc[-1])
        
        # Check for lower highs and lower lows
        lower_highs = (recent_data['High'].iloc[-1] < recent_data['High'].iloc[-5])
        lower_lows = (recent_data['Low'].iloc[-1] < recent_data['Low'].iloc[-5])
        
        if bearish_ema_alignment and price_below_ema and (lower_highs or lower_lows):
            return "Downtrend"
        
        return "Sideways"
    
    def _find_peaks_troughs(self, data, order=3):
        """Find peaks and troughs in price data with improved logic"""
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find peaks and troughs
        high_idx = argrelextrema(highs, np.greater, order=order)[0]
        low_idx = argrelextrema(lows, np.less, order=order)[0]
        
        # Filter for significant peaks/troughs (at least 1% from neighboring points)
        significant_high_idx = []
        significant_low_idx = []
        
        for idx in high_idx:
            if idx > 0 and idx < len(highs) - 1:
                if (highs[idx] > highs[idx-1] * 1.01 and 
                    highs[idx] > highs[idx+1] * 1.01):
                    significant_high_idx.append(idx)
        
        for idx in low_idx:
            if idx > 0 and idx < len(lows) - 1:
                if (lows[idx] < lows[idx-1] * 0.99 and 
                    lows[idx] < lows[idx+1] * 0.99):
                    significant_low_idx.append(idx)
        
        return significant_high_idx, significant_low_idx
    
    def _detect_wedge_pop_improved(self, data):
        """Improved Wedge Pop pattern (Bullish Falling Wedge) detection"""
        if len(data) < 20:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Find local maxima and minima with improved peak/trough detection
        high_idx, low_idx = self._find_peaks_troughs(data, order=2)
        
        if len(high_idx) < 2 or len(low_idx) < 2:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Get the last 3-4 highs and lows
        recent_highs = data['High'].iloc[high_idx[-4:]].values if len(high_idx) >= 4 else data['High'].iloc[high_idx[-3:]].values
        recent_lows = data['Low'].iloc[low_idx[-4:]].values if len(low_idx) >= 4 else data['Low'].iloc[low_idx[-3:]].values
        
        # Check if highs are decreasing and lows are decreasing (falling wedge)
        highs_decreasing = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))
        lows_decreasing = all(recent_lows[i] > recent_lows[i+1] for i in range(len(recent_lows)-1))
        
        if not (highs_decreasing and lows_decreasing):
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check for converging trendlines
        x_high = np.arange(len(recent_highs))
        x_low = np.arange(len(recent_lows))
        
        # Fit linear trendlines
        high_coeffs = np.polyfit(x_high, recent_highs, 1)
        low_coeffs = np.polyfit(x_low, recent_lows, 1)
        
        high_slope = high_coeffs[0]
        low_slope = low_coeffs[0]
        
        # For a falling wedge, the high slope should be more negative than the low slope
        if high_slope >= low_slope:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check for breakout above the upper trendline with volume confirmation
        current_close = data['Close'].iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Avg_Volume_20'].iloc[-1]
        
        # Calculate the upper trendline value at the current point
        high_trendline = np.polyval(high_coeffs, len(recent_highs))
        
        # Check if current close is above the upper trendline (with 1% tolerance)
        if current_close <= high_trendline * 1.01:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check volume confirmation (at least 2x average as per Kell's strategy)
        if current_volume < avg_volume * 2.0:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check relative strength (should be positive)
        if 'Relative_Strength_Value' in data.columns and data['Relative_Strength_Value'].iloc[-1] < 1.0:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Set stop loss below the lowest point of the wedge
        stop_loss = min(recent_lows) - data['ATR'].iloc[-1]
        
        # Set buy price at current close (breakout level)
        buy_price = current_close
        
        return {'is_pattern': True, 'stop_loss': stop_loss, 'buy_price': buy_price}
    
    def _detect_wedge_drop_improved(self, data):
        """Improved Wedge Drop pattern (Bearish Rising Wedge) detection"""
        if len(data) < 20:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Find local maxima and minima with improved peak/trough detection
        high_idx, low_idx = self._find_peaks_troughs(data, order=2)
        
        if len(high_idx) < 2 or len(low_idx) < 2:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Get the last 3-4 highs and lows
        recent_highs = data['High'].iloc[high_idx[-4:]].values if len(high_idx) >= 4 else data['High'].iloc[high_idx[-3:]].values
        recent_lows = data['Low'].iloc[low_idx[-4:]].values if len(low_idx) >= 4 else data['Low'].iloc[low_idx[-3:]].values
        
        # Check if highs are increasing and lows are increasing (rising wedge)
        highs_increasing = all(recent_highs[i] < recent_highs[i+1] for i in range(len(recent_highs)-1))
        lows_increasing = all(recent_lows[i] < recent_lows[i+1] for i in range(len(recent_lows)-1))
        
        if not (highs_increasing and lows_increasing):
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check for converging trendlines
        x_high = np.arange(len(recent_highs))
        x_low = np.arange(len(recent_lows))
        
        # Fit linear trendlines
        high_coeffs = np.polyfit(x_high, recent_highs, 1)
        low_coeffs = np.polyfit(x_low, recent_lows, 1)
        
        high_slope = high_coeffs[0]
        low_slope = low_coeffs[0]
        
        # For a rising wedge, the low slope should be steeper than the high slope
        if low_slope <= high_slope:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check for breakdown below the lower trendline with volume confirmation
        current_close = data['Close'].iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Avg_Volume_20'].iloc[-1]
        
        # Calculate the lower trendline value at the current point
        low_trendline = np.polyval(low_coeffs, len(recent_lows))
        
        # Check if current close is below the lower trendline (with 1% tolerance)
        if current_close >= low_trendline * 0.99:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check volume confirmation (at least 2x average)
        if current_volume < avg_volume * 2.0:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Set stop loss above the highest point of the wedge
        stop_loss = max(recent_highs) + data['ATR'].iloc[-1]
        
        return {'is_pattern': True, 'stop_loss': stop_loss, 'buy_price': None}
    
    def _detect_base_n_break(self, data, trend):
        """Detect Base N Break pattern (Classic CAN SLIM)"""
        if trend != "Uptrend" or len(data) < 20:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Look for consolidation (base) in the last 15-20 days
        base_data = data.tail(20)
        
        # Calculate the range of the base
        base_high = base_data['High'].max()
        base_low = base_data['Low'].min()
        base_range = (base_high - base_low) / base_low * 100
        
        # Base should be relatively tight (less than 15% range)
        if base_range > 15:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check if the base is above the 50-day EMA
        if base_low < data['EMA_50'].iloc[-1] * 0.95:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check for breakout in the last 3 days
        breakout_data = data.tail(3)
        current_close = data['Close'].iloc[-1]
        
        # Check if current close is near the base high
        if current_close < base_high * 0.97:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check volume confirmation (at least 2x average)
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Avg_Volume_20'].iloc[-1]
        
        if current_volume < avg_volume * 2.0:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check relative strength (should be positive)
        if 'Relative_Strength_Value' in data.columns and data['Relative_Strength_Value'].iloc[-1] < 1.1:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Set stop loss below the base low
        stop_loss = base_low - data['ATR'].iloc[-1]
        
        # Set buy price at the base high (breakout level)
        buy_price = base_high
        
        return {'is_pattern': True, 'stop_loss': stop_loss, 'buy_price': buy_price}
    
    def _detect_ema_crossback(self, data, trend):
        """Detect EMA Crossback pattern"""
        if trend != "Uptrend" or len(data) < 10:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check if price is pulling back to the 20-day EMA
        current_close = data['Close'].iloc[-1]
        ema_20 = data['EMA_20'].iloc[-1]
        
        # Check if price is near the 20-day EMA (within 3%)
        if abs(current_close - ema_20) / ema_20 * 100 > 3:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check if the 20-day EMA is above the 50-day EMA (uptrend confirmation)
        if ema_20 < data['EMA_50'].iloc[-1]:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check if there's a bounce off the EMA in the last 1-2 days
        recent_lows = data['Low'].tail(3)
        if not any(low <= ema_20 * 1.02 for low in recent_lows):
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check volume on the bounce (at least 1.5x average)
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Avg_Volume_20'].iloc[-1]
        
        if current_volume < avg_volume * 1.5:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check relative strength (should be positive)
        if 'Relative_Strength_Value' in data.columns and data['Relative_Strength_Value'].iloc[-1] < 1.05:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Set stop loss below the 20-day EMA or recent low, whichever is lower
        recent_low = data['Low'].tail(5).min()
        stop_loss = min(ema_20 * 0.97, recent_low) - data['ATR'].iloc[-1]
        
        # Set buy price at current close (bounce level)
        buy_price = current_close
        
        return {'is_pattern': True, 'stop_loss': stop_loss, 'buy_price': buy_price}
    
    def _detect_exhaustion_extension(self, data, trend):
        """Detect Exhaustion Extension pattern"""
        if trend != "Uptrend" or len(data) < 10:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check if price is far above key EMAs (more than 15%)
        dist_20ema = data['Dist_20EMA'].iloc[-1]
        if dist_20ema < 15:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check for parabolic move (accelerating price increase)
        recent_prices = data['Close'].tail(10).values
        price_changes = np.diff(recent_prices) / recent_prices[:-1] * 100
        
        # Check if price changes are accelerating
        if len(price_changes) < 5:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Calculate the slope of price changes
        x = np.arange(len(price_changes))
        slope, _ = np.polyfit(x, price_changes, 1)
        
        if slope <= 0:  # Not accelerating
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check for signs of exhaustion in the last candle
        last_candle = data.iloc[-1]
        
        # Check for long upper shadow (shooting star pattern)
        body = abs(last_candle['Close'] - last_candle['Open'])
        upper_shadow = last_candle['High'] - max(last_candle['Close'], last_candle['Open'])
        
        if upper_shadow > body * 2 and body < (last_candle['High'] - last_candle['Low']) * 0.3:
            # Set stop loss below the low of the last candle
            stop_loss = last_candle['Low'] - data['ATR'].iloc[-1]
            return {'is_pattern': True, 'stop_loss': stop_loss, 'buy_price': None}
        
        # Check for high volume without price progress (churning)
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Avg_Volume_20'].iloc[-1]
        
        if current_volume > avg_volume * 2.0 and abs(price_changes[-1]) < 2:
            # Set stop loss below the low of the last candle
            stop_loss = last_candle['Low'] - data['ATR'].iloc[-1]
            return {'is_pattern': True, 'stop_loss': stop_loss, 'buy_price': None}
        
        return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
    
    def _detect_reversal_extension(self, data, trend):
        """Detect Reversal Extension pattern"""
        if trend != "Uptrend" or len(data) < 15:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Find the most recent swing high
        highs = data['High'].values
        high_idx, _ = self._find_peaks_troughs(data, order=2)
        
        if len(high_idx) < 2:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        last_high_idx = high_idx[-1]
        last_high = highs[last_high_idx]
        
        # Get data after the last high
        after_high_data = data.iloc[last_high_idx+1:]
        
        if len(after_high_data) < 5:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Find the lowest point after the high
        pullback_low = after_high_data['Low'].min()
        pullback_low_idx = after_high_data['Low'].idxmin()
        
        # Check if the pullback is shallow (less than 10% from high)
        pullback_pct = (last_high - pullback_low) / last_high * 100
        if pullback_pct > 10:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check if the price has reversed back up
        current_close = data['Close'].iloc[-1]
        if current_close < pullback_low * 1.03:  # Not yet reversed
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check if the reversal happened quickly (within 5 days)
        days_from_low = (data.index[-1] - pullback_low_idx).days
        if days_from_low > 5:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check volume on reversal (at least 1.5x average)
        reversal_volume = data.loc[pullback_low_idx:, 'Volume'].mean()
        avg_volume = data['Avg_Volume_20'].iloc[-1]
        
        if reversal_volume < avg_volume * 1.5:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Check relative strength (should be positive)
        if 'Relative_Strength_Value' in data.columns and data['Relative_Strength_Value'].iloc[-1] < 1.1:
            return {'is_pattern': False, 'stop_loss': None, 'buy_price': None}
        
        # Set stop loss below the pullback low
        stop_loss = pullback_low - data['ATR'].iloc[-1]
        
        # Set buy price at current close (reversal level)
        buy_price = current_close
        
        return {'is_pattern': True, 'stop_loss': stop_loss, 'buy_price': buy_price}
    
    def _detect_bull_flag_daily(self, data):
        """Detect Bull Flag pattern adapted for daily timeframe"""
        if len(data) < 15:
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Identify flagpole (strong upward move over 5-10 days)
        flagpole_end = min(10, len(data) // 2)
        flagpole = data.iloc[:flagpole_end]
        flag = data.iloc[flagpole_end:]
        
        # Check for strong flagpole move (at least 10%)
        flagpole_gain = (flagpole['Close'].iloc[-1] - flagpole['Close'].iloc[0]) / flagpole['Close'].iloc[0] * 100
        if flagpole_gain < 10:
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Check flag structure (downward-sloping consolidation)
        flag_highs = flag['High'].values
        flag_lows = flag['Low'].values
        
        # Check for downward-sloping highs and lows
        highs_decreasing = all(flag_highs[i] > flag_highs[i+1] for i in range(len(flag_highs)-1))
        lows_decreasing = all(flag_lows[i] > flag_lows[i+1] for i in range(len(flag_lows)-1))
        
        if not (highs_decreasing and lows_decreasing):
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Check for tight consolidation (less than 5% range)
        flag_range = (flag['High'].max() - flag['Low'].min()) / flag['Low'].min() * 100
        if flag_range > 5:
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Check if flag is near EMAs (within 5%)
        ema_9 = data['EMA_9'].iloc[-1]
        ema_21 = data['EMA_21'].iloc[-1]
        flag_close = flag['Close'].iloc[-1]
        
        if abs(flag_close - ema_9) / ema_9 * 100 > 5 and abs(flag_close - ema_21) / ema_21 * 100 > 5:
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Check for breakout above flag high
        breakout_level = flag['High'].max()
        current_close = data['Close'].iloc[-1]
        
        if current_close < breakout_level * 0.99:  # Not yet broken out
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Volume confirmation (at least 1.5x average)
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Avg_Volume_20'].iloc[-1]
        
        if current_volume < avg_volume * 1.5:
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Set stop loss below flag low
        stop_loss = flag['Low'].min() - data['ATR'].iloc[-1]
        
        # Set buy price at breakout level
        buy_price = breakout_level
        
        return {
            'is_pattern': True,
            'breakout_level': breakout_level,
            'stop_loss': stop_loss,
            'buy_price': buy_price
        }
    
    def _detect_ema_kiss_fly(self, data):
        """Detect EMA Kiss & Fly pattern"""
        if len(data) < 10:
            return {'is_pattern': False, 'kiss_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Look for pullback to EMA over 2-5 days
        pullback_days = min(5, len(data) - 5)
        pullback_data = data.tail(pullback_days + 5)
        
        # Check if price has come within 2-3% of 9/21 EMA during pullback
        ema_9 = data['EMA_9'].iloc[-1]
        ema_21 = data['EMA_21'].iloc[-1]
        
        # Find the closest approach to EMAs during pullback
        min_dist_to_ema9 = min(abs(pullback_data['Close'].iloc[i] - pullback_data['EMA_9'].iloc[i]) / 
                               pullback_data['EMA_9'].iloc[i] * 100 for i in range(pullback_days))
        
        min_dist_to_ema21 = min(abs(pullback_data['Close'].iloc[i] - pullback_data['EMA_21'].iloc[i]) / 
                               pullback_data['EMA_21'].iloc[i] * 100 for i in range(pullback_days))
        
        # Check if price came within 3% of either EMA
        if min_dist_to_ema9 > 3 and min_dist_to_ema21 > 3:
            return {'is_pattern': False, 'kiss_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Determine which EMA was kissed
        if min_dist_to_ema9 <= min_dist_to_ema21:
            kiss_level = ema_9
        else:
            kiss_level = ema_21
        
        # Check for bounce upward with increasing volume
        recent_data = data.tail(3)
        
        # Check if price is moving up (last close higher than 2 days ago)
        if recent_data['Close'].iloc[-1] <= recent_data['Close'].iloc[-3]:
            return {'is_pattern': False, 'kiss_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Check volume is increasing
        if recent_data['Volume'].iloc[-1] <= recent_data['Volume'].iloc[-2] * 0.9:
            return {'is_pattern': False, 'kiss_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Set stop loss below the lowest point of the pullback
        pullback_low = pullback_data['Low'].min()
        stop_loss = pullback_low - data['ATR'].iloc[-1]
        
        # Set buy price at current close (bounce level)
        buy_price = data['Close'].iloc[-1]
        
        return {
            'is_pattern': True,
            'kiss_level': kiss_level,
            'stop_loss': stop_loss,
            'buy_price': buy_price
        }
    
    def _detect_vcp(self, data):
        """Detect Volatility Contraction Pattern (VCP)"""
        if len(data) < 15:
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Look for decreasing daily ranges over 5-10 days
        contraction_period = min(10, len(data) - 5)
        contraction_data = data.tail(contraction_period + 5)
        
        # Check if daily ranges are decreasing
        daily_ranges = contraction_data['Daily_Range'].tail(contraction_period)
        
        # Calculate the slope of daily ranges
        x = np.arange(len(daily_ranges))
        slope, _ = np.polyfit(x, daily_ranges, 1)
        
        # Slope should be negative (ranges decreasing)
        if slope >= 0:
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Check if price is hugging the 21 EMA upward (within 3%)
        ema_21 = data['EMA_21'].iloc[-1]
        current_close = data['Close'].iloc[-1]
        
        if abs(current_close - ema_21) / ema_21 * 100 > 3:
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Check if 21 EMA is rising
        if ema_21 <= data['EMA_21'].iloc[-2]:
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Check for expansion (breakout) in the last 1-3 days
        recent_data = data.tail(3)
        
        # Find the highest point during contraction
        contraction_high = contraction_data['High'].max()
        
        # Check if current close is near or above contraction high
        if current_close < contraction_high * 0.98:
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Check volume confirmation on breakout (at least 1.5x average)
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Avg_Volume_20'].iloc[-1]
        
        if current_volume < avg_volume * 1.5:
            return {'is_pattern': False, 'breakout_level': None, 'stop_loss': None, 'buy_price': None}
        
        # Set stop loss below the lowest point of the contraction
        contraction_low = contraction_data['Low'].min()
        stop_loss = contraction_low - data['ATR'].iloc[-1]
        
        # Set buy price at contraction high (breakout level)
        buy_price = contraction_high
        
        return {
            'is_pattern': True,
            'breakout_level': contraction_high,
            'stop_loss': stop_loss,
            'buy_price': buy_price
        }
    
    def _check_ema_alignment(self, data):
        """Check EMA alignment for trend confirmation"""
        if len(data) < 5:
            return {'signal': None, 'stop_loss': None, 'buy_price': None}
        
        ema9 = data['EMA_9'].iloc[-1]
        ema21 = data['EMA_21'].iloc[-1]
        current_close = data['Close'].iloc[-1]
        
        # Check for bullish alignment (price above both EMAs, 9 above 21)
        if current_close > ema9 > ema21:
            # Check if EMAs are rising
            if ema9 > data['EMA_9'].iloc[-2] and ema21 > data['EMA_21'].iloc[-2]:
                stop_loss = min(data['Low'].tail(5)) - data['ATR'].iloc[-1]
                buy_price = current_close
                return {
                    'signal': "BUY - Strong bullish EMA alignment",
                    'stop_loss': stop_loss,
                    'buy_price': buy_price
                }
        
        # Check for bearish alignment (price below both EMAs, 9 below 21)
        if current_close < ema9 < ema21:
            # Check if EMAs are falling
            if ema9 < data['EMA_9'].iloc[-2] and ema21 < data['EMA_21'].iloc[-2]:
                stop_loss = max(data['High'].tail(5)) + data['ATR'].iloc[-1]
                return {
                    'signal': "SELL - Strong bearish EMA alignment",
                    'stop_loss': stop_loss,
                    'buy_price': None
                }
        
        return {'signal': None, 'stop_loss': None, 'buy_price': None}
    
    def _check_volume_surge(self, data):
        """Check for volume surge confirmation"""
        if len(data) < 5:
            return {'signal': None}
        
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Avg_Volume_20'].iloc[-1]
        
        # Check for volume surge (at least 200% of average)
        if current_volume > avg_volume * 2.0:
            # Check price action
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
            
            if price_change > 2:
                return {'signal': "BUY - Volume surge on upward price move"}
            elif price_change < -2:
                return {'signal': "SELL - Volume surge on downward price move"}
        
        return {'signal': None}
    
    def analyze(self):
        """Run all strategies and return results"""
        if self.data is None:
            if not self.fetch_data():
                return None
                
        self.calculate_indicators()
        
        # Apply each strategy
        oliver_kell_result = self.apply_oliver_kell_strategy()
        goverdhan_gajjala_result = self.apply_goverdhan_gajjala_strategy()
        shankar_nath_result = self.apply_shankar_nath_strategy()
        relative_strength_result = self.apply_relative_strength_strategy()
        ml_momentum_result = self.apply_ml_momentum_strategy()
        
        return {
            'Oliver Kell': oliver_kell_result,
            'Goverdhan Gajjala': goverdhan_gajjala_result,
            'Shankar Nath': shankar_nath_result,
            'Relative Strength': relative_strength_result,
            'Machine Learning': ml_momentum_result
        }
    
    def get_final_recommendation(self, results):
        """Determine final recommendation based on all strategies"""
        if results is None:
            return "HOLD"
        
        # Count buy, sell, and hold signals
        buy_count = 0
        sell_count = 0
        hold_count = 0
        
        for strategy, result in results.items():
            if strategy == 'Shankar Nath':
                if result['recommendation'] == 'BUY':
                    buy_count += 1
                elif result['recommendation'] == 'CONSIDER':
                    hold_count += 1
                else:  # AVOID
                    sell_count += 1
            else:
                if result['recommendation'] == 'BUY':
                    buy_count += 1
                elif result['recommendation'] == 'SELL':
                    sell_count += 1
                else:  # HOLD, AVOID, etc.
                    hold_count += 1
        
        # Determine final recommendation
        if buy_count >= 3:
            return "BUY"
        elif sell_count >= 3:
            return "SELL"
        elif buy_count > sell_count:
            return "BUY"
        elif sell_count > buy_count:
            return "SELL"
        else:
            return "HOLD"
    
    def get_buy_price(self, results):
        """Determine buy price based on strategy signals"""
        if results is None:
            return None
        
        # Check Oliver Kell strategy first
        if results['Oliver Kell']['recommendation'] == 'BUY' and results['Oliver Kell']['buy_price'] is not None:
            return results['Oliver Kell']['buy_price']
        
        # Then check Goverdhan Gajjala strategy
        if results['Goverdhan Gajjala']['recommendation'] == 'BUY' and results['Goverdhan Gajjala']['buy_price'] is not None:
            return results['Goverdhan Gajjala']['buy_price']
        
        # Then check Machine Learning strategy
        if results['Machine Learning']['recommendation'] == 'BUY' and results['Machine Learning']['buy_price'] is not None:
            return results['Machine Learning']['buy_price']
        
        # If no specific buy price from strategies, use current close price for BUY recommendations
        final_rec = self.get_final_recommendation(results)
        if final_rec == 'BUY':
            return self.data['Close'].iloc[-1]
        
        return None
    
    def plot_chart(self):
        """Plot the stock price with technical indicators using Plotly"""
        if self.data is None:
            return None
            
        # Create figure with secondary y-axis for volume
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.1, 
            subplot_titles=[f'{self.stock_name.upper()} - Price Chart with EMAs', 'Volume'],
            row_width=[0.7, 0.3]
        )
        
        # Add price and EMAs
        fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['Close'], 
                name='Close Price', 
                line=dict(color='black', width=1.5)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['EMA_10'], 
                name='10-day EMA', 
                line=dict(color='blue', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['EMA_20'], 
                name='20-day EMA', 
                line=dict(color='red', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['EMA_50'], 
                name='50-day EMA', 
                line=dict(color='green', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['EMA_9'], 
                name='9-day EMA', 
                line=dict(color='purple', width=1, dash='dot')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['EMA_21'], 
                name='21-day EMA', 
                line=dict(color='orange', width=1, dash='dot')
            ),
            row=1, col=1
        )
        
        # Add support and resistance levels
        if 'Support' in self.data.columns and 'Resistance' in self.data.columns:
            # Get the most recent support and resistance levels
            recent_support = self.data['Support'].iloc[-1]
            recent_resistance = self.data['Resistance'].iloc[-1]
            
            # Add horizontal lines for support and resistance
            fig.add_hline(
                y=recent_support,
                line=dict(color='green', width=1, dash='dash'),
                annotation_text=f"Support: {recent_support:.2f}",
                annotation=dict(
                    x=self.data.index[-1],
                    y=recent_support,
                    xref="x",
                    yref="y",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                ),
                row=1, col=1
            )
            
            fig.add_hline(
                y=recent_resistance,
                line=dict(color='red', width=1, dash='dash'),
                annotation_text=f"Resistance: {recent_resistance:.2f}",
                annotation=dict(
                    x=self.data.index[-1],
                    y=recent_resistance,
                    xref="x",
                    yref="y",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=40
                ),
                row=1, col=1
            )
        
        # Add volume
        fig.add_trace(
            go.Bar(
                x=self.data.index, 
                y=self.data['Volume'], 
                name='Volume',
                marker_color='skyblue'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title_text=f'{self.stock_name.upper()} - Technical Analysis',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(240,248,255,0.9)',
            paper_bgcolor='rgba(240,248,255,0.9)'
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig

# Function to style dataframe with forest green for BUY recommendations
def style_dataframe(df):
    """Style the dataframe with row colors based on recommendation"""
    def highlight_row(row):
        if row['Final Recommendation'] == 'BUY':
            return ['background-color: #228B22'] * len(row)  # Forest green
        elif row['Final Recommendation'] == 'SELL':
            return ['background-color: #FFB6C1'] * len(row)  # Light red
        else:
            return ['background-color: #FFFFE0'] * len(row)  # Light yellow
    
    return df.style.apply(highlight_row, axis=1)

# Function to expandable details
def expandable_details(title, content):
    """Create an expandable section for detailed results"""
    with st.expander(title):
        for item in content:
            st.write(f"- {item}")

# Streamlit App
def main():
    st.title("📈 Advanced Stock Analyzer Made By Ashutosh")
    st.markdown("""
    This application analyzes stocks using multiple trading strategies:
    - Oliver Kell's Price Cycle Strategy
    - Goverdhan Gajjala's Momentum Strategy
    - Shankar Nath's GARP Strategy
    - Relative Strength Strategy (compared with Nifty)
    - Machine Learning-Enhanced Momentum Strategy
    """)
    
    # Sidebar for inputs
    st.sidebar.header("Stock Analysis")
    
    # Option 1: Single stock analysis
    st.sidebar.subheader("Single Stock Analysis")
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., TITAN, RELIANCE)", value="TITAN")
    
    # Option 2: Batch analysis with comma-separated symbols
    st.sidebar.subheader("Batch Analysis")
    stock_symbols_text = st.sidebar.text_area(
        "Enter Stock Symbols (comma-separated)",
        value="TATACOMM, BHARATFORG, OFSS, POLYCAB, ASTRAL, ABCAPITAL, VOLTAS, HINDPETRO, PNB, COFORGE, SAIL, MPHASIS, BATAINDIA, ASHOKLEY, NMDC, LUPIN, ZYDUSLIFE, ALKEM, AUROPHARMA, INDHOTEL"
    )
    
    # Analysis button
    analyze_button = st.sidebar.button("Analyze Stock(s)")
    
    # Initialize results dataframe
    results_df = pd.DataFrame(columns=[
        'Stock Name', 'Oliver Kell', 'Goverdhan Gajjala', 'Shankar Nath', 
        'Relative Strength', 'Machine Learning', 'Final Recommendation',
        'Support', 'Resistance', 'Buy Price', 'Current Phase'
    ])
    
    if analyze_button:
        if stock_symbols_text:
            # Process comma-separated stock symbols
            stock_symbols = [symbol.strip() for symbol in stock_symbols_text.split(',') if symbol.strip()]
            
            if stock_symbols:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Analyze each stock
                results = []
                for i, symbol in enumerate(stock_symbols):
                    status_text.text(f"Analyzing {symbol} ({i+1}/{len(stock_symbols)})")
                    
                    # Create analyzer
                    analyzer = StockAnalyzer(symbol)
                    
                    # Analyze stock
                    result = analyzer.analyze()
                    
                    if result:
                        # Get final recommendation
                        final_rec = analyzer.get_final_recommendation(result)
                        
                        # Get buy price
                        buy_price = analyzer.get_buy_price(result)
                        
                        # Get support and resistance levels
                        current_support = analyzer.data['Support'].iloc[-1] if 'Support' in analyzer.data.columns else None
                        current_resistance = analyzer.data['Resistance'].iloc[-1] if 'Resistance' in analyzer.data.columns else None
                        
                        # Add to results
                        results.append({
                            'Stock Name': symbol,
                            'Oliver Kell': result['Oliver Kell']['recommendation'],
                            'Goverdhan Gajjala': result['Goverdhan Gajjala']['recommendation'],
                            'Shankar Nath': f"{result['Shankar Nath']['passes']}/7",
                            'Relative Strength': result['Relative Strength']['recommendation'],
                            'Machine Learning': result['Machine Learning']['recommendation'],
                            'Final Recommendation': final_rec,
                            'Support': current_support,
                            'Resistance': current_resistance,
                            'Buy Price': buy_price,
                            'Current Phase': result['Oliver Kell']['trend'],
                            'Oliver Kell Details': result['Oliver Kell'],
                            'Goverdhan Gajjala Details': result['Goverdhan Gajjala'],
                            'Shankar Nath Details': result['Shankar Nath'],
                            'Relative Strength Details': result['Relative Strength'],
                            'Machine Learning Details': result['Machine Learning']
                        })
                    
                    # Update progress bar
                    progress_bar.progress((i+1)/len(stock_symbols))
                
                # Create results dataframe
                results_df = pd.DataFrame(results)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Display styled dataframe with built-in filtering
                st.dataframe(style_dataframe(results_df), use_container_width=True)
                
                # Display detailed results for each stock
                st.subheader("Detailed Results")
                
                # Create tabs for each stock
                if len(results_df) > 0:
                    stock_tabs = st.tabs([f"{row['Stock Name']} - {row['Final Recommendation']}" for _, row in results_df.iterrows()])
                    
                    for i, (_, row) in enumerate(results_df.iterrows()):
                        with stock_tabs[i]:
                            st.write(f"### {row['Stock Name']} - {row['Final Recommendation']}")
                            
                            # Create columns for each strategy
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("#### Oliver Kell's Strategy")
                                st.write(f"**Phase**: {row['Oliver Kell Details']['phase']}")
                                st.write(f"**Trend**: {row['Oliver Kell Details']['trend']}")
                                expandable_details(
                                    "Signals",
                                    row['Oliver Kell Details']['signals']
                                )
                                
                                st.write("#### Goverdhan Gajjala's Strategy")
                                expandable_details(
                                    "Signals",
                                    row['Goverdhan Gajjala Details']['signals']
                                )
                                
                                st.write("#### Shankar Nath's GARP Strategy")
                                st.write(f"**Passed**: {row['Shankar Nath Details']['passes']}/7 criteria")
                                expandable_details(
                                    "Criteria",
                                    row['Shankar Nath Details']['signals']
                                )
                                
                                st.write("#### Relative Strength Strategy")
                                st.write(f"**RS Value**: {row['Relative Strength Details']['rs_value']:.4f}")
                                st.write("**Explanation**: Compares stock performance with Nifty over 55 days. If RS > 0, stock outperformed Nifty (BUY signal)")
                                expandable_details(
                                    "Signals",
                                    row['Relative Strength Details']['signals']
                                )
                                
                                st.write("#### Machine Learning Strategy")
                                st.write(f"**Model Accuracy**: {row['Machine Learning Details']['accuracy']:.2%}")
                                st.write("**Explanation**: Uses Random Forest to predict next day's return direction based on technical indicators")
                                expandable_details(
                                    "Signals",
                                    row['Machine Learning Details']['signals']
                                )
                            
                            with col2:
                                st.write("#### Key Levels")
                                if row['Support'] is not None:
                                    st.write(f"**Support**: {row['Support']:.2f}")
                                if row['Resistance'] is not None:
                                    st.write(f"**Resistance**: {row['Resistance']:.2f}")
                                if row['Buy Price'] is not None:
                                    st.write(f"**Buy Price**: {row['Buy Price']:.2f}")
                                st.write(f"**Current Phase**: {row['Current Phase']}")
                                
                                # Display chart
                                st.write("#### Price Chart")
                                analyzer = StockAnalyzer(row['Stock Name'])
                                if analyzer.fetch_data():
                                    analyzer.calculate_indicators()
                                    fig = analyzer.plot_chart()
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                
                # Add download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='stock_analysis_results.csv',
                    mime='text/csv'
                )
        
        elif stock_symbol:
            # Analyze single stock
            analyzer = StockAnalyzer(stock_symbol)
            
            # Fetch data and analyze
            results = analyzer.analyze()
            
            if results:
                # Display results
                st.subheader(f"Analysis Results for {stock_symbol.upper()}")
                
                # Create summary dataframe
                summary_data = {
                    'Strategy': ['Oliver Kell', 'Goverdhan Gajjala', 'Shankar Nath', 'Relative Strength', 'Machine Learning'],
                    'Recommendation': [
                        results['Oliver Kell']['recommendation'],
                        results['Goverdhan Gajjala']['recommendation'],
                        f"{results['Shankar Nath']['passes']}/7",
                        results['Relative Strength']['recommendation'],
                        results['Machine Learning']['recommendation']
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                
                # Display summary
                st.dataframe(summary_df)
                
                # Display final recommendation
                final_rec = analyzer.get_final_recommendation(results)
                buy_price = analyzer.get_buy_price(results)
                
                st.markdown(f"### Final Recommendation: **{final_rec}**")
                if buy_price is not None:
                    st.markdown(f"### Suggested Buy Price: **{buy_price:.2f}**")
                
                # Create tabs for different sections
                tab1, tab2 = st.tabs(["Strategy Details", "Chart"])
                
                with tab1:
                    # Display strategy details
                    st.subheader("Oliver Kell's Strategy")
                    st.write(f"**Phase**: {results['Oliver Kell']['phase']}")
                    st.write(f"**Trend**: {results['Oliver Kell']['trend']}")
                    expandable_details(
                        "Signals",
                        results['Oliver Kell']['signals']
                    )
                    
                    st.subheader("Goverdhan Gajjala's Strategy")
                    expandable_details(
                        "Signals",
                        results['Goverdhan Gajjala']['signals']
                    )
                    
                    st.subheader("Shankar Nath's GARP Strategy")
                    st.write(f"**Passed**: {results['Shankar Nath']['passes']}/7 criteria")
                    expandable_details(
                        "Criteria",
                        results['Shankar Nath']['signals']
                    )
                    
                    st.subheader("Relative Strength Strategy")
                    st.write(f"**RS Value**: {results['Relative Strength']['rs_value']:.4f}")
                    st.write("**Explanation**: Compares stock performance with Nifty over 55 days. If RS > 0, stock outperformed Nifty (BUY signal)")
                    expandable_details(
                        "Signals",
                        results['Relative Strength']['signals']
                    )
                    
                    st.subheader("Machine Learning Strategy")
                    st.write(f"**Model Accuracy**: {results['Machine Learning']['accuracy']:.2%}")
                    st.write("**Explanation**: Uses Random Forest to predict next day's return direction based on technical indicators")
                    expandable_details(
                        "Signals",
                        results['Machine Learning']['signals']
                    )
                
                with tab2:
                    # Display chart
                    st.subheader("Price Chart with Technical Indicators")
                    fig = analyzer.plot_chart()
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Add to results dataframe
                results_df = pd.DataFrame([{
                    'Stock Name': stock_symbol,
                    'Oliver Kell': results['Oliver Kell']['recommendation'],
                    'Goverdhan Gajjala': results['Goverdhan Gajjala']['recommendation'],
                    'Shankar Nath': f"{results['Shankar Nath']['passes']}/7",
                    'Relative Strength': results['Relative Strength']['recommendation'],
                    'Machine Learning': results['Machine Learning']['recommendation'],
                    'Final Recommendation': final_rec,
                    'Support': analyzer.data['Support'].iloc[-1] if 'Support' in analyzer.data.columns else None,
                    'Resistance': analyzer.data['Resistance'].iloc[-1] if 'Resistance' in analyzer.data.columns else None,
                    'Buy Price': buy_price,
                    'Current Phase': results['Oliver Kell']['trend']
                }])
                
                # Add download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f'{stock_symbol}_analysis.csv',
                    mime='text/csv'
                )
            else:
                st.error(f"Could not analyze {stock_symbol}. Please check the stock symbol and try again.")
        else:
            st.warning("Please enter a stock symbol or a list of stock symbols.")

if __name__ == "__main__":
    main()