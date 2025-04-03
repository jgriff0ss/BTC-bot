import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, EMAIndicator
import time
import os
from dotenv import load_dotenv
import requests
import json
import logging
from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD_DEV,
    SMA_FAST, SMA_SLOW, EMA_FAST,
    STOCH_K, STOCH_D, STOCH_OVERSOLD, STOCH_OVERBOUGHT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

# Set page config
st.set_page_config(
    page_title="BTC Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        margin-top: 10px;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2c8ac0;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .signal-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .signal-card.long {
        border-left-color: #28a745;
    }
    .signal-card.short {
        border-left-color: #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'price_history' not in st.session_state:
    st.session_state.price_history = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = None

def get_btc_data():
    try:
        btc = yf.Ticker("BTC-USD")
        data = btc.history(period="1d", interval="1m")
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except Exception as e:
        logging.error(f"Error fetching BTC data: {e}")
        return None

def calculate_indicators(df):
    # Calculate RSI
    rsi_indicator = RSIIndicator(close=df['Close'], window=RSI_PERIOD)
    df['RSI'] = rsi_indicator.rsi()

    # Calculate MACD
    macd = MACD(close=df['Close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    # Calculate Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=BB_PERIOD, window_dev=BB_STD_DEV)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()

    # Calculate Moving Averages
    df['SMA_Fast'] = SMAIndicator(close=df['Close'], window=SMA_FAST).sma_indicator()
    df['SMA_Slow'] = SMAIndicator(close=df['Close'], window=SMA_SLOW).sma_indicator()
    df['EMA_Fast'] = EMAIndicator(close=df['Close'], window=EMA_FAST).ema_indicator()

    # Calculate Stochastic Oscillator
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=STOCH_K, smooth_window=STOCH_D)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    return df

def check_signals(df):
    current_price = df['Close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd_hist = df['MACD_Hist'].iloc[-1]
    bb_position = (current_price - df['BB_Lower'].iloc[-1]) / (df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1])
    stoch_k = df['Stoch_K'].iloc[-1]
    stoch_d = df['Stoch_D'].iloc[-1]
    
    # Calculate signal strength (0-100)
    signal_strength = 0
    signal_type = None
    reasons = []

    # RSI signals
    if rsi < RSI_OVERSOLD:
        signal_strength += 30
        reasons.append(f"RSI oversold ({rsi:.2f})")
    elif rsi > RSI_OVERBOUGHT:
        signal_strength -= 30
        reasons.append(f"RSI overbought ({rsi:.2f})")

    # MACD signals
    if macd_hist > 0:
        signal_strength += 20
        reasons.append("MACD histogram positive")
    elif macd_hist < 0:
        signal_strength -= 20
        reasons.append("MACD histogram negative")

    # Bollinger Bands signals
    if current_price < df['BB_Lower'].iloc[-1]:
        signal_strength += 20
        reasons.append("Price below lower Bollinger Band")
    elif current_price > df['BB_Upper'].iloc[-1]:
        signal_strength -= 20
        reasons.append("Price above upper Bollinger Band")

    # Stochastic signals
    if stoch_k < STOCH_OVERSOLD:
        signal_strength += 15
        reasons.append(f"Stochastic oversold ({stoch_k:.2f})")
    elif stoch_k > STOCH_OVERBOUGHT:
        signal_strength -= 15
        reasons.append(f"Stochastic overbought ({stoch_k:.2f})")

    # Moving Average signals
    if df['SMA_Fast'].iloc[-1] > df['SMA_Slow'].iloc[-1]:
        signal_strength += 15
        reasons.append("Fast SMA above Slow SMA")
    elif df['SMA_Fast'].iloc[-1] < df['SMA_Slow'].iloc[-1]:
        signal_strength -= 15
        reasons.append("Fast SMA below Slow SMA")

    # Determine signal type
    if signal_strength > 50:
        signal_type = "LONG"
    elif signal_strength < -50:
        signal_type = "SHORT"
    else:
        signal_type = "NEUTRAL"

    return {
        'signal_type': signal_type,
        'signal_strength': abs(signal_strength),
        'reasons': reasons,
        'current_price': current_price
    }

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data)
        if response.status_code != 200:
            logging.error(f"Failed to send Telegram message: {response.text}")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

def main():
    st.title("📈 BTC Trading Bot Dashboard")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        update_interval = st.slider("Update Interval (seconds)", 10, 300, 60)
        show_indicators = st.multiselect(
            "Show Indicators",
            ["RSI", "MACD", "Bollinger Bands", "Moving Averages", "Stochastic"],
            default=["RSI", "MACD", "Bollinger Bands"]
        )
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard provides real-time BTC price monitoring and trading signals based on technical analysis.
        
        **Features:**
        - Real-time price updates
        - Technical indicators
        - Trading signals
        - Interactive charts
        """)
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Current Price")
        current_price = get_btc_data()
        if current_price:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #1f77b4;">${current_price:,.2f}</h2>
                <p style="color: #666;">BTC/USD</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 24h Change")
        if current_price:
            btc = yf.Ticker("BTC-USD")
            hist = btc.history(period="1d")
            if not hist.empty:
                change = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                color = "green" if change >= 0 else "red"
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="color: {color};">{change:+.2f}%</h2>
                    <p style="color: #666;">24h Change</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### Last Update")
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #1f77b4;">{current_time}</h2>
            <p style="color: #666;">Last Price Update</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Trading Signals
    st.markdown("### Trading Signals")
    if current_price:
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period="1d", interval="1m")
        df = calculate_indicators(df)
        signals = check_signals(df)
        
        signal_color = {
            "LONG": "#28a745",
            "SHORT": "#dc3545",
            "NEUTRAL": "#6c757d"
        }
        
        st.markdown(f"""
        <div class="signal-card {signals['signal_type'].lower()}">
            <h3 style="color: {signal_color[signals['signal_type']]}; margin: 0;">{signals['signal_type']} Signal</h3>
            <p style="margin: 5px 0;">Signal Strength: {signals['signal_strength']:.1f}%</p>
            <p style="margin: 5px 0;">Current Price: ${signals['current_price']:,.2f}</p>
            <p style="margin: 5px 0;">Reasons:</p>
            <ul style="margin: 5px 0; padding-left: 20px;">
                {''.join([f'<li>{reason}</li>' for reason in signals['reasons']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    st.markdown("### Price Charts")
    if current_price:
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period="1d", interval="1m")
        
        # Price Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='BTC/USD'
        ))
        
        if "Bollinger Bands" in show_indicators:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name='BB Upper',
                line=dict(color='gray', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name='BB Lower',
                line=dict(color='gray', dash='dash')
            ))
        
        fig.update_layout(
            title='BTC/USD Price',
            yaxis_title='Price (USD)',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Indicators Chart
        if "RSI" in show_indicators or "MACD" in show_indicators:
            fig = go.Figure()
            
            if "RSI" in show_indicators:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color='blue')
                ))
                fig.add_hline(y=RSI_OVERBOUGHT, line_dash="dash", line_color="red")
                fig.add_hline(y=RSI_OVERSOLD, line_dash="dash", line_color="green")
            
            if "MACD" in show_indicators:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    name='MACD',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['MACD_Signal'],
                    name='Signal',
                    line=dict(color='orange')
                ))
            
            fig.update_layout(
                title='Technical Indicators',
                yaxis_title='Value',
                template='plotly_white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh
    time.sleep(update_interval)
    st.experimental_rerun()

if __name__ == "__main__":
    main() 