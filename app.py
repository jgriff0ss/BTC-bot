import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time
from trading_bot import TradingBot
import config

# Configure Streamlit for public sharing
st.set_page_config(
    page_title="BTC Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add authentication
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.session_state["credentials"] and st.session_state["password"] == st.session_state["credentials"][st.session_state["username"]]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # First run or input changed
    if "password_correct" not in st.session_state:
        st.session_state["credentials"] = {
            "admin": "admin123",  # Replace with your desired username/password
            "user": "user123"
        }
        st.session_state["password_correct"] = False
        st.session_state["username"] = ""

    # Show input for password
    if not st.session_state["password_correct"]:
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        if "password" in st.session_state:
            st.error("😕 User not known or password incorrect")
        return False
    return True

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'price_history' not in st.session_state:
    st.session_state.price_history = []
if 'signals' not in st.session_state:
    st.session_state.signals = []

def run_bot():
    """Run the trading bot in a separate thread"""
    while st.session_state.is_running:
        try:
            # Update price history
            current_price = st.session_state.bot.fetch_market_data()
            if current_price:
                st.session_state.price_history.append({
                    'timestamp': pd.Timestamp.now(),
                    'price': current_price
                })
                # Keep only last 100 prices
                st.session_state.price_history = st.session_state.price_history[-100:]

            # Check for signals
            df = st.session_state.bot.calculate_indicators()
            if df is not None:
                st.session_state.bot.check_signals(df)

            time.sleep(60)  # Check every minute
        except Exception as e:
            st.error(f"Error in bot thread: {str(e)}")
            time.sleep(60)

def create_price_chart():
    """Create a price chart with indicators"""
    if not st.session_state.price_history:
        return None

    df = pd.DataFrame(st.session_state.price_history)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3])

    # Price chart
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['price'],
                  mode='lines', name='BTC Price',
                  line=dict(color='#00ff00')),
        row=1, col=1
    )

    # Volume chart
    fig.add_trace(
        go.Bar(x=df['timestamp'], y=df['price'],
               name='Volume', marker_color='#00ff00'),
        row=2, col=1
    )

    fig.update_layout(
        title='BTC Price and Volume',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        yaxis2_title='Volume',
        height=800,
        template='plotly_dark'
    )

    return fig

def main():
    st.title("📈 BTC Trading Bot Dashboard")

    # Check password before showing content
    if not check_password():
        st.stop()

    # Sidebar for controls
    with st.sidebar:
        st.header("Bot Controls")
        
        if not st.session_state.is_running:
            if st.button("Start Bot", type="primary"):
                st.session_state.bot = TradingBot()
                st.session_state.is_running = True
                threading.Thread(target=run_bot, daemon=True).start()
                st.success("Bot started successfully!")
        else:
            if st.button("Stop Bot", type="secondary"):
                st.session_state.is_running = False
                st.warning("Bot stopped!")

        st.header("Configuration")
        st.number_input("Signal Strength Threshold", 
                       min_value=0, max_value=100, 
                       value=config.SIGNAL_STRENGTH_THRESHOLD)
        
        st.number_input("RSI Oversold", 
                       min_value=0, max_value=100, 
                       value=config.RSI_OVERSOLD)
        
        st.number_input("RSI Overbought", 
                       min_value=0, max_value=100, 
                       value=config.RSI_OVERBOUGHT)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Price Chart")
        fig = create_price_chart()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for price data...")

    with col2:
        st.header("Bot Status")
        if st.session_state.is_running:
            st.success("Bot is running")
            if st.session_state.price_history:
                current_price = st.session_state.price_history[-1]['price']
                st.metric("Current BTC Price", f"${current_price:,.2f}")
        else:
            st.warning("Bot is stopped")

        st.header("Recent Signals")
        for signal in st.session_state.signals[-5:]:
            with st.expander(f"Signal at {signal['timestamp']}"):
                st.write(signal['message'])

if __name__ == "__main__":
    main() 