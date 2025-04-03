import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Exchange Configuration
EXCHANGE = "hyperliquid"  # Using HyperLiquid as the exchange
SYMBOL = "BTC"  # Trading pair
TIMEFRAME = "1h"     # 1-hour timeframe

# API Configuration
API_KEY = "your_api_key_here"
API_SECRET = "your_api_secret_here"

# Trading Parameters
LEVERAGE = 2  # Trading leverage (1-125)
POSITION_SIZE = 0.01  # Size of each trade in BTC
MAX_POSITIONS = 1     # Maximum number of concurrent positions
STOP_LOSS_PERCENTAGE = 2.0  # Stop loss percentage
TAKE_PROFIT_PERCENTAGE = 4.0  # Take profit percentage

# Technical Analysis Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Risk Management
MAX_DAILY_LOSS = 100  # Maximum daily loss in USDT
MAX_DRAWDOWN = 10  # Maximum drawdown percentage
MAX_TRADES_PER_DAY = 5

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "trading_bot.log"

# Error Messages
ERROR_MESSAGES = {
    "api_error": "Error connecting to exchange API.",
    "insufficient_funds": "Insufficient funds for trade.",
    "invalid_order": "Invalid order parameters.",
    "market_closed": "Market is currently closed.",
    "rate_limit": "Exchange rate limit reached.",
    "unknown_error": "An unknown error occurred.",
    "daily_loss_limit": "Maximum daily loss limit reached."
}

# Trading Parameters
TRADE_SIZE = 0.001
STOP_LOSS_PERCENTAGE = 2.0
TAKE_PROFIT_PERCENTAGE = 3.0

# Technical Indicators
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_PERIOD = 20
BB_STD_DEV = 2

SMA_FAST = 20
SMA_SLOW = 50
EMA_FAST = 20

STOCH_K = 14
STOCH_D = 3
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80

# Volume Analysis
VOLUME_SPIKE_THRESHOLD = 1.5  # 50% above average
VWAP_PERIOD = 20

# Trend Analysis
ADX_PERIOD = 14
ADX_STRONG_TREND = 25

# Signal Parameters
SIGNAL_COOLDOWN = 3600  # 1 hour cooldown between signals
SIGNAL_STRENGTH_THRESHOLD = 50  # Minimum signal strength to trigger alert

