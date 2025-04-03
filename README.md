# Bitcoin Futures Trading Bot

A Python-based trading bot for Bitcoin futures with leverage trading on HyperLiquid. The bot uses technical analysis (RSI and MACD) to make trading decisions and includes risk management features.

## Features

- Automated Bitcoin futures trading with leverage on HyperLiquid
- Technical analysis using RSI and MACD indicators
- Risk management with stop-loss and take-profit orders
- Daily loss limits and maximum drawdown protection
- Detailed logging of all trades and operations
- Configurable trading parameters

## Prerequisites

- Python 3.8 or higher
- HyperLiquid account with futures trading enabled
- HyperLiquid API key and secret with futures trading permissions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd btc_trading_bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project directory and add your HyperLiquid API credentials:
```
HYPERLIQUID_API_KEY=your_api_key_here
HYPERLIQUID_API_SECRET=your_api_secret_here
```

## Configuration

You can modify the trading parameters in `config.py`:

- `LEVERAGE`: Trading leverage (1-125)
- `POSITION_SIZE`: Size of each trade in BTC
- `STOP_LOSS_PERCENTAGE`: Stop loss percentage
- `TAKE_PROFIT_PERCENTAGE`: Take profit percentage
- `MAX_DAILY_LOSS`: Maximum daily loss percentage
- `MAX_DRAWDOWN`: Maximum drawdown percentage
- `MAX_TRADES_PER_DAY`: Maximum number of trades per day

## Usage

1. Start the trading bot:
```bash
python trading_bot.py
```

2. The bot will:
   - Initialize the trading environment
   - Set up the specified leverage
   - Start monitoring the market
   - Execute trades based on technical signals
   - Manage positions with stop-loss and take-profit orders
   - Log all activities to `trading_bot.log`

## Risk Warning

Trading cryptocurrency futures with leverage is extremely risky and can result in significant losses. This bot is for educational purposes only. Always:

- Start with small amounts
- Monitor the bot's performance
- Set appropriate risk limits
- Never trade with money you cannot afford to lose

## Disclaimer

This trading bot is provided as-is without any guarantees. The creators are not responsible for any financial losses incurred through the use of this bot. Use at your own risk. 