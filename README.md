# BTC Trading Bot Dashboard

A real-time Bitcoin trading bot dashboard with technical analysis and trading signals.

## Features

- Real-time BTC price monitoring
- Technical indicators (RSI, MACD, VWAP)
- Trading signals and alerts
- Interactive price charts
- Volume analysis
- Risk metrics

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your credentials:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   API_KEY=your_api_key
   API_SECRET=your_api_secret
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Live Demo

Visit [your-streamlit-url] to see the live dashboard.

## Security

- Login required to access the dashboard
- Environment variables for sensitive data
- Rate limiting on API calls
- Secure WebSocket connections

## Contributing

Feel free to submit issues and enhancement requests! 