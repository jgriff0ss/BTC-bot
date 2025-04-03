import ccxt
import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from datetime import datetime
import logging
import time
import os
import json
import requests
from dotenv import load_dotenv
from config import *
import hmac
import hashlib
import ta

# Load environment variables
load_dotenv()

class BitcoinTradingBot:
    def __init__(self):
        self.wallet_address = os.getenv('WALLET_ADDRESS')
        self.api_key = os.getenv('HYPERLIQUID_API_KEY')
        self.api_secret = os.getenv('HYPERLIQUID_API_SECRET')
        
        if not self.wallet_address:
            raise ValueError("Wallet address not found in environment variables")
        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials not found in environment variables")
            
        self.base_url = "https://api.hyperliquid.xyz"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize trading state
        self.current_position = None
        self.daily_trades = 0
        self.daily_pnl = 0
        self.initial_balance = 0
        self.max_daily_loss = 0
        self.max_drawdown = 0
        self.peak_balance = 0
        self.price_history = []
        
        # Setup Telegram
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if not self.telegram_token or not self.telegram_chat_id:
            self.logger.warning("Telegram credentials not found. Alerts will only be logged locally.")

        # Initialize liquidity tracking
        self.liquidity_history = []
        self.last_liquidity_alert = None
        self.liquidity_alert_threshold = 0.1  # 10% change threshold

        # Initialize signal tracking
        self.last_signal_time = None
        self.signal_cooldown = 3600  # 1 hour cooldown between signals
        self.signal_strength = 0  # Track signal strength

    def send_telegram_message(self, message):
        """Send a message to Telegram."""
        try:
            if not self.telegram_token or not self.telegram_chat_id:
                return

            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=data)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to send Telegram message: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {str(e)}")

    def _make_request(self, endpoint, data):
        """Make a request to the HyperLiquid API."""
        try:
            url = f"{self.base_url}{endpoint}"
            headers = {
                "Content-Type": "application/json"
            }

            # Add authentication headers for exchange endpoints
            if endpoint == "/exchange":
                timestamp = str(int(time.time()))
                headers.update({
                    "Authorization": f"Basic {self.wallet_address}",
                    "X-Timestamp": timestamp
                })

            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"Request error: {str(e)}")
            return None

    def _generate_signature(self, data, timestamp):
        """Generate signature for authenticated requests."""
        try:
            if not self.api_secret:
                self.logger.error("API secret not found")
                return None

            # Create signature string
            message = f"{timestamp}{json.dumps(data)}"
            signature = hmac.new(
                self.api_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            self.logger.error(f"Error generating signature: {str(e)}")
            return None

    def _get_balance(self):
        """Get current wallet balance"""
        try:
            # Get user account info
            response = self._make_request("/info", {
                "type": "clearinghouse",
                "user": self.wallet_address
            })
            
            if response and isinstance(response, list) and len(response) > 0:
                # Get USDC balance from the first element
                user_state = response[0]
                if isinstance(user_state, dict):
                    margin_summary = user_state.get('marginSummary', {})
                    if margin_summary:
                        account_value = float(margin_summary.get('accountValue', 0))
                        self.logger.info(f"Current account value: {account_value} USDC")
                        return account_value
                    else:
                        self.logger.warning("No margin summary found in user state")
                else:
                    self.logger.warning("Invalid user state format")
            else:
                self.logger.warning("Empty or invalid response from API")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to get balance: {str(e)}")
            return 0

    def setup(self):
        """Initialize the trading environment"""
        try:
            # Fetch initial market data
            market_data = self._make_request("/info", {"type": "metaAndAssetCtxs"})
            if not market_data or not isinstance(market_data, list) or len(market_data) < 2:
                self.logger.error("Failed to fetch market data")
                return False
                
            meta = market_data[0]
            assets = market_data[1]
            
            # Find BTC metadata
            btc_meta = next((coin for coin in meta.get('universe', []) if coin.get('name') == 'BTC'), None)
            if not btc_meta:
                self.logger.error("BTC not found in universe list")
                return False
                
            # Get BTC market data
            btc_data = assets[0]  # BTC is typically the first asset
            if not btc_data:
                self.logger.error("Failed to get BTC market data")
                return False
                
            btc_price = float(btc_data.get('markPx', 0))
            max_leverage = int(btc_meta.get('maxLeverage', 40))  # Default to 40 if not found
            
            if not btc_price:
                self.logger.error("Failed to get BTC price")
                return False
            
            # Get initial balance
            self.initial_balance = self._get_balance()
            if not self.initial_balance:
                self.logger.error("Failed to get wallet balance. Please check if your wallet has sufficient funds.")
                return False
                
            # Calculate minimum required balance for trading
            position_size = float(os.getenv('POSITION_SIZE', 0.01))
            leverage = min(int(os.getenv('LEVERAGE', 5)), max_leverage)  # Ensure we don't exceed max leverage
            min_required = (position_size * btc_price) / leverage
            
            self.logger.info(f"Current BTC price: {btc_price:.2f} USDT")
            self.logger.info(f"Maximum leverage available: {max_leverage}x")
            self.logger.info(f"Your wallet balance: {self.initial_balance:.2f} USDT")
            self.logger.info(f"Minimum required balance for trading: {min_required:.2f} USDT")
            
            if self.initial_balance < min_required:
                self.logger.error(f"Insufficient balance. You need at least {min_required:.2f} USDT to trade with the current settings.")
                return False
                
            self.peak_balance = self.initial_balance
            self.max_daily_loss = self.initial_balance * (float(os.getenv('MAX_DAILY_LOSS_PERCENTAGE', 5.0)) / 100)
            self.max_drawdown = self.initial_balance * (float(os.getenv('MAX_DRAWDOWN_PERCENTAGE', 10.0)) / 100)
            
            self.logger.info(f"Maximum daily loss: {self.max_daily_loss:.2f} USDT")
            self.logger.info(f"Maximum drawdown: {self.max_drawdown:.2f} USDT")
            
            # Initialize price history
            self.price_history.append({
                'timestamp': datetime.now(),
                'price': btc_price
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            return False

    def fetch_market_data(self):
        """Fetch current market data and update price history."""
        try:
            market_data = self._make_request("/info", {"type": "metaAndAssetCtxs"})
            if market_data and isinstance(market_data, list) and len(market_data) >= 2:
                # First element contains metadata, second contains market data
                meta = market_data[0]
                assets = market_data[1]
                
                # Find BTC in the universe list
                btc_meta = next((coin for coin in meta.get('universe', []) if coin.get('name') == 'BTC'), None)
                if not btc_meta:
                    self.logger.error("BTC not found in universe list")
                    return None
                    
                # Find BTC in the assets list
                btc_data = assets[0]  # BTC is typically the first asset
                if btc_data:
                    mark_price = float(btc_data.get('markPx', 0))
                    if mark_price:
                        self.price_history.append({
                            'timestamp': datetime.now(),
                            'price': mark_price
                        })
                        
                        # Keep only the last 100 price points
                        if len(self.price_history) > 100:
                            self.price_history = self.price_history[-100:]
                        
                        return mark_price
                return None
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return None

    def fetch_liquidity_data(self):
        """Fetch and analyze liquidity pool data at different levels."""
        try:
            # Get order book data
            orderbook_data = self._make_request("/info", {"type": "l2Book", "coin": "BTC"})
            if not orderbook_data:
                return None

            # Analyze different price levels
            price_levels = {
                'near': 0.001,  # 0.1% from current price
                'medium': 0.01,  # 1% from current price
                'far': 0.05     # 5% from current price
            }

            # Get current price
            current_price = float(orderbook_data.get('markPx', 0))
            if not current_price:
                return None

            # Initialize liquidity analysis
            liquidity_analysis = {
                'total_liquidity': 0,
                'levels': {},
                'volume_profile': {
                    'bids': {},
                    'asks': {}
                }
            }

            # Analyze bids (buy orders)
            for bid in orderbook_data.get('bids', []):
                price = float(bid[0])
                size = float(bid[1])
                liquidity_analysis['total_liquidity'] += size

                # Calculate distance from current price
                distance = (current_price - price) / current_price

                # Categorize by distance
                for level, threshold in price_levels.items():
                    if distance <= threshold:
                        if level not in liquidity_analysis['levels']:
                            liquidity_analysis['levels'][level] = {'bids': 0, 'asks': 0}
                        liquidity_analysis['levels'][level]['bids'] += size

                # Add to volume profile
                price_key = f"{price:.2f}"
                liquidity_analysis['volume_profile']['bids'][price_key] = size

            # Analyze asks (sell orders)
            for ask in orderbook_data.get('asks', []):
                price = float(ask[0])
                size = float(ask[1])
                liquidity_analysis['total_liquidity'] += size

                # Calculate distance from current price
                distance = (price - current_price) / current_price

                # Categorize by distance
                for level, threshold in price_levels.items():
                    if distance <= threshold:
                        if level not in liquidity_analysis['levels']:
                            liquidity_analysis['levels'][level] = {'bids': 0, 'asks': 0}
                        liquidity_analysis['levels'][level]['asks'] += size

                # Add to volume profile
                price_key = f"{price:.2f}"
                liquidity_analysis['volume_profile']['asks'][price_key] = size

            # Store liquidity data
            current_time = datetime.now()
            self.liquidity_history.append({
                'timestamp': current_time,
                'liquidity': liquidity_analysis['total_liquidity'],
                'analysis': liquidity_analysis
            })

            # Keep only last 24 hours of liquidity data
            cutoff_time = current_time - pd.Timedelta(hours=24)
            self.liquidity_history = [
                data for data in self.liquidity_history 
                if data['timestamp'] > cutoff_time
            ]

            # Check for significant liquidity changes
            if len(self.liquidity_history) >= 2:
                current_data = self.liquidity_history[-1]
                previous_data = self.liquidity_history[-2]
                
                # Check total liquidity change
                liquidity_change = (current_data['liquidity'] - previous_data['liquidity']) / previous_data['liquidity']
                
                # Check level-specific changes
                level_changes = {}
                for level in price_levels.keys():
                    if level in current_data['analysis']['levels'] and level in previous_data['analysis']['levels']:
                        current_bids = current_data['analysis']['levels'][level]['bids']
                        current_asks = current_data['analysis']['levels'][level]['asks']
                        previous_bids = previous_data['analysis']['levels'][level]['bids']
                        previous_asks = previous_data['analysis']['levels'][level]['asks']
                        
                        bid_change = (current_bids - previous_bids) / previous_bids if previous_bids > 0 else 0
                        ask_change = (current_asks - previous_asks) / previous_asks if previous_asks > 0 else 0
                        
                        level_changes[level] = {
                            'bids': bid_change,
                            'asks': ask_change
                        }

                # Alert on significant changes
                if abs(liquidity_change) >= self.liquidity_alert_threshold:
                    if (not self.last_liquidity_alert or 
                        (current_time - self.last_liquidity_alert).total_seconds() > 3600):
                        
                        message = (
                            f"💧 <b>Liquidity Analysis Alert</b> 💧\n\n"
                            f"Current Price: ${current_price:,.2f}\n"
                            f"Total Liquidity Change: {liquidity_change*100:.2f}%\n"
                            f"Current Total Liquidity: {current_data['liquidity']:.2f} BTC\n\n"
                            f"<b>Level Analysis:</b>\n"
                        )

                        for level, changes in level_changes.items():
                            message += (
                                f"\n{level.title()} Level ({price_levels[level]*100:.1f}%):\n"
                                f"Bid Change: {changes['bids']*100:.2f}%\n"
                                f"Ask Change: {changes['asks']*100:.2f}%\n"
                            )

                        # Add significant price levels
                        message += "\n<b>Significant Price Levels:</b>\n"
                        for side in ['bids', 'asks']:
                            sorted_levels = sorted(
                                current_data['analysis']['volume_profile'][side].items(),
                                key=lambda x: float(x[0]),
                                reverse=(side == 'bids')
                            )[:5]  # Top 5 levels
                            
                            message += f"\n{side.title()}:"
                            for price, size in sorted_levels:
                                message += f"\n${price}: {size:.2f} BTC"

                        message += "\n----------------------------------------"
                        
                        self.logger.info(message)
                        self.send_telegram_message(message)
                        self.last_liquidity_alert = current_time

            return liquidity_analysis['total_liquidity']

        except Exception as e:
            self.logger.error(f"Error fetching liquidity data: {str(e)}")
            return None

    def calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price"""
        df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3
        df['VP'] = df['Typical_Price'] * df['volume']
        df['VWAP'] = df['VP'].cumsum() / df['volume'].cumsum()
        return df

    def calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        df['TR'] = pd.DataFrame({
            'HL': abs(df['high'] - df['low']),
            'HD': abs(df['high'] - df['close'].shift(1)),
            'LD': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        
        df['+DM'] = (df['high'] - df['high'].shift(1)).clip(lower=0)
        df['-DM'] = (df['low'].shift(1) - df['low']).clip(lower=0)
        
        df['+DI'] = 100 * (df['+DM'].ewm(span=period).mean() / df['TR'].ewm(span=period).mean())
        df['-DI'] = 100 * (df['-DM'].ewm(span=period).mean() / df['TR'].ewm(span=period).mean())
        
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        df['ADX'] = df['DX'].ewm(span=period).mean()
        return df

    def detect_support_resistance(self, df, window=20):
        """Detect support and resistance levels"""
        df['High_Roll_Max'] = df['high'].rolling(window=window, center=True).max()
        df['Low_Roll_Min'] = df['low'].rolling(window=window, center=True).min()
        return df['High_Roll_Max'].iloc[-1], df['Low_Roll_Min'].iloc[-1]

    def calculate_risk_metrics(self, current_price, support_level, resistance_level):
        """Calculate risk metrics for the trade"""
        risk_reward_ratio = abs((resistance_level - current_price) / (current_price - support_level))
        position_size = self.calculate_position_size(current_price, support_level)
        return risk_reward_ratio, position_size

    def calculate_position_size(self, current_price, stop_loss_price):
        """Calculate dynamic position size based on volatility"""
        risk_per_trade = float(config.MAX_DAILY_LOSS) * 0.1  # 10% of max daily loss
        price_risk = abs(current_price - stop_loss_price)
        return round(risk_per_trade / price_risk, 8)

    def check_signals(self, df):
        """Enhanced signal checking with volume and risk analysis"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Calculate additional indicators
            df = self.calculate_vwap(df)
            df = self.calculate_adx(df)
            resistance_level, support_level = self.detect_support_resistance(df)
            risk_reward_ratio, suggested_position_size = self.calculate_risk_metrics(
                current_price, support_level, resistance_level
            )

            # Volume analysis
            volume_sma = df['volume'].rolling(window=20).mean()
            volume_spike = df['volume'].iloc[-1] > volume_sma.iloc[-1] * 1.5

            # Signal strength calculation
            signal_strength = 0
            signal_reasons = []
            
            # RSI Analysis
            rsi = df['rsi'].iloc[-1]
            if rsi < config.RSI_OVERSOLD:
                signal_strength += 20
                signal_reasons.append(f"RSI oversold ({rsi:.2f})")
            elif rsi > config.RSI_OVERBOUGHT:
                signal_strength -= 20
                signal_reasons.append(f"RSI overbought ({rsi:.2f})")

            # MACD Analysis
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            if macd > macd_signal and macd - macd_signal > 0:
                signal_strength += 20
                signal_reasons.append("MACD bullish crossover")
            elif macd < macd_signal and macd_signal - macd > 0:
                signal_strength -= 20
                signal_reasons.append("MACD bearish crossover")

            # ADX Trend Strength
            adx = df['ADX'].iloc[-1]
            if adx > 25:
                signal_strength += 10
                signal_reasons.append(f"Strong trend (ADX: {adx:.2f})")

            # Volume Analysis
            if volume_spike:
                signal_strength += 10
                signal_reasons.append("Unusual volume spike detected")

            # VWAP Analysis
            if current_price < df['VWAP'].iloc[-1]:
                signal_strength += 10
                signal_reasons.append("Price below VWAP")
            else:
                signal_strength -= 10
                signal_reasons.append("Price above VWAP")

            # Generate signal if strength threshold is met
            if abs(signal_strength) >= config.SIGNAL_STRENGTH_THRESHOLD:
                signal_type = "LONG" if signal_strength > 0 else "SHORT"
                
                message = (
                    f"🔔 <b>TRADING SIGNAL ALERT</b> 🔔\n\n"
                    f"💡 <b>Summary:</b> Consider a {signal_type.lower()} position at ${current_price:,.2f} because "
                    f"multiple indicators show {'oversold' if signal_type == 'LONG' else 'overbought'} conditions "
                    f"with strong {'bullish' if signal_type == 'LONG' else 'bearish'} momentum\n\n"
                    f"Signal Strength: {abs(signal_strength)}%\n"
                    f"Current BTC Price: ${current_price:,.2f}\n\n"
                    f"<b>Risk Metrics:</b>\n"
                    f"Risk/Reward Ratio: {risk_reward_ratio:.2f}\n"
                    f"Suggested Position Size: {suggested_position_size} BTC\n"
                    f"Support Level: ${support_level:,.2f}\n"
                    f"Resistance Level: ${resistance_level:,.2f}\n\n"
                    f"<b>Technical Indicators:</b>\n"
                    f"RSI: {rsi:.2f}\n"
                    f"MACD: {macd:.2f}\n"
                    f"MACD Signal: {macd_signal:.2f}\n"
                    f"ADX: {adx:.2f}\n"
                    f"VWAP: ${df['VWAP'].iloc[-1]:,.2f}\n"
                    f"Volume vs Average: {(df['volume'].iloc[-1] / volume_sma.iloc[-1]):,.2f}x\n\n"
                    f"<b>Signal Reasons:</b>\n"
                    + "\n".join(f"• {reason}" for reason in signal_reasons) + "\n\n"
                    f"<b>Suggested Action:</b> Consider opening a {signal_type} position\n"
                    f"----------------------------------------"
                )
                
                self.logger.info(message)
                self.send_telegram_message(message)

        except Exception as e:
            self.logger.error(f"Error checking signals: {str(e)}")
            
    def calculate_indicators(self):
        """Calculate technical indicators with enhanced metrics"""
        try:
            # Fetch historical data for calculations
            data = self.exchange.fetch_ohlcv('BTC/USDT', '1m', limit=100)
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=config.RSI_PERIOD)

            # MACD
            macd = ta.trend.macd(
                df['close'],
                window_slow=config.MACD_SLOW,
                window_fast=config.MACD_FAST,
                window_sign=config.MACD_SIGNAL
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()

            # Enhanced indicators
            df = self.calculate_vwap(df)
            df = self.calculate_adx(df)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return None

    def check_risk_limits(self):
        """Check if risk limits are exceeded."""
        try:
            # Check daily trade limit
            if self.daily_trades >= MAX_TRADES_PER_DAY:
                self.logger.warning("Maximum daily trades reached")
                return False

            # Check daily loss limit
            if self.daily_pnl <= -MAX_DAILY_LOSS:
                self.logger.warning("Maximum daily loss reached")
                return False

            # Check drawdown
            current_price = self.fetch_market_data()
            if current_price:
                drawdown = (self.initial_balance - current_price) / self.initial_balance * 100
                if drawdown >= MAX_DRAWDOWN:
                    self.logger.warning("Maximum drawdown reached")
                    return False

            return True
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return False

    def execute_trade(self, signal):
        """Execute a trade based on the given signal ('long' or 'short')."""
        try:
            # Check risk limits before executing trade
            if not self.check_risk_limits():
                self.logger.error("Risk limits exceeded, cannot execute trade")
                return False

            # Get current market data
            market_data = self._make_request("/info", {"type": "metaAndAssetCtxs"})
            if not market_data or len(market_data) < 2:
                self.logger.error("Failed to get market data")
                return False

            # Find BTC metadata and market data
            btc_meta = None
            btc_data = None
            for asset in market_data[0]["universe"]:
                if asset["name"] == "BTC":
                    btc_meta = asset
                    break

            if not btc_meta:
                self.logger.error("Could not find BTC metadata")
                return False

            # Get current BTC price and size decimals
            size_decimals = btc_meta["szDecimals"]
            formatted_size = format(float(os.getenv('POSITION_SIZE', 0.01)), f'.{size_decimals}f')

            # Prepare order data
            order_data = {
                "type": "order",
                "user": self.wallet_address,
                "asset": "BTC",
                "isBuy": signal == "long",
                "sz": formatted_size,
                "limitPx": str(market_data[1][0]["markPx"]),  # Use mark price
                "reduceOnly": False,
                "cloid": str(int(time.time() * 1000))  # Client order ID
            }

            # Execute the trade
            response = self._make_request("/exchange", order_data)
            if not response:
                self.logger.error("Failed to execute trade")
                return False

            # Update current position details
            self.current_position = {
                "type": signal,
                "entry_price": float(market_data[1][0]["markPx"]),
                "amount": float(os.getenv('POSITION_SIZE', 0.01)),
                "stop_loss": float(market_data[1][0]["markPx"]) * (0.95 if signal == "long" else 1.05),  # 5% stop loss
                "take_profit": float(market_data[1][0]["markPx"]) * (1.1 if signal == "long" else 0.9)  # 10% take profit
            }

            self.logger.info(f"Opened {signal} position: {self.current_position}")
            return True

        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return False

    def check_position(self):
        """Check current position and close if stop loss or take profit is hit."""
        try:
            if not self.current_position:
                return

            current_price = self.fetch_market_data()
            if not current_price:
                return

            position = self.current_position

            # Check stop loss
            if (position['type'] == 'long' and current_price <= position['stop_loss']) or \
               (position['type'] == 'short' and current_price >= position['stop_loss']):
                self.close_position('stop_loss')

            # Check take profit
            elif (position['type'] == 'long' and current_price >= position['take_profit']) or \
                 (position['type'] == 'short' and current_price <= position['take_profit']):
                self.close_position('take_profit')

        except Exception as e:
            self.logger.error(f"Error checking position: {str(e)}")

    def close_position(self, reason):
        """Close the current position."""
        try:
            if not self.current_position:
                return

            position = self.current_position
            current_price = self.fetch_market_data()
            if not current_price:
                return

            # Calculate PnL
            if position['type'] == 'long':
                pnl = (current_price - position['entry_price']) / position['entry_price'] * 100 * float(os.getenv('LEVERAGE', 5))
            else:
                pnl = (position['entry_price'] - current_price) / position['entry_price'] * 100 * float(os.getenv('LEVERAGE', 5))

            # Close position
            order_data = {
                "type": "order",
                "user": self.wallet_address,
                "asset": "BTC",
                "isBuy": position['type'] == 'short',  # Buy to close short, sell to close long
                "limitPx": current_price,
                "sz": position['amount'],
                "reduceOnly": True,
                "cloid": str(int(time.time() * 1000))  # Client order ID
            }
            
            response = self._make_request("/exchange", order_data)
            if response:
                self.daily_pnl += pnl
                self.logger.info(f"Closed {position['type']} position at {current_price} ({reason})")
                self.logger.info(f"Position PnL: {pnl:.2f}%")
                self.current_position = None

        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")

    def send_test_signal(self):
        """Send a test signal to verify alert format."""
        try:
            current_price = self.fetch_market_data()
            if not current_price:
                current_price = 82367.00  # Fallback price for testing

            # Generate test signal
            message = (
                f"🔔 <b>TEST SIGNAL ALERT</b> 🔔\n\n"
                f"💡 <b>Summary:</b> Consider a long position at ${current_price:,.2f} because "
                f"multiple indicators show oversold conditions with strong bullish momentum\n\n"
                f"Signal Strength: 75%\n"
                f"Current BTC Price: ${current_price:,.2f}\n\n"
                f"<b>Technical Indicators:</b>\n"
                f"RSI: 28.50\n"
                f"MACD: 150.25\n"
                f"MACD Signal: 120.75\n"
                f"BB Upper: $84,500.00\n"
                f"BB Lower: $81,200.00\n"
                f"SMA20: $82,100.00\n"
                f"SMA50: $81,800.00\n"
                f"Stoch K: 25.30\n"
                f"Stoch D: 20.15\n\n"
                f"<b>Signal Reasons:</b>\n"
                f"• RSI oversold (28.50)\n"
                f"• MACD bullish crossover\n"
                f"• Price below lower BB\n"
                f"• Golden cross (SMA20 > SMA50)\n"
                f"• Stochastic oversold\n\n"
                f"<b>Suggested Action:</b> Consider opening a LONG position\n"
                f"----------------------------------------"
            )
            
            self.logger.info(message)
            self.send_telegram_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending test signal: {str(e)}")

    def run(self):
        """Main trading loop."""
        self.logger.info("Starting trading signal alerts...")
        self.send_telegram_message("🤖 BTC Trading Bot Started\nMonitoring for trading signals and liquidity changes...")
        
        # Send test signal
        self.send_test_signal()
        
        while True:
            try:
                # Fetch market data and update price history
                current_price = self.fetch_market_data()
                if current_price:
                    self.logger.info(f"Current BTC price: ${current_price:,.2f}")
                
                # Fetch and analyze liquidity data
                current_liquidity = self.fetch_liquidity_data()
                if current_liquidity:
                    self.logger.info(f"Current BTC liquidity: {current_liquidity:.2f} BTC")
                
                # Calculate indicators and check signals
                df = self.calculate_indicators()
                if df is not None:
                    self.check_signals(df)

                # Wait before next iteration
                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    bot = BitcoinTradingBot()
    bot.run() 