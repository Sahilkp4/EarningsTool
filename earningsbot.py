import os
import sys
import time
import logging
import asyncio
import json
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
#fix these comments when running on ubuntu aaabbccddD
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python 3.8 fallback
from typing import List, Dict, Optional, Set

import websockets
import finnhub
from alpaca_trade_api.rest import REST as AlpacaREST
from typing import List, Dict, Set, Optional



# ─── Constants & Configuration ───────────────────────────────────────────────
TZ = ZoneInfo("America/New_York")
load_dotenv('/home/ubuntu/.env')  # Load environment variables from .env file
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
APCA_KEY        = os.getenv("APCA_API_KEY_ID")
APCA_SECRET     = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

MONITOR_END_HOUR   = 15 #end hour
MONITOR_END_MINUTE = 33 #end minute

SURPRISE_THRESHOLD = 1 # % min earnings suprise
MAX_SURPRISE       = 600 # max suprise
MC_THRESHOLD       = 900_000_000 # market cap in millions of dollars
TRAIL_PERCENT      = 0.1  # 0.1%- very tight trailing stop

# NEW GLOBAL VARIABLES FOR JSON-DRIVEN TRADING
TOTAL_ACCOUNT_PCT = 100  # Percentage of total account to use for trading
TARGET_PROFIT_FACTOR = 1.5  # Multiplier for target profit from JSON bins

ALPACA_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"  # or v2/crypto or v2/stocks, adjust as needed
ALPACA_MAX_SUBSCRIBE = 30
ALPACA_SUBSCRIBE_WAIT = 5  # seconds wait per batch before switching
# ─── Logging Setup ────────────────────────────────────────────────────────────

def configure_logging() -> logging.Logger:
    logger = logging.getLogger("earnings_trader")
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s",
            "%Y-%m-%d %H:%M:%S %Z"
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        
        # Main session file handler
        fh = logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "session_output.txt"),
            mode="a", encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    
    return logger

def configure_daily_tickers_logger() -> logging.Logger:
    """Configure a separate logger for daily tickers output"""
    daily_logger = logging.getLogger("daily_tickers")
    daily_logger.setLevel(logging.INFO)
    
    if not daily_logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s",
            "%Y-%m-%d %H:%M:%S %Z"
        )
        
        # Daily tickers file handler
        fh = logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "dailytickers.txt"),
            mode="a", encoding="utf-8"
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        daily_logger.addHandler(fh)
        
        # Prevent propagation to avoid duplicate messages in main log
        daily_logger.propagate = False
    
    return daily_logger

# Initialize both loggers
logger = configure_logging()
daily_logger = configure_daily_tickers_logger()



# ─── Utils ───────────────────────────────────────────────────────────────────
def get_prev_business_day(ref: date) -> date:
    d = ref
    while True:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            logger.debug(f"Previous business day for {ref} is {d}")
            return d

def is_within_earnings_window(entry: dict, start: str, end: str) -> bool:
    ok = ((entry.get("date") == start and entry.get("hour") == "amc") or
          (entry.get("date") == end   and entry.get("hour") == "bmo"))
    logger.debug(f"{entry.get('symbol')} window check: {ok} (date={entry.get('date')}, hour={entry.get('hour')})")
    return ok

def sleep_until_market_open(api: AlpacaREST):
    logger.info("Starting market open wait loop")
    while True:
        """
        #FAKE CLOCK !!!!
        class FakeClock:
            is_open = True
            timestamp = datetime(2025, 6, 25, 9, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        clock = FakeClock()
        """
        # REAL CLOCK !!!
        clock = api.get_clock()

        logger.debug(f"Market status check: is_open={clock.is_open}, timestamp={clock.timestamp}")

        if clock.is_open:
            logger.info("Market opened at %s", clock.timestamp.astimezone(TZ))
            return

        # Convert next_open to a native datetime and normalize tz
        next_open_raw = clock.next_open
        if isinstance(next_open_raw, str):
            # Handle 'Z' suffix and other ISO format variations
            next_open_str = next_open_raw.replace('Z', '+00:00')
            next_open = datetime.fromisoformat(next_open_str)
        else:
            next_open_str = str(next_open_raw).replace('Z', '+00:00')
            next_open = datetime.fromisoformat(next_open_str)

        #  Always ensure timezone awareness
        if next_open.tzinfo is None:
            next_open = next_open.replace(tzinfo=ZoneInfo("UTC"))

        # Convert to target timezone
        next_open = next_open.astimezone(TZ)

        now = datetime.now(TZ)

        try:
            secs = (next_open - now).total_seconds()
        except Exception as e:
            logger.exception(f"❌ Failed to compute time delta: next_open={next_open}, now={now}")
            raise e

        logger.debug(f"Market closed. Now={now.time()}, Next open={next_open.time()} in {secs:.1f}s")

        if secs > 300:
            logger.debug("Sleeping for 5 minutes until next check")
            time.sleep(300)
        else:
            logger.debug(f"Close to open time, sleeping for {max(0, secs):.1f}s")
            time.sleep(max(0, secs))



def get_cutoff_time(today: date) -> datetime:
    cutoff = datetime(
        today.year, today.month, today.day,
        MONITOR_END_HOUR, MONITOR_END_MINUTE, tzinfo=TZ
    )
    logger.debug(f"Monitor cutoff time set to {cutoff.time()}")
    return cutoff

# ─── Core Logic ───────────────────────────────────────────────────────────────
def fetch_earnings_calendar(fh: finnhub.Client, frm: str, to: str) -> List[dict]:
    logger.debug(f"Fetching earnings calendar from {frm} to {to}")
    try:
        cal = fh.earnings_calendar(symbol=None, _from=frm, to=to)
        entries = cal.get("earningsCalendar", [])
        logger.info(f"Fetched {len(entries)} earnings entries from {frm} to {to}")
        if not entries:
            logger.warning("No earnings entries found in calendar")
        return entries
    except Exception as e:
        logger.error(f"Failed to fetch earnings calendar: {e}")
        return []

def filter_candidates(
    entries: List[dict],
    fh: finnhub.Client,
    frm: str,
    to: str
) -> List[dict]:
    logger.info(f"Starting candidate filtering for {len(entries)} entries")
    candidates: List[dict] = []
    last_profile_time = 0.0
    filtered_count = {
        'missing_data': 0,
        'poor_surprise': 0,
        'wrong_window': 0,
        'profile_error': 0,
        'high_mc': 0,
        'passed': 0
    }

    for e in entries:
        sym = e.get("symbol")
        est, act = e.get("epsEstimate"), e.get("epsActual")
        logger.debug(f"Evaluating {sym}: est={est}, act={act}, date={e.get('date')}, hour={e.get('hour')}")

        if est is None or act is None or est == 0 or act < 0:
            logger.debug(f"Skipping {sym}: missing/zero estimate or negative actual")
            filtered_count['missing_data'] += 1
            continue

        surprise = (act - est) / abs(est) * 100
        logger.debug(f"{sym} surprise={surprise:.2f}%")
        if surprise <= SURPRISE_THRESHOLD or surprise > MAX_SURPRISE:
            logger.debug(f"Skipping {sym}: surprise {surprise:.2f}% outside [{SURPRISE_THRESHOLD}, {MAX_SURPRISE}]")
            filtered_count['poor_surprise'] += 1
            continue

        if not is_within_earnings_window(e, frm, to):
            logger.debug(f"Skipping {sym}: not in earnings window")
            filtered_count['wrong_window'] += 1
            continue

        # rate-limit company_profile2
        now = time.monotonic()
        if now - last_profile_time < 1.1:
            pause = 1.1 - (now - last_profile_time)
            logger.debug(f"Rate limiting: sleeping {pause:.2f}s before profile fetch for {sym}")
            time.sleep(pause)

        try:
            logger.debug(f"Fetching company profile for {sym}")
            profile = fh.company_profile2(symbol=sym)
            last_profile_time = time.monotonic()
            mc = float(profile.get("marketCapitalization", 0))
            logger.debug(f"{sym} profile: MC={mc:.1f}M, country={profile.get('country')}")
        except Exception as exc:
            logger.warning(f"Profile fetch failed for {sym}: {exc}")
            filtered_count['profile_error'] += 1
            continue

        if mc <= MC_THRESHOLD:
            candidates.append({"symbol": sym, "surprise": surprise, "market_cap": mc})
            logger.info(f"✓ Candidate {sym}: surprise={surprise:.2f}%, mc={mc:.1f}M")
            filtered_count['passed'] += 1
        else:
            logger.debug(f"Skipping {sym}: MC {mc:.1f}M > {MC_THRESHOLD}M")
            filtered_count['high_mc'] += 1

    logger.info(f"Filtering results: {filtered_count}")
    logger.info(f"Total candidates after filter: {len(candidates)}")
    return candidates


def group_candidates_by_surprise(
    cands: List[Dict[str, float]]
) -> List[Dict[str, float]]:
    logger.debug(f"Sorting {len(cands)} candidates by surprise")
    sorted_c = sorted(cands, key=lambda x: x["surprise"], reverse=True)
    
    # Log message that goes to both main log and dailytickers.txt
    message = "Sorted candidates by surprise: " + \
              ", ".join(f"{c['symbol']}({c['surprise']:.1f}%)" for c in sorted_c)
    
    # Log to main logger (console + session_output.txt)
    logger.info(message)
    
    # Log to daily tickers file
    daily_logger.info(message)
    
    return sorted_c

def preload_prev_closes(
    api: AlpacaREST,
    candidates: List[Dict[str, float]],
    day: str
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    logger.info(f"Preloading previous closes for {len(candidates)} symbols on {day}")
    start_iso = f"{day}T00:00:00-04:00"
    end_iso   = f"{day}T23:59:59-04:00"

    for c in candidates:
        sym = c["symbol"]
        try:
            logger.debug(f"Fetching previous close for {sym}")
            bars = api.get_bars(
                sym, timeframe="1Day",
                start=start_iso, end=end_iso,
                limit=1, adjustment="raw"
            )
            if bars:
                out[sym] = bars[0].c
                logger.debug(f"{sym} prev_close=${bars[0].c:.2f} (volume: {bars[0].v})")
            else:
                logger.warning(f"No bar data found for {sym} on {day}")
        except Exception as e:
            logger.warning(f"Failed to fetch prev_close for {sym}: {e}")
        time.sleep(0.1)  # Rate limiting

    logger.info(f"Successfully loaded {len(out)} previous closes")
    return out


# ─── NEW JSON-DRIVEN TRADING FUNCTIONS ──────────────────────────────────────

def load_earnings_bins(filepath: str = "earnings_bins.json") -> Dict:
    """Load earnings bins configuration from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"✅ Loaded earnings bins from {filepath}")
        logger.info(f"   Minimum candidates required: {data['metadata']['minimum_candidates_analysis']['recommended_minimum']}")
        logger.info(f"   Number of bins: {len(data['bins'])}")
        return data
    except Exception as e:
        logger.error(f"❌ Failed to load earnings bins from {filepath}: {e}")
        raise

def find_bin_for_percentage(pct_increase: float, bins: List[Dict]) -> Optional[Dict]:
    """Find the appropriate bin for a given percentage increase"""
    for bin_data in bins:
        if bin_data["range_min"] <= pct_increase < bin_data["range_max"]:
            logger.debug(f"Found bin for {pct_increase:.2f}%: {bin_data['range_min']:.1f}%-{bin_data['range_max']:.1f}%")
            return bin_data
    logger.debug(f"No bin found for {pct_increase:.2f}%")
    return None

def calculate_position_size(total_cash: float, min_candidates: int) -> float:
    """Calculate position size based on total cash and minimum candidates"""
    position_size = (total_cash * (TOTAL_ACCOUNT_PCT / 100)) / min_candidates
    logger.debug(f"Position size: ${position_size:.2f} (cash: ${total_cash:.2f}, factor: {TOTAL_ACCOUNT_PCT}%, candidates: {min_candidates})")
    return position_size

class TradeManager:
    """Manages multiple trades simultaneously"""
    
    def __init__(self, api: AlpacaREST, finnhub_key: str, apca_key: str, apca_secret: str):
        self.api = api
        self.finnhub_key = finnhub_key
        self.apca_key = apca_key
        self.apca_secret = apca_secret
        self.active_trades = {}  # symbol -> trade_info
        self.price_data = {}  # symbol -> latest_price
        self.price_lock = asyncio.Lock()
        self.cutoff_time = get_cutoff_time(datetime.now(TZ).date())
        
    async def add_trade(self, symbol: str, order_info: Dict, bin_data: Dict):
        """Add a new trade to monitor"""
        trade_info = {
            'symbol': symbol,
            'quantity': order_info['qty'],
            'entry_price': order_info['filled_avg_price'],
            'order_id': order_info['order_id'],
            'child_orders': order_info.get('child_orders', []),
            'bin_data': bin_data,
            'target_hit': False,
            'peak_price': order_info['filled_avg_price'],
            'trailing_stop_active': False,
            'alpaca_trailing_order': None
        }
        
        async with self.price_lock:
            self.active_trades[symbol] = trade_info
            self.price_data[symbol] = order_info['filled_avg_price']
        
        logger.info(f"📊 Added trade to monitor: {symbol} @ ${trade_info['entry_price']:.2f}")
        logger.info(f"   Stop loss: {bin_data['stop_loss']}%, Target: {bin_data['target_profit'] * TARGET_PROFIT_FACTOR:.1f}%")

    async def place_bracket_order(self, symbol: str, current_price: float, position_size: float, bin_data: Dict) -> Optional[Dict]:
        """Place a bracket order with stop loss and take profit"""
        try:
            # Calculate quantity based on position size
            qty = int(position_size / current_price)
            if qty <= 0:
                logger.error(f"❌ Invalid quantity for {symbol}: {qty}")
                return None
            
            # Calculate stop loss and target prices
            stop_loss_pct = bin_data['stop_loss'] / 100  # Convert to decimal
            target_profit_pct = (bin_data['target_profit']) / 100  # Convert to decimal
            
            stop_loss_price = current_price * (1 + stop_loss_pct)  # stop_loss is negative, so this reduces price
            target_price = current_price * (1 + (target_profit_pct * TARGET_PROFIT_FACTOR))
            
            logger.info(f"📊 Placing bracket order for {symbol}:")
            logger.info(f"   Qty: {qty} @ ${current_price:.2f}")
            logger.info(f"   Stop Loss: ${stop_loss_price:.2f} ({bin_data['stop_loss']}%)")
            logger.info(f"   Target: ${target_price:.2f} ({bin_data['target_profit'] * TARGET_PROFIT_FACTOR:.1f}%)")
            
            order = await asyncio.to_thread(
                self.api.submit_order,
                symbol=symbol,
                qty=qty,
                side="buy",
                type="market",
                time_in_force="day",
                order_class="bracket",
                stop_loss={"stop_price": stop_loss_price},
                take_profit={"limit_price": target_price}
            )
            
            # Poll for fill
            max_polls = 20
            for poll in range(max_polls):
                order_status = await asyncio.to_thread(self.api.get_order, order.id)
                
                if order_status.status == "filled":
                    filled_price = float(order_status.filled_avg_price)
                    filled_qty = int(order_status.filled_qty)
                    
                    # Get child orders
                    child_orders = []
                    await asyncio.sleep(1)  # Wait for child orders to be created
                    try:
                        all_orders = await asyncio.to_thread(self.api.list_orders, status="open", symbols=[symbol])
                        for child_order in all_orders:
                            if hasattr(child_order, 'parent_id') and child_order.parent_id == order.id:
                                child_orders.append({
                                    "id": child_order.id,
                                    "type": child_order.order_type,
                                    "side": child_order.side
                                })
                    except Exception as e:
                        logger.warning(f"Could not retrieve child orders for {symbol}: {e}")
                    
                    logger.info(f"✅ {symbol} filled: {filled_qty} @ ${filled_price:.2f}")
                    return {
                        'symbol': symbol,
                        'qty': filled_qty,
                        'filled_avg_price': filled_price,
                        'order_id': order.id,
                        'child_orders': child_orders
                    }
                
                elif order_status.status in ["rejected", "canceled"]:
                    logger.error(f"❌ Order rejected for {symbol}: {order_status.status}")
                    return None
                
                await asyncio.sleep(0.5)
            
            logger.error(f"❌ Order for {symbol} did not fill within timeout")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to place bracket order for {symbol}: {e}")
            return None

    async def monitor_all_trades(self):
        """Monitor all active trades with websockets"""
        if not self.active_trades:
            logger.info("No trades to monitor")
            return
        
        symbols = list(self.active_trades.keys())
        logger.info(f"🔍 Starting monitoring for {len(symbols)} trades: {symbols}")
        
        # Start websocket tasks
        finnhub_task = asyncio.create_task(self._finnhub_websocket(symbols))
        alpaca_task = asyncio.create_task(self._alpaca_websocket(symbols))
        monitor_task = asyncio.create_task(self._trade_logic_monitor())
        
        try:
            await asyncio.gather(finnhub_task, alpaca_task, monitor_task, return_exceptions=True)
        finally:
            # Cleanup
            for task in [finnhub_task, alpaca_task, monitor_task]:
                if not task.done():
                    task.cancel()
            logger.info("✅ Trade monitoring completed")

    async def _finnhub_websocket(self, symbols: List[str]):
        """Finnhub websocket for price updates"""
        uri = f"wss://ws.finnhub.io?token={self.finnhub_key}"
        connection_attempts = 0
        max_attempts = 3

        while connection_attempts < max_attempts and self.active_trades:
            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                    # Subscribe to all symbols
                    for symbol in symbols:
                        if symbol in self.active_trades:  # Check if still active
                            subscribe_msg = json.dumps({"type": "subscribe", "symbol": symbol})
                            await ws.send(subscribe_msg)
                            await asyncio.sleep(0.1)  # Rate limiting
                    
                    logger.info(f"📡 Finnhub websocket monitoring: {symbols}")
                    connection_attempts = 0
                    
                    async for raw in ws:
                        if not self.active_trades:  # Exit if no active trades
                            break
                            
                        data = json.loads(raw)
                        if data.get("type") == "trade":
                            for trade in data.get("data", []):
                                symbol = trade.get("s")
                                price = trade.get("p")
                                if symbol and price and symbol in self.active_trades:
                                    async with self.price_lock:
                                        self.price_data[symbol] = price
                                        
            except Exception as e:
                connection_attempts += 1
                logger.warning(f"Finnhub websocket error (attempt {connection_attempts}): {e}")
                if connection_attempts < max_attempts:
                    await asyncio.sleep(5)

    async def _alpaca_websocket(self, symbols: List[str]):
        """Alpaca websocket for price updates"""
        uri = ALPACA_WS_URL
        connection_attempts = 0
        max_attempts = 3

        while connection_attempts < max_attempts and self.active_trades:
            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                    # Authenticate
                    auth_msg = json.dumps({
                        "action": "auth",
                        "key": self.apca_key,
                        "secret": self.apca_secret
                    })
                    await ws.send(auth_msg)
                    
                    # Wait for auth response
                    auth_resp = await ws.recv()
                    auth_data = json.loads(auth_resp)
                    
                    # Handle different auth response formats
                    if auth_data[0].get("msg") == "connected":
                        auth_resp2 = await ws.recv()
                        auth_data2 = json.loads(auth_resp2)
                        if auth_data2[0].get("msg") != "authenticated":
                            raise RuntimeError("Alpaca auth failed")
                    elif auth_data[0].get("msg") != "authenticated":
                        raise RuntimeError("Alpaca auth failed")
                    
                    # Subscribe to symbols in chunks
                    chunk_size = min(30, len(symbols))
                    for i in range(0, len(symbols), chunk_size):
                        chunk = symbols[i:i + chunk_size]
                        active_chunk = [s for s in chunk if s in self.active_trades]
                        
                        if active_chunk:
                            subscribe_msg = json.dumps({
                                "action": "subscribe",
                                "trades": active_chunk
                            })
                            await ws.send(subscribe_msg)
                    
                    logger.info(f"📡 Alpaca websocket monitoring: {symbols}")
                    connection_attempts = 0
                    
                    async for raw in ws:
                        if not self.active_trades:
                            break
                            
                        data = json.loads(raw)
                        for msg in data:
                            if msg.get("T") == "t":  # Trade
                                symbol = msg.get("S")
                                price = msg.get("p")
                                if symbol and price and symbol in self.active_trades:
                                    async with self.price_lock:
                                        self.price_data[symbol] = price
                                        
            except Exception as e:
                connection_attempts += 1
                logger.warning(f"Alpaca websocket error (attempt {connection_attempts}): {e}")
                if connection_attempts < max_attempts:
                    await asyncio.sleep(5)

    async def _trade_logic_monitor(self):
        """Monitor trade logic for all positions"""
        last_log_time = time.monotonic()
        log_interval = 30  # seconds
        
        while self.active_trades:
            now = datetime.now(TZ)
            
            # Check cutoff time
            if now >= self.cutoff_time:
                logger.info("⏰ Cutoff time reached - closing all positions")
                await self._close_all_positions("TIME_EXIT")
                break
            
            # Process each active trade
            trades_to_remove = []
            
            async with self.price_lock:
                current_trades = dict(self.active_trades)
                current_prices = dict(self.price_data)
            
            for symbol, trade_info in current_trades.items():
                if symbol not in current_prices:
                    continue
                    
                current_price = current_prices[symbol]
                entry_price = trade_info['entry_price']
                bin_data = trade_info['bin_data']
                
                # Calculate current P&L
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                # Check stop loss
                stop_loss_pct = bin_data['stop_loss']
                if pnl_pct <= stop_loss_pct:
                    logger.info(f"🛑 Stop loss triggered for {symbol}: {pnl_pct:.2f}% <= {stop_loss_pct:.2f}%")
                    success = await self._handle_stop_loss(symbol, trade_info)
                    if success:
                        trades_to_remove.append(symbol)
                    continue
                
                # Check target profit
                target_pct = bin_data['target_profit'] 
                if not trade_info['target_hit'] and pnl_pct >= target_pct:
                    logger.info(f"🎯 Target hit for {symbol}: {pnl_pct:.2f}% >= {target_pct:.2f}%")
                    await self._handle_target_hit(symbol, trade_info, current_price)
                    async with self.price_lock:
                        self.active_trades[symbol]['target_hit'] = True
                        self.active_trades[symbol]['peak_price'] = current_price
                
                # Handle trailing stop logic
                elif trade_info['target_hit']:
                    await self._handle_trailing_stop(symbol, trade_info, current_price)
            
            # Remove completed trades
            for symbol in trades_to_remove:
                async with self.price_lock:
                    self.active_trades.pop(symbol, None)
                    self.price_data.pop(symbol, None)
            
            # Periodic logging
            current_time = time.monotonic()
            if current_time - last_log_time > log_interval:
                await self._log_trade_status()
                last_log_time = current_time
            
            await asyncio.sleep(0.1)  # 100ms monitoring loop

    async def _handle_stop_loss(self, symbol: str, trade_info: Dict) -> bool:
        """Handle stop loss trigger with double-check mechanism and retry logic"""
        try:
            # First check if bracket order stop loss was executed
            await asyncio.sleep(1)
            
            try:
                position = await asyncio.to_thread(self.api.get_position, symbol)
                current_qty = int(position.qty)
                
                if current_qty > 0:
                    logger.warning(f"⚠️ Bracket stop loss failed for {symbol} - executing manual sell of {current_qty} shares")
                    
                    # Cancel remaining child orders
                    for child_order in trade_info['child_orders']:
                        try:
                            await asyncio.to_thread(self.api.cancel_order, child_order['id'])
                        except Exception as e:
                            logger.debug(f"Could not cancel child order {child_order['id']}: {e}")
                    
                    # Execute market sell with retry logic for partial fills
                    await self._execute_sell_with_retry(symbol, "STOP_LOSS")
                else:
                    logger.info(f"✅ Bracket stop loss worked for {symbol}")
                    
            except Exception as e:
                if "position does not exist" in str(e).lower():
                    logger.info(f"✅ Position already closed for {symbol}")
                else:
                    logger.error(f"❌ Error checking position for {symbol}: {e}")
                    
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to handle stop loss for {symbol}: {e}")
            return False

    async def _execute_sell_with_retry(self, symbol: str, reason: str, max_retries: int = 100):
        """Execute sell with comprehensive retry logic for partial fills"""
        entry_price = self.active_trades[symbol]['entry_price']
        current_price = self.price_data.get(symbol, entry_price)
        pnl = (current_price - entry_price) / entry_price * 100
        
        logger.info(f"{reason}: {symbol} @ ${current_price:.2f} (P&L: {pnl:+.2f}%)")
        
        attempt = 0
        while attempt <= max_retries:
            try:
                # 🛡️ Always refresh current position before selling
                try:
                    current_position = await asyncio.to_thread(self.api.get_position, symbol)
                    remaining_qty = int(current_position.qty)
                    if remaining_qty <= 0:
                        logger.info(f"✅ No shares left to sell on attempt #{attempt + 1}")
                        break
                    logger.debug(f"📊 Position check before attempt #{attempt + 1}: {remaining_qty} shares")
                except Exception as e:
                    if "position does not exist" in str(e).lower():
                        logger.info(f"✅ No position found on attempt #{attempt + 1}: position already closed")
                        break
                    else:
                        logger.warning(f"⚠️ Could not verify position on attempt #{attempt + 1}: {e}")
                        # Continue with last known quantity as fallback
                        remaining_qty = self.active_trades[symbol]['quantity']

                # 🛑 Hard stop check
                if remaining_qty <= 0:
                    logger.info(f"🛑 Skipping sell attempt — no shares left to sell (qty={remaining_qty})")
                    break

                # Submit the sell order for actual remaining shares
                sell_order = await asyncio.to_thread(
                    self.api.submit_order,
                    symbol=symbol,
                    qty=remaining_qty,
                    side="sell",
                    type="market",
                    time_in_force="day"
                )
                logger.info(f"📤 Sell order attempt #{attempt + 1}: Submitted {remaining_qty} shares")

                # Poll for fill with partial fill tracking
                max_polls = 20
                total_sold = 0
                
                for poll in range(max_polls):
                    await asyncio.sleep(0.5)
                    fetched_order = await asyncio.to_thread(self.api.get_order, sell_order.id)
                    filled_qty = int(fetched_order.filled_qty or 0)
                    
                    if filled_qty > total_sold:
                        new_fill = filled_qty - total_sold
                        total_sold = filled_qty
                        avg_price = float(fetched_order.filled_avg_price or current_price)
                        logger.info(f"✅ Partial sell fill: {new_fill} shares @ ${avg_price:.2f} (total sold: {total_sold}/{remaining_qty})")
                    
                    if fetched_order.status == "filled":
                        logger.info(f"✅ Sell order completely filled: {total_sold} shares")
                        break
                    elif fetched_order.status in ["rejected", "canceled", "expired"]:
                        if total_sold > 0:
                            logger.warning(f"⚠️ Sell order {fetched_order.status} after selling {total_sold}/{remaining_qty} shares")
                        else:
                            logger.warning(f"⚠️ Sell order {fetched_order.status} - no shares sold")
                        break

                # Update our tracking based on what actually sold
                if total_sold > 0:
                    async with self.price_lock:
                        self.active_trades[symbol]['quantity'] -= total_sold
                        logger.info(f"📊 Updated position: {self.active_trades[symbol]['quantity']} shares remaining")
                    
                    # If we sold everything, we're done
                    if self.active_trades[symbol]['quantity'] <= 0:
                        logger.info(f"✅ POSITION FULLY CLOSED for {symbol} on attempt #{attempt + 1}")
                        break

            except Exception as e:
                logger.error(f"❌ Sell order failed on attempt #{attempt + 1}: {e}")

                if "position does not exist" in str(e).lower() or "insufficient shares" in str(e).lower():
                    logger.info(f"✅ Position appears to be already closed: {e}")
                    break

            attempt += 1

        # Final position verification
        try:
            final_position = await asyncio.to_thread(self.api.get_position, symbol)
            final_qty = int(final_position.qty)
            if final_qty > 0:
                logger.critical(f"🚨 UNABLE TO FULLY EXIT POSITION: {final_qty} shares remaining after {max_retries} retries")
            else:
                logger.info(f"✅ Position verification: {symbol} fully closed")
        except Exception as e:
            if "position does not exist" in str(e).lower():
                logger.info(f"✅ Final verification: {symbol} position successfully closed")
            else:
                logger.warning(f"⚠️ Could not verify final position for {symbol}: {e}")

    async def _handle_target_hit(self, symbol: str, trade_info: Dict, current_price: float):
        """Handle target hit - cancel bracket and prepare for trailing stop"""
        try:
            # Cancel bracket child orders
            for child_order in trade_info['child_orders']:
                try:
                    await asyncio.to_thread(self.api.cancel_order, child_order['id'])
                    logger.info(f"📤 Cancelled bracket child order for {symbol}: {child_order['id']}")
                except Exception as e:
                    logger.debug(f"Could not cancel child order {child_order['id']}: {e}")
            
            # Could optionally place Alpaca trailing stop here, but we'll use websocket trailing
            logger.info(f"🎯 {symbol} target hit - now using websocket trailing stop at {TRAIL_PERCENT*100:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Failed to handle target hit for {symbol}: {e}")

    async def _handle_trailing_stop(self, symbol: str, trade_info: Dict, current_price: float):
        """Handle trailing stop logic with robust position verification"""
        peak_price = trade_info['peak_price']
        
        # Update peak if price went higher
        if current_price > peak_price:
            async with self.price_lock:
                self.active_trades[symbol]['peak_price'] = current_price
            return
        
        # Check for trailing stop trigger
        decline_pct = (peak_price - current_price) / peak_price
        if decline_pct >= TRAIL_PERCENT:
            pnl_pct = (current_price - trade_info['entry_price']) / trade_info['entry_price'] * 100
            logger.info(f"📉 Trailing stop triggered for {symbol}: {decline_pct*100:.2f}% decline from peak ${peak_price:.2f} (P&L: {pnl_pct:+.2f}%)")
            
            # Execute trailing stop sell with retry logic
            await self._execute_sell_with_retry(symbol, "TRAILING_STOP")
            
            # Remove from active trades after sell attempt
            async with self.price_lock:
                self.active_trades.pop(symbol, None)
                self.price_data.pop(symbol, None)

    async def _close_all_positions(self, reason: str):
        """Close all remaining positions with robust retry logic"""
        logger.info(f"🔄 Closing all positions due to: {reason}")
        
        symbols_to_close = list(self.active_trades.keys())
        close_tasks = []
        
        for symbol in symbols_to_close:
            try:
                # Check if we actually have a position
                position = await asyncio.to_thread(self.api.get_position, symbol)
                current_qty = int(position.qty)
                
                if current_qty > 0:
                    # Cancel any existing orders first
                    trade_info = self.active_trades[symbol]
                    for child_order in trade_info.get('child_orders', []):
                        try:
                            await asyncio.to_thread(self.api.cancel_order, child_order['id'])
                            logger.debug(f"Cancelled order {child_order['id']} for {symbol}")
                        except Exception as e:
                            logger.debug(f"Could not cancel order {child_order['id']}: {e}")
                    
                    # Create sell task for this symbol
                    task = asyncio.create_task(self._execute_sell_with_retry(symbol, reason, max_retries=50))
                    close_tasks.append(task)
                    
            except Exception as e:
                if "position does not exist" in str(e).lower():
                    logger.debug(f"No position for {symbol}")
                else:
                    logger.error(f"❌ Error checking position for {symbol}: {e}")
        
        # Execute all sell orders concurrently
        if close_tasks:
            logger.info(f"📤 Executing {len(close_tasks)} concurrent sell orders...")
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # Clear all active trades
        async with self.price_lock:
            self.active_trades.clear()
            self.price_data.clear()
            
        logger.info(f"✅ All position closure attempts completed for {reason}")

    async def _log_trade_status(self):
        """Log current status of all trades"""
        if not self.active_trades:
            return
            
        logger.info(f"📊 Trade Status Update ({len(self.active_trades)} active trades):")
        
        async with self.price_lock:
            for symbol, trade_info in self.active_trades.items():
                current_price = self.price_data.get(symbol, trade_info['entry_price'])
                entry_price = trade_info['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                status = "🎯 TRAILING" if trade_info['target_hit'] else "📈 MONITORING"
                peak_info = f" (peak: ${trade_info['peak_price']:.2f})" if trade_info['target_hit'] else ""
                
                logger.info(f"   {status} {symbol}: ${current_price:.2f} | P&L: {pnl_pct:+.2f}%{peak_info}")


# ─── NEW MAIN TRADING FUNCTIONS ──────────────────────────────────────────────

async def find_opening_prices_and_trade(
    symbols: List[str],
    prev_close: Dict[str, float],
    bins_data: Dict,
    api: AlpacaREST,
    finnhub_key: str,
    apca_key: str,
    apca_secret: str,
    position_size: float
):
    """Find opening prices and immediately place trades for viable candidates"""
    
    logger.info("Waiting for market to open...")
    sleep_until_market_open(api)
    time.sleep(1)
    logger.info(f"Market open - monitoring {len(symbols)} symbols for trading opportunities")
    
    # Calculate cutoff time (9:31:30 America/New_York for tighter window)
    now = datetime.now(TZ)
    cutoff_time = now.replace(hour=9, minute=31, second=30, microsecond=0)
    
    if now > cutoff_time:
        logger.error(f"❌ Already past cutoff time of 9:31:30 - exiting")
        raise RuntimeError("Already past cutoff time")
    
    # Shared state for price monitoring
    price_results = {}
    pending_symbols = set(symbols)
    results_lock = asyncio.Lock()
    cutoff_reached = asyncio.Event()
    
    # Initialize trade manager
    trade_manager = TradeManager(api, finnhub_key, apca_key, apca_secret)
    
    # Track trades placed
    trades_placed = 0
    bins = bins_data['bins']
    
    async def cutoff_monitor():
        """Monitor cutoff time"""
        while True:
            if datetime.now(TZ) >= cutoff_time:
                logger.info(f"⏰ Cutoff time {cutoff_time.time()} reached")
                cutoff_reached.set()
                break
            await asyncio.sleep(0.1)
    
    async def process_trade_opportunity(symbol: str, price: float, pct_increase: float):
        """Process a potential trade opportunity"""
        nonlocal trades_placed
        
        # Find appropriate bin
        bin_data = find_bin_for_percentage(pct_increase, bins)
        if not bin_data:
            logger.debug(f"❌ No bin found for {symbol} at {pct_increase:.2f}%")
            return
        
        logger.info(f"🔥 TRADE OPPORTUNITY: {symbol} @ ${price:.2f} ({pct_increase:.2f}%) - matches bin range")
        
        # Place bracket order immediately
        order_result = await trade_manager.place_bracket_order(symbol, price, position_size, bin_data)
        
        if order_result:
            await trade_manager.add_trade(symbol, order_result, bin_data)
            trades_placed += 1
            logger.info(f"✅ Trade #{trades_placed} placed: {symbol}")
        else:
            logger.error(f"❌ Failed to place trade for {symbol}")
    
    async def finnhub_price_monitor():
        """Monitor prices via Finnhub WebSocket"""
        uri = f"wss://ws.finnhub.io?token={finnhub_key}"
        attempt = 0
        max_attempts = 3
        
        while attempt < max_attempts and not cutoff_reached.is_set():
            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                    # Subscribe to all symbols
                    for symbol in symbols:
                        subscribe_msg = json.dumps({"type": "subscribe", "symbol": symbol})
                        await ws.send(subscribe_msg)
                        await asyncio.sleep(0.05)  # Rate limiting
                    
                    logger.info("📡 Finnhub price monitoring active")
                    attempt = 0
                    
                    while not cutoff_reached.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            data = json.loads(raw)
                            
                            if data.get("type") == "trade":
                                for trade in data.get("data", []):
                                    symbol = trade.get("s")
                                    price = trade.get("p")
                                    
                                    if symbol and price and symbol in pending_symbols:
                                        if symbol in prev_close:
                                            pct_increase = (price - prev_close[symbol]) / prev_close[symbol] * 100
                                            
                                            async with results_lock:
                                                if symbol in pending_symbols:
                                                    pending_symbols.remove(symbol)
                                                    price_results[symbol] = {
                                                        'price': price,
                                                        'pct_increase': pct_increase,
                                                        'source': 'finnhub'
                                                    }
                                            
                                            # Process trade opportunity immediately
                                            await process_trade_opportunity(symbol, price, pct_increase)
                                        
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.debug(f"Finnhub WS processing error: {e}")
                            
            except Exception as e:
                attempt += 1
                logger.warning(f"Finnhub WS error (attempt {attempt}): {e}")
                if attempt < max_attempts and not cutoff_reached.is_set():
                    await asyncio.sleep(2)
    
    async def alpaca_price_monitor():
        """Monitor prices via Alpaca WebSocket"""
        uri = ALPACA_WS_URL
        attempt = 0
        max_attempts = 3
        
        while attempt < max_attempts and not cutoff_reached.is_set():
            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                    # Authenticate
                    auth_msg = json.dumps({
                        "action": "auth",
                        "key": apca_key,
                        "secret": apca_secret
                    })
                    await ws.send(auth_msg)
                    
                    # Handle auth response
                    auth_resp = json.loads(await ws.recv())
                    if auth_resp[0].get("msg") == "connected":
                        auth_resp2 = json.loads(await ws.recv())
                        if auth_resp2[0].get("msg") != "authenticated":
                            raise RuntimeError("Alpaca auth failed")
                    elif auth_resp[0].get("msg") != "authenticated":
                        raise RuntimeError("Alpaca auth failed")
                    
                    # Subscribe in chunks
                    chunk_size = 30
                    for i in range(0, len(symbols), chunk_size):
                        chunk = symbols[i:i + chunk_size]
                        subscribe_msg = json.dumps({
                            "action": "subscribe",
                            "trades": chunk
                        })
                        await ws.send(subscribe_msg)
                    
                    logger.info("📡 Alpaca price monitoring active")
                    attempt = 0
                    
                    while not cutoff_reached.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            data = json.loads(raw)
                            
                            for msg in data:
                                if msg.get("T") == "t":  # Trade
                                    symbol = msg.get("S")
                                    price = msg.get("p")
                                    
                                    if symbol and price and symbol in pending_symbols:
                                        if symbol in prev_close:
                                            pct_increase = (price - prev_close[symbol]) / prev_close[symbol] * 100
                                            
                                            async with results_lock:
                                                if symbol in pending_symbols:
                                                    pending_symbols.remove(symbol)
                                                    price_results[symbol] = {
                                                        'price': price,
                                                        'pct_increase': pct_increase,
                                                        'source': 'alpaca'
                                                    }
                                            
                                            # Process trade opportunity immediately
                                            await process_trade_opportunity(symbol, price, pct_increase)
                                            
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.debug(f"Alpaca WS processing error: {e}")
                            
            except Exception as e:
                attempt += 1
                logger.warning(f"Alpaca WS error (attempt {attempt}): {e}")
                if attempt < max_attempts and not cutoff_reached.is_set():
                    await asyncio.sleep(2)
    
    # Start all monitoring tasks
    cutoff_task = asyncio.create_task(cutoff_monitor())
    finnhub_task = asyncio.create_task(finnhub_price_monitor())
    alpaca_task = asyncio.create_task(alpaca_price_monitor())
    
    try:
        # Wait for cutoff time
        await cutoff_reached.wait()
        logger.info(f"🛑 Price discovery phase complete - {trades_placed} trades placed")
        
    finally:
        # Cancel monitoring tasks
        for task in [cutoff_task, finnhub_task, alpaca_task]:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(cutoff_task, finnhub_task, alpaca_task, return_exceptions=True)
    
    # Now monitor all placed trades
    if trade_manager.active_trades:
        logger.info(f"🔍 Starting trade monitoring for {len(trade_manager.active_trades)} positions")
        await trade_manager.monitor_all_trades()
    else:
        logger.info("No trades were placed - session complete")


def main():
    logger.info("🚀 Starting Enhanced JSON-Driven Earnings Trading Bot")

    if not all([FINNHUB_API_KEY, APCA_KEY, APCA_SECRET]):
        logger.error("❌ Missing required API keys - check environment variables")
        logger.error("Required: FINNHUB_API_KEY, APCA_API_KEY_ID, APCA_API_SECRET_KEY")
        sys.exit(1)

    try:
        fh = finnhub.Client(api_key=FINNHUB_API_KEY)
        alp = AlpacaREST(APCA_KEY, APCA_SECRET, ALPACA_BASE_URL)
        logger.info("✅ Finnhub and Alpaca clients initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize API clients: {e}")
        sys.exit(1)

    # Load earnings bins configuration
    try:
        bins_data = load_earnings_bins("earnings_bins.json")
        min_candidates = bins_data['metadata']['minimum_candidates_analysis']['recommended_minimum']
        logger.info(f"✅ Loaded trading configuration - minimum candidates: {min_candidates}")
    except Exception as e:
        logger.error(f"❌ Failed to load earnings bins configuration: {e}")
        sys.exit(1)

    today = datetime.now(TZ).date()
    prev_day = get_prev_business_day(today)
    start, end = prev_day.isoformat(), today.isoformat()
    logger.info(f"📅 Trading session: {start} → {end}")

    # Fetch and filter earnings data (using existing functions - DO NOT CHANGE)
    entries = fetch_earnings_calendar(fh, start, end)
    if not entries:
        logger.error("❌ No earnings entries found - exiting")
        sys.exit(1)

    candidates = filter_candidates(entries, fh, start, end)
    if not candidates:
        logger.error("❌ No viable candidates found - exiting")
        sys.exit(1)

    candidates = group_candidates_by_surprise(candidates)

    # Check minimum candidates requirement
    if len(candidates) < min_candidates:
        logger.error(f"❌ Only {len(candidates)} candidates found, minimum required: {min_candidates}")
        logger.error("Not performing any trades due to insufficient candidates")
        sys.exit(0)  # Exit successfully but don't trade

    logger.info(f"✅ {len(candidates)} candidates meet minimum requirement of {min_candidates}")

    # Load previous closes (using existing function - DO NOT CHANGE)
    prev_close = preload_prev_closes(alp, candidates, start)
    if not prev_close:
        logger.error("❌ No previous closes loaded - exiting")
        sys.exit(1)

    # Calculate position sizing
    try:
        account = alp.get_account()
        total_cash = float(account.cash)
        position_size = calculate_position_size(total_cash, min_candidates)
        
        logger.info(f"💰 Account cash: ${total_cash:.2f}")
        logger.info(f"📊 Position size per trade: ${position_size:.2f}")
        logger.info(f"🎯 Using {TOTAL_ACCOUNT_PCT}% of account with {TARGET_PROFIT_FACTOR}x target multiplier")
        
        if position_size < 100:  # Minimum viable position size
            logger.error(f"❌ Position size too small: ${position_size:.2f} < $100")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Failed to get account information: {e}")
        sys.exit(1)

    # Start trading session
    symbols = [c["symbol"] for c in candidates]
    logger.info(f"🎯 Starting trading session with {len(symbols)} candidates")
    
    try:
        asyncio.run(find_opening_prices_and_trade(
            symbols=symbols,
            prev_close=prev_close,
            bins_data=bins_data,
            api=alp,
            finnhub_key=FINNHUB_API_KEY,
            apca_key=APCA_KEY,
            apca_secret=APCA_SECRET,
            position_size=position_size
        ))
    except RuntimeError as e:
        logger.error(f"❌ Trading session failed: {e}")
        sys.exit(1)

    logger.info("🏁 Enhanced Trading Session Completed")


if __name__ == "__main__":
    main()