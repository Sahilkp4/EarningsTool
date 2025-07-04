import os
import sys
import time
import logging
import asyncio
import json
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
#fix these comments when running on ubuntu aaabbcc
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

STOP_LOSS_FACTOR   = 0.96 # percentage of total to stop loss at
TARGET_FACTOR      = 1.0267 # percentage of total to target
MONITOR_END_HOUR   = 15 #end hour
MONITOR_END_MINUTE = 33 #end minute

SURPRISE_THRESHOLD = 1 # % min earnings suprise
MAX_SURPRISE       = 600 # man suprise
MC_THRESHOLD       = 1_000_000 # market cap in millions of dollars
MIN_PCT_INCREASE   = 3.0  # Minimum threshold for % increase at open
# ─── Logging Setup ────────────────────────────────────────────────────────────
def configure_logging() -> logging.Logger:
    logger = logging.getLogger("earnings_trader")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s",
            "%Y-%m-%d %H:%M:%S %Z"
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "session_output.txt"),
            mode="a", encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

logger = configure_logging()

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
            mc = float(profile.get("marketCapitalization", 0)) / 1e6
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
    logger.info("Sorted candidates by surprise: " +
                 ", ".join(f"{c['symbol']}({c['surprise']:.1f}%)"
                           for c in sorted_c))
    return sorted_c

#1 Opening Function
async def get_opening_prices_with_window(
    symbols: List[str],
    finnhub_key: str,
    api: AlpacaREST,
    prev_close: Dict[str, float],
    window_seconds: int = 15,
    max_rest_symbols: int = 50
) -> Optional[Dict[str, float]]:
    """
    Find opening prices with a two-phase approach:
    1. Race to find the FIRST candidate that meets minimum criteria
    2. Once found, wait 15 seconds to see if better candidates emerge
    """
    logger.info("Waiting for market to open...")
    sleep_until_market_open(api)
    time.sleep(1)
    logger.info(f"Market is open. Racing to find first viable candidate from {len(symbols)} symbols")

    # Calculate cutoff time (9:31:01 America/New_York)
    now = datetime.now(TZ)
    cutoff_time = now.replace(hour=9, minute=31, second=1, microsecond=0)
    
    # If we're already past cutoff, exit immediately
    if now > cutoff_time:
        logger.error(f"❌ Already past cutoff time of 9:31:01 ({cutoff_time}) - exiting")
        raise RuntimeError("Already past cutoff time of 9:31:01")

    # For large candidate lists, prioritize WebSocket for all symbols
    # but limit REST API to most promising subset to respect rate limits
    rest_symbols = symbols[:max_rest_symbols] if len(symbols) > max_rest_symbols else symbols

    if len(symbols) > max_rest_symbols:
        logger.info(f"Large candidate list detected. WebSocket: all {len(symbols)}, REST API: top {len(rest_symbols)}")

    # Shared state
    results = {}
    pending_symbols = set(symbols)  # WebSocket tracks all symbols
    pending_rest_symbols = set(rest_symbols)  # REST API tracks subset
    results_lock = asyncio.Lock()
    first_viable_found = asyncio.Event()
    cutoff_reached = asyncio.Event()
    first_viable_time = None

    # Start cutoff monitor task
    cutoff_monitor_task = asyncio.create_task(
        monitor_cutoff_time(cutoff_time, cutoff_reached)
    )

    # Start both data fetchers
    websocket_task = asyncio.create_task(
        websocket_price_fetcher(
            symbols, finnhub_key, prev_close, results, pending_symbols,
            results_lock, first_viable_found, cutoff_reached
        )
    )

    rest_api_task = asyncio.create_task(
        rest_api_price_fetcher(
            rest_symbols, finnhub_key, prev_close, results, pending_rest_symbols,
            results_lock, first_viable_found, cutoff_reached, api=api, max_calls_per_batch=45
        )
    )

    try:
        # Phase 1: Wait for first viable candidate OR cutoff time
        logger.info(f"Phase 1: Racing to find first candidate with >{MIN_PCT_INCREASE}% increase...")
        
        # Wait for either first viable candidate or cutoff time
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(first_viable_found.wait()),
                asyncio.create_task(cutoff_reached.wait())
            ],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
        
        # Check if cutoff was reached first
        if cutoff_reached.is_set():
            logger.error(f"❌ Cutoff time of 9:31:01 reached without finding viable candidates - exiting")
            raise RuntimeError("No viable candidate found before 9:31:01")
        
        # First viable candidate was found
        first_viable_time = time.monotonic()
        
        async with results_lock:
            viable_candidates = [
                (symbol, data) for symbol, data in results.items()
                if data["pct_increase"] >= MIN_PCT_INCREASE
            ]

        logger.info(f"🎯 First viable candidate found! Starting {window_seconds}s window for better options...")

        # Phase 2: Wait additional time for potentially better candidates
        window_start = time.monotonic()
        while (time.monotonic() - window_start) < window_seconds:
            # Check if cutoff time reached during window
            if cutoff_reached.is_set():
                logger.warning("Cutoff time reached during collection window - proceeding with current results")
                break
                
            await asyncio.sleep(0.1)

            # Check if we found significantly better candidates
            async with results_lock:
                current_viable = [
                    (symbol, data) for symbol, data in results.items()
                    if data["pct_increase"] >= MIN_PCT_INCREASE
                ]

                if len(current_viable) > len(viable_candidates):
                    logger.info(f"Found {len(current_viable)} viable candidates so far...")
                    viable_candidates = current_viable

        logger.info(f"Phase 2 complete. Final evaluation of {len(viable_candidates)} candidates.")

    finally:
        # Cancel all tasks
        cutoff_monitor_task.cancel()
        websocket_task.cancel()
        rest_api_task.cancel()
        await asyncio.gather(
            cutoff_monitor_task, websocket_task, rest_api_task, 
            return_exceptions=True
        )

    # Select best candidate from viable options
    best_candidate = select_best_candidate(results, MIN_PCT_INCREASE)

    # Return the best candidate for main() to handle buying
    return best_candidate


async def monitor_cutoff_time(cutoff_time: datetime, cutoff_reached: asyncio.Event):
    """Monitor for cutoff time and signal when reached"""
    while True:
        now = datetime.now(TZ)
        if now >= cutoff_time:
            logger.info(f"Cutoff time {cutoff_time} reached")
            cutoff_reached.set()
            break
        
        # Check every 100ms for precision
        await asyncio.sleep(0.1)


# 2. WEBSOCKET FUNCTION WITH PING/PONG AND CUTOFF MONITORING
async def websocket_price_fetcher(
    symbols: List[str],
    finnhub_key: str,
    prev_close: Dict[str, float],
    results: Dict,
    pending_symbols: Set[str],
    results_lock: asyncio.Lock,
    first_viable_found: asyncio.Event,
    cutoff_reached: asyncio.Event,
):
    """WebSocket price fetching task with ping/pong monitoring and cutoff checking"""
    uri = f"wss://ws.finnhub.io?token={finnhub_key}"
    connection_attempts = 0
    max_attempts = 5

    while connection_attempts < max_attempts:
        try:
            # Check if cutoff reached before attempting connection
            if cutoff_reached.is_set():
                logger.debug("WebSocket: Cutoff time reached, exiting")
                break
                
            logger.debug(f"WebSocket connection attempt {connection_attempts + 1}")

            # Add ping interval for connection monitoring
            async with websockets.connect(
                uri,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10,   # Wait 10 seconds for pong
                close_timeout=5
            ) as ws:
                logger.info("WebSocket connected, subscribing to symbols")

                # Subscribe to all symbols
                for symbol in symbols:
                    # Check cutoff before each subscription
                    if cutoff_reached.is_set():
                        logger.debug("WebSocket: Cutoff reached during subscription, exiting")
                        return
                        
                    subscribe_msg = json.dumps({"type": "subscribe", "symbol": symbol})
                    await ws.send(subscribe_msg)
                    await asyncio.sleep(0.05)

                logger.info("WebSocket ready, listening for trades...")

                # Track last message time for additional monitoring
                last_message_time = time.monotonic()
                heartbeat_interval = 30  # Log heartbeat every 30 seconds

                while True:
                    # Check cutoff time at start of each loop
                    if cutoff_reached.is_set():
                        logger.debug("WebSocket: Cutoff time reached, closing connection")
                        break
                        
                    try:
                        # Wait for message with timeout
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)  # Reduced timeout for faster cutoff response
                        last_message_time = time.monotonic()

                        data = json.loads(raw)

                        # Handle different message types
                        if data.get("type") == "trade":
                            found_viable = await process_websocket_trades(
                                data, prev_close, results, pending_symbols,
                                results_lock
                            )

                            # Signal if we found first viable candidate
                            if found_viable and not first_viable_found.is_set():
                                first_viable_found.set()

                        elif data.get("type") == "ping":
                            # Respond to server ping
                            pong_msg = json.dumps({"type": "pong"})
                            await ws.send(pong_msg)
                            logger.debug("Responded to WebSocket ping")

                        elif data.get("type") in ["error", "status"]:
                            logger.debug(f"WebSocket {data.get('type')}: {data}")

                    except asyncio.TimeoutError:
                        # Check if we should send a heartbeat log
                        current_time = time.monotonic()
                        if current_time - last_message_time > heartbeat_interval:
                            logger.debug("WebSocket waiting for data (connection alive)")
                            last_message_time = current_time
                        continue

                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"WebSocket connection closed: {e}")
                        break

        except asyncio.CancelledError:
            logger.debug("WebSocket task cancelled")
            break
        except Exception as e:
            connection_attempts += 1
            logger.error(f"WebSocket error (attempt {connection_attempts}): {e}")
            if connection_attempts < max_attempts and not cutoff_reached.is_set():
                await asyncio.sleep(2)
            else:
                logger.error("Max WebSocket connection attempts reached or cutoff time reached")
                break


async def rest_api_price_fetcher(
    symbols: List[str],
    finnhub_key: str,
    prev_close: Dict[str, float],
    results: Dict,
    pending_symbols: Set[str],
    results_lock: asyncio.Lock,
    first_viable_found: asyncio.Event,
    cutoff_reached: asyncio.Event,
    api: AlpacaREST,
    max_calls_per_batch: int = 45
):
    """REST API price fetching task with cutoff monitoring"""
    client = finnhub.Client(api_key=finnhub_key)
    
    while len(pending_symbols) > 0:
        # Check cutoff time before each batch
        if cutoff_reached.is_set():
            logger.debug("REST API: Cutoff time reached, exiting")
            break
            
        try:
            # Process symbols in batches to respect rate limits
            current_batch = list(pending_symbols)[:max_calls_per_batch]
            logger.debug(f"REST API processing batch of {len(current_batch)} symbols")
            
            for symbol in current_batch:
                # Check cutoff before each API call
                if cutoff_reached.is_set():
                    logger.debug("REST API: Cutoff reached during batch processing, exiting")
                    return
                    
                try:
                    # Get current price
                    quote = client.quote(symbol)
                    current_price = quote.get('c')
                    
                    if current_price and current_price > 0:
                        # Use your existing helper function
                        found_viable = await process_rest_price(
                            symbol, current_price, prev_close, results, 
                            pending_symbols, results_lock
                        )
                        
                        # Signal if we found first viable candidate
                        if found_viable and not first_viable_found.is_set():
                            first_viable_found.set()
                    
                    # Rate limiting delay
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.debug(f"REST API error for {symbol}: {e}")
                    continue
            
            # Wait between batches if there are more symbols and cutoff not reached
            if len(pending_symbols) > 0 and not cutoff_reached.is_set():
                await asyncio.sleep(2)
                
        except asyncio.CancelledError:
            logger.debug("REST API task cancelled")
            break
        except Exception as e:
            logger.error(f"REST API batch processing error: {e}")
            await asyncio.sleep(1)


#  HELPER FUNCTIONS
async def process_websocket_trades(
    data: Dict,
    prev_close: Dict[str, float],
    results: Dict,
    pending_symbols: Set[str],
    results_lock: asyncio.Lock,
) -> bool:
    """Process trade data from WebSocket, return True if viable candidate found"""
    found_viable = False

    for trade in data.get("data", []):
        symbol = trade.get("s")
        price = trade.get("p")
        timestamp = trade.get("t", 0)

        if symbol and price and symbol in pending_symbols:
            async with results_lock:
                if symbol in pending_symbols:
                    pending_symbols.remove(symbol)

                    pct_increase = calculate_percentage_increase(symbol, price, prev_close)
                    results[symbol] = {
                        "price": price,
                        "pct_increase": pct_increase,
                        "source": "websocket",
                        "timestamp": timestamp
                    }

                    if pct_increase >= MIN_PCT_INCREASE:
                        trade_time = datetime.fromtimestamp(timestamp / 1000) if timestamp else "now"
                        logger.info(f"🔥 VIABLE WebSocket: {symbol} @ ${price:.2f} ({pct_increase:.2f}%) at {trade_time}")
                        found_viable = True
                    else:
                        logger.info(f"📊 WebSocket: {symbol} @ ${price:.2f} ({pct_increase:.2f}%) - below threshold")

    return found_viable


async def process_rest_price(
    symbol: str,
    price: float,
    prev_close: Dict[str, float],
    results: Dict,
    pending_symbols: Set[str],
    results_lock: asyncio.Lock,
) -> bool:
    """Process price data from REST API, return True if viable candidate found"""
    async with results_lock:
        if symbol in pending_symbols:
            pending_symbols.remove(symbol)

            pct_increase = calculate_percentage_increase(symbol, price, prev_close)
            results[symbol] = {
                "price": price,
                "pct_increase": pct_increase,
                "source": "rest_api",
                "timestamp": int(time.time() * 1000)
            }

            if pct_increase >= MIN_PCT_INCREASE:
                logger.info(f"🚀 VIABLE REST API: {symbol} @ ${price:.2f} ({pct_increase:.2f}%)")
                return True
            else:
                logger.info(f"📊 REST API: {symbol} @ ${price:.2f} ({pct_increase:.2f}%) - below threshold")
                return False

    return False


def calculate_percentage_increase(symbol: str, price: float, prev_close: Dict[str, float]) -> float:
    """Calculate percentage increase from previous close"""
    if symbol in prev_close:
        pct_increase = (price - prev_close[symbol]) / prev_close[symbol] * 100
        return pct_increase
    return 0.0


# 5. SELECT BEST CANDIDATE FUNCTION
def select_best_candidate(results: Dict, min_pct_increase: float) -> Optional[Dict[str, float]]:
    """Select the best candidate from viable results"""
    viable_candidates = [
        (symbol, data) for symbol, data in results.items()
        if data["pct_increase"] >= min_pct_increase
    ]

    if not viable_candidates:
        logger.warning("No viable candidates found meeting minimum criteria")
        return None

    # Sort by percentage increase (descending)
    viable_candidates.sort(key=lambda x: x[1]["pct_increase"], reverse=True)

    best_symbol, best_data = viable_candidates[0]

    logger.info(f"🏆 SELECTED: {best_symbol} @ {best_data['pct_increase']:.2f}% "
               f"(${best_data['price']:.2f}, source: {best_data['source']})")

    if len(viable_candidates) > 1:
        other_candidates = [(s, f"{d['pct_increase']:.2f}%") for s, d in viable_candidates[1:]]
        logger.info(f"Other viable candidates: {other_candidates}")

    return {
        "symbol": best_symbol,
        "price": best_data["price"],
        "pct_increase": best_data["pct_increase"],
        "source": best_data["source"]
    }


# 6. EVALUATE AND BUY FUNCTION
def evaluate_candidates_and_buy(
    api: AlpacaREST,
    best_candidate: Optional[dict]
) -> Optional[dict]:
    """
    Execute buy order for the best candidate that met the criteria (if any)
    Returns order details if successful, None if failed
    """
    if not best_candidate:
        logger.info("No candidates met the 3% increase threshold")
        return None

    sym = best_candidate["symbol"]
    price = best_candidate["price"]
    pct_increase = best_candidate["pct_increase"]

    # Double-check market is open
    """
    #FAKE CLOCK !!!!
    class FakeClock:
        is_open = True
        timestamp = datetime(2025, 6, 25, 9, 30, 0, tzinfo=ZoneInfo("America/New_York"))
    clock = FakeClock()
    """
    # REAL CLOCK !!!
    clock = api.get_clock()
    if not clock.is_open:
        logger.info("Market not open yet — waiting before placing buy order...")
        sleep_until_market_open(api)

    logger.info(f"📈 Best candidate: {sym} with {pct_increase:.2f}% increase - executing buy!")

    # Place the buy order and return the order details
    order_result = place_buy_order(api, sym, price)

    if order_result:
        logger.info(f"✅ Successfully executed buy order for {sym}")
        return order_result
    else:
        logger.error(f"❌ Failed to execute buy order for {sym}")
        return None


# 7. BUY!!!!!!!!
def place_buy_order(api: AlpacaREST, symbol: str, price: float) -> dict:
    """
    1) Submit market buy.
    2) Poll until filled.
    3) Return dict with symbol, qty, and actual filled_avg_price.
    """
    try:
        account = api.get_account()
        cash = float(account.cash)
        logger.debug(f"Account cash available: ${cash:.2f}")

        qty = int(cash // price)
        if qty <= 0:
            logger.error(f"Insufficient funds: cash=${cash:.2f}, price=${price:.2f}")
            return {}

        logger.info(f"Submitting BUY order: {symbol} qty={qty} @ approx ${price:.2f} (total: ${qty * price:.2f})")

        order = api.submit_order(
            symbol=symbol, qty=qty,
            side="buy", type="market", time_in_force="day"
        )
        logger.info(f"Order submitted with ID: {order.id}")

        # Wait for fill with timeout
        poll_count = 0
        max_polls = 30  # 30 * 0.5s = 15 second timeout

        while poll_count < max_polls:
            o = api.get_order(order.id)
            logger.debug(f"Order {order.id} status={o.status} (poll {poll_count + 1})")

            if o.status in ["filled", "partially_filled"]:
                filled_price = float(o.filled_avg_price)
                filled_qty = int(o.filled_qty)
                logger.info(f"🎉 Order {order.id} FILLED: {filled_qty} shares @ ${filled_price:.2f}")
                return {
                    "symbol": symbol,
                    "qty": filled_qty,
                    "filled_avg_price": filled_price,
                    "order_id": order.id
                }
            elif o.status in ["rejected", "canceled", "expired"]:
                logger.error(f"Order {order.id} failed with status: {o.status}")
                return {}

            poll_count += 1
            time.sleep(0.5)

        logger.error(f"Order {order.id} timeout after {max_polls * 0.5}s")
        return {}

    except Exception as e:
        logger.error(f"Buy order failed for {symbol}: {e}")
        return {}


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

async def monitor_trade_ws(
    api: AlpacaREST,
    symbol: str,
    qty: int,
    entry_price: float
):
    """
    Monitor the trade via WebSocket with enhanced logging and order verification
    """
    cutoff = get_cutoff_time(datetime.now(TZ).date())
    hit_target = False
    peak = entry_price
    tick_count = 0
    last_log_time = time.monotonic()

    uri = f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"
    logger.info(f"🔍 Starting trade monitoring for {symbol}")
    logger.info(f"Entry: ${entry_price:.2f}, Stop: ${entry_price * STOP_LOSS_FACTOR:.2f}, Target: ${entry_price * TARGET_FACTOR:.2f}")
    logger.info(f"Monitor until: {cutoff.time()}")

    # Helper function to verify sell order
    async def verify_and_handle_sell(reason: str, price: float):
        pnl = (price - entry_price) / entry_price * 100
        logger.info(f"{reason}: {symbol} @ ${price:.2f} (P&L: {pnl:+.2f}%)")
        
        # Submit sell order
        sell_order = await asyncio.to_thread(api.submit_order,
            symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day"
        )
        
        # Verify the order was accepted for the full quantity
        actual_sell_qty = int(sell_order.qty)
        if actual_sell_qty != qty:
            logger.error(f"❌ SELL ORDER MODIFIED: Requested {qty}, accepted {actual_sell_qty}")
            logger.error(f"❌ REMAINING POSITION: {qty - actual_sell_qty} shares of {symbol}")
            
            # Check actual position after the order
            try:
                final_position = await asyncio.to_thread(api.get_position, symbol)
                remaining_qty = int(final_position.qty)
                if remaining_qty > 0:
                    logger.error(f"❌ POSITION NOT FULLY CLOSED: {remaining_qty} shares remaining")
            except Exception as e:
                logger.warning(f"Could not verify final position: {e}")
        else:
            logger.info(f"✅ SELL ORDER ACCEPTED: {actual_sell_qty} shares")

    max_attempts = 5
    attempt = 0

    while attempt < max_attempts:
        try:
            async with websockets.connect(uri) as ws:
                attempt = 0  # Reset attempt counter on successful connection
                await ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
                logger.debug(f"Subscribed to monitoring feed for {symbol}")

                async for raw in ws:
                    data = json.loads(raw)
                    if data.get("type") != "trade":
                        continue

                    price = data["data"][-1]["p"]
                    now = datetime.now(TZ)
                    tick_count += 1

                    # Log every 100 ticks or every 60 seconds
                    current_time = time.monotonic()
                    if tick_count % 100 == 0 or (current_time - last_log_time) > 60:
                        pnl = (price - entry_price) / entry_price * 100
                        logger.info(f"📊 {symbol} @ ${price:.2f} (P&L: {pnl:+.2f}%) [tick #{tick_count}]")
                        last_log_time = current_time
                    else:
                        logger.debug(f"Tick {symbol} @ ${price:.2f} time={now.time()}")

                    # TIME EXIT CHECK
                    if not hit_target and now >= cutoff:
                        await verify_and_handle_sell("⏰ TIME EXIT", price)
                        await ws.close()
                        return

                    # STOP LOSS CHECK
                    if price <= entry_price * STOP_LOSS_FACTOR:
                        await verify_and_handle_sell("🛑 STOP LOSS", price)
                        await ws.close()
                        return

                    # PROFIT TARGET CHECK
                    if not hit_target and price >= entry_price * TARGET_FACTOR:
                        hit_target = True
                        peak = price
                        pnl = (price - entry_price) / entry_price * 100
                        logger.info(f"🎯 TARGET HIT: {symbol} @ ${price:.2f} (P&L: {pnl:+.2f}%) - Now trailing...")

                    # TRAILING STOP LOGIC
                    if hit_target:
                        if price > peak:
                            old_peak = peak
                            peak = price
                            logger.debug(f"New peak for {symbol}: ${old_peak:.2f} → ${peak:.2f}")
                        elif price < peak:
                            decline = (peak - price) / peak * 100
                            await verify_and_handle_sell(f"📉 TRAILING STOP", price)
                            await ws.close()
                            return

        except Exception as e:
            attempt += 1
            if attempt < max_attempts:
                logger.warning(f"WebSocket connection failed (attempt {attempt}/{max_attempts}): {e}")
                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            else:
                logger.error(f"Max WebSocket connection attempts reached ({max_attempts})")
                break

    logger.info(f"✅ Trade monitoring completed for {symbol}")

def main():
    logger.info("🚀 Starting Enhanced Earnings Trading Bot")

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

    today = datetime.now(TZ).date()
    prev_day = get_prev_business_day(today)
    start, end = prev_day.isoformat(), today.isoformat()
    logger.info(f"📅 Trading session: {start} → {end}")

    # Fetch and filter earnings data
    entries = fetch_earnings_calendar(fh, start, end)
    if not entries:
        logger.error("❌ No earnings entries found - exiting")
        sys.exit(1)

    candidates = filter_candidates(entries, fh, start, end)
    if not candidates:
        logger.error("❌ No viable candidates found - exiting")
        sys.exit(1)

    candidates = group_candidates_by_surprise(candidates)

    # Load previous closes
    prev_close = preload_prev_closes(alp, candidates, start)
    if not prev_close:
        logger.error("❌ No previous closes loaded - exiting")
        sys.exit(1)

    # Check market status before proceeding
    clock = alp.get_clock()

    # Get opening prices using new dual-source approach
    symbols = [c["symbol"] for c in candidates]
    logger.info("🔌 Starting dual-source price monitoring (WebSocket + REST API)...")

    try:
        best_candidate = asyncio.run(get_opening_prices_with_window(
            symbols=symbols,
            finnhub_key=FINNHUB_API_KEY,
            api=alp,
            prev_close=prev_close,
            window_seconds=15,
            max_rest_symbols=50
        ))
    except RuntimeError as e:
        logger.error(f"❌ Exiting due to timing constraint: {e}")
        sys.exit(1)

    if not best_candidate:
        logger.error("❌ No candidates met the % increase threshold - exiting")
        sys.exit(1)

    # Execute buy order for the best candidate
    buy_order = evaluate_candidates_and_buy(alp, best_candidate)
    if not buy_order:
        logger.error("❌ Buy order failed - exiting")
        sys.exit(1)

    # Monitor the trade
    sym   = buy_order["symbol"]
    qty   = buy_order["qty"]
    entry = float(buy_order["filled_avg_price"])

    logger.info(f"🎯 Now monitoring position: {qty} shares of {sym} @ ${entry:.2f}")
    asyncio.run(monitor_trade_ws(alp, sym, qty, entry))

    logger.info("🏁 Trading session completed")


if __name__ == "__main__":
    main()