import os
import sys
import time
import logging
import asyncio
import json
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
#fix these comments when running on ubuntu
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python 3.8 fallback
from typing import List, Dict, Optional, Set

import websockets
import finnhub
from alpaca_trade_api.rest import REST as AlpacaREST

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
        clock = api.get_clock()
        logger.debug(f"Market status check: is_open={clock.is_open}, timestamp={clock.timestamp}")
        if clock.is_open:
            logger.info("Market opened at %s", clock.timestamp.astimezone(TZ))
            return
        next_open = clock.next_open.astimezone(TZ)
        now = datetime.now(TZ)
        secs = (next_open - now).total_seconds()
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

async def pre_connect_and_wait_for_opening(
    symbols: List[str], 
    key: str,
    api: AlpacaREST,
    prev_close: Dict[str, float]
) -> Optional[Dict[str, float]]:
    """
    Wait until market opens, then connect to WebSocket and wait for first trades.
    """
    logger.info("Waiting for market to open before connecting to WebSocket...")
    sleep_until_market_open(api)
    
    uri = f"wss://ws.finnhub.io?token={key}"
    logger.info(f"Market is open. Connecting to Finnhub WebSocket for {len(symbols)} symbols")

    best_candidate = None
    pending_symbols = set(symbols)
    connection_attempts = 0
    max_attempts = 3

    
    while connection_attempts < max_attempts:
        try:
            logger.debug(f"WebSocket connection attempt {connection_attempts + 1}/{max_attempts}")
            async with websockets.connect(uri) as ws:
                logger.info("WebSocket connected successfully, subscribing to symbols")
                
                # Subscribe to all symbols while market is still closed
                for s in symbols:
                    subscribe_msg = json.dumps({"type": "subscribe", "symbol": s})
                    await ws.send(subscribe_msg)
                    logger.debug(f"Pre-subscribed to {s}")
                    await asyncio.sleep(0.1)  # Small delay between subscriptions
                
                logger.info("All symbols subscribed. Waiting for market open and first trades...")
                
                # Check if market is open yet
                clock = api.get_clock()
                if not clock.is_open:
                    next_open = clock.next_open.astimezone(TZ)
                    logger.info(f"Market opens at {next_open.time()}, WebSocket ready and waiting...")
                
                message_count = 0
                start_time = time.monotonic()
                
                # Wait for trades to start flowing
                while pending_symbols:
                    try:
                        # Set a reasonable timeout for each message
                        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        message_count += 1
                        
                        if message_count % 100 == 0:
                            logger.debug(f"Processed {message_count} WebSocket messages, still waiting for: {pending_symbols}")
                        
                        data = json.loads(raw)
                        
                        if data.get("type") == "trade":
                            for t in data["data"]:
                                sym, price, timestamp = t["s"], t["p"], t.get("t", 0)
                                if sym in pending_symbols:
                                    pending_symbols.remove(sym)
                                    trade_time = datetime.fromtimestamp(timestamp / 1000, tz=TZ) if timestamp else "unknown"
                                    logger.info(f"🎯 OPENING PRICE {sym} @ ${price:.2f} (time: {trade_time}")
                                    
                                    # Immediately check if this meets our 3% threshold
                                    if sym in prev_close:
                                        pct_increase = (price - prev_close[sym]) / prev_close[sym] * 100
                                        logger.info(f"{sym}: {pct_increase:.2f}% increase from previous close ${prev_close[sym]:.2f}")
                                        
                                        if pct_increase > MIN_PCT_INCREASE:
                                            MIN_PCT_INCREASE = pct_increase
                                            best_candidate = {
                                                "symbol": sym,
                                                "price": price,
                                                "pct_increase": pct_increase
                                            }
                                            logger.info(f"New best candidate: {sym} @ {pct_increase:.2f}%")
                                    
                                    if not pending_symbols:
                                        logger.info("All opening prices processed!")
                                        break
                        
                        elif data.get("type") == "ping":
                            logger.debug("Received WebSocket ping")
                            
                        else:
                            logger.debug(f"Received WebSocket message type: {data.get('type')}")
                            
                    except asyncio.TimeoutError:
                        clock = api.get_clock()
                        logger.warning(f"No messages for 30s, market_open: {clock.is_open}, pending: {pending_symbols}")
    
                        # Only trigger after 9:35 AM if market is open
                        if clock.is_open and datetime.now(TZ).time() >= datetime.strptime("09:35:00", "%H:%M:%S").time():
                            logger.error("Market open for 5+ minutes but no trade data received")
                            break
                
                return best_candidate
                
        except Exception as e:
            connection_attempts += 1
            logger.error(f"WebSocket connection failed (attempt {connection_attempts}): {e}")
            if connection_attempts < max_attempts:
                logger.info(f"Retrying WebSocket connection in 5 seconds...")
                await asyncio.sleep(5)
            else:
                logger.error("Max WebSocket connection attempts reached")
                break
    
    logger.warning(f"WebSocket connection failed after {max_attempts} attempts")
    return best_candidate

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
            
            if o.status == "filled":
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

def evaluate_candidates_and_buy(
    api: AlpacaREST,
    best_candidate: Optional[dict]
) -> Optional[dict]:
    """
    Execute buy order for the best candidate that met the criteria (if any)
    """
    if not best_candidate:
        logger.info("No candidates met the 3% increase threshold")
        return None
    
    sym = best_candidate["symbol"]
    price = best_candidate["price"]
    pct_increase = best_candidate["pct_increase"]
    
    clock = api.get_clock()
    if not clock.is_open:
        logger.info("Market not open yet — waiting before placing buy order...")
        sleep_until_market_open(api)

    logger.info(f"📈 Best candidate: {sym} with {pct_increase:.2f}% increase - executing buy!")
    return place_buy_order(api, sym, price)

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
    Monitor the trade via WebSocket with enhanced logging
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
    
    max_attempts = 3
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
                        pnl = (price - entry_price) / entry_price * 100
                        logger.info(f"⏰ TIME EXIT: {symbol} @ ${price:.2f} (P&L: {pnl:+.2f}%)")
                        await asyncio.to_thread(api.submit_order,
                            symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day"
                        )
                        return

                    # STOP LOSS CHECK
                    if price <= entry_price * STOP_LOSS_FACTOR:
                        pnl = (price - entry_price) / entry_price * 100
                        logger.info(f"🛑 STOP LOSS: {symbol} @ ${price:.2f} (P&L: {pnl:+.2f}%)")
                        await asyncio.to_thread(api.submit_order,
                            symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day"
                        )
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
                            pnl = (price - entry_price) / entry_price * 100
                            logger.info(f"📉 TRAILING STOP: {symbol} @ ${price:.2f} (peak: ${peak:.2f}, decline: {decline:.2f}%, P&L: {pnl:+.2f}%)")
                            await asyncio.to_thread(api.submit_order,
                                symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day"
                            )
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
  

    # Pre-connect to WebSocket and wait for opening prices
    symbols = [c["symbol"] for c in candidates]
    logger.info("🔌 Waiting for market open to start WebSocket connection...")
    
    best_candidate = asyncio.run(pre_connect_and_wait_for_opening(
        symbols, FINNHUB_API_KEY, alp, prev_close
    ))
    
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