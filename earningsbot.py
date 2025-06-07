import os
import sys
import time
import logging
import asyncio
import json

from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional

import websockets
import finnhub
from alpaca_trade_api.rest import REST as AlpacaREST

# ─── Constants & Configuration ───────────────────────────────────────────────
TZ = ZoneInfo("America/New_York")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
APCA_KEY        = os.getenv("APCA_API_KEY_ID")
APCA_SECRET     = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

STOP_LOSS_FACTOR   = 0.96
TARGET_FACTOR      = 1.0267
MONITOR_END_HOUR   = 15
MONITOR_END_MINUTE = 33

SURPRISE_THRESHOLD = 10
MAX_SURPRISE       = 600
MC_THRESHOLD       = 100_000

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
    logger.debug(f"{entry.get('symbol')} window check: {ok}")
    return ok

def sleep_until_market_open(api: AlpacaREST):
    while True:
        clock = api.get_clock()
        if clock.is_open:
            logger.info("Market opened at %s", clock.timestamp.astimezone(TZ))
            return
        next_open = clock.next_open.astimezone(TZ)
        now = datetime.now(TZ)
        secs = (next_open - now).total_seconds()
        logger.debug(f"Market closed. Now={now.time()}, Next open={next_open.time()} in {secs:.1f}s")
        if secs > 300:
            time.sleep(300)
        else:
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
    cal = fh.earnings_calendar(symbol=None, _from=frm, to=to)
    entries = cal.get("earningsCalendar", [])
    logger.info(f"Fetched {len(entries)} earnings entries from {frm} to {to}")
    return entries

def filter_candidates(
    entries: List[dict],
    fh: finnhub.Client,
    frm: str,
    to: str
) -> List[dict]:
    candidates: List[dict] = []
    last_profile_time = 0.0
    for e in entries:
        sym = e.get("symbol")
        est, act = e.get("epsEstimate"), e.get("epsActual")
        logger.debug(f"Evaluating {sym}: est={est}, act={act}")
        if est is None or act is None or est == 0 or act < 0:
            logger.debug(f"Skipping {sym}: missing/zero estimate or negative actual")
            continue

        surprise = (act - est) / abs(est) * 100
        logger.debug(f"{sym} surprise={surprise:.2f}%")
        if surprise <= SURPRISE_THRESHOLD or surprise > MAX_SURPRISE:
            logger.debug(f"Skipping {sym}: surprise outside [{SURPRISE_THRESHOLD}, {MAX_SURPRISE}]")
            continue
        if not is_within_earnings_window(e, frm, to):
            continue

        # rate-limit company_profile2
        now = time.monotonic()
        if now - last_profile_time < 1.1:
            pause = 1.1 - (now - last_profile_time)
            logger.debug(f"Sleeping {pause:.2f}s to respect REST limits")
            time.sleep(pause)

        try:
            profile = fh.company_profile2(symbol=sym)
            last_profile_time = time.monotonic()
            mc = float(profile.get("marketCapitalization", 0)) / 1e6
            logger.debug(f"{sym} MC={mc:.1f}M")
        except Exception as exc:
            logger.warning(f"Profile fetch failed for {sym}: {exc}")
            continue

        if mc <= MC_THRESHOLD:
            candidates.append({"symbol": sym, "surprise": surprise})
            logger.info(f"Candidate {sym}: surprise={surprise:.2f}%, mc={mc:.1f}M")
        else:
            logger.debug(f"Skipping {sym}: MC {mc:.1f}M > {MC_THRESHOLD}M")

    logger.info(f"Total candidates after filter: {len(candidates)}")
    return candidates

def group_candidates_by_surprise(
    cands: List[Dict[str, float]]
) -> List[Dict[str, float]]:
    sorted_c = sorted(cands, key=lambda x: x["surprise"], reverse=True)
    logger.debug("Sorted candidates by surprise: " +
                 ", ".join(f"{c['symbol']}({c['surprise']:.1f}%)"
                           for c in sorted_c))
    return sorted_c

async def fetch_opening_prices_ws(
    symbols: List[str],
    key: str,
    timeout: float = 10.0
) -> Dict[str, float]:
    uri = f"wss://ws.finnhub.io?token={key}"
    logger.info(f"Connecting to Finnhub WS for {len(symbols)} symbols")
    opening: Dict[str, float] = {}
    pending = set(symbols)
    start = time.monotonic()

    async with websockets.connect(uri) as ws:
        for s in symbols:
            await ws.send(json.dumps({"type": "subscribe", "symbol": s}))
            logger.debug(f"Subscribed WS {s}")
        while pending and (time.monotonic() - start) < timeout:
            raw = await asyncio.wait_for(ws.recv(),
                                         timeout=timeout - (time.monotonic() - start))
            data = json.loads(raw)
            if data.get("type") == "trade":
                for t in data["data"]:
                    sym, price = t["s"], t["p"]
                    if sym in pending:
                        opening[sym] = price
                        pending.remove(sym)
                        logger.info(f"First open price {sym} @ {price:.2f}")
    logger.debug(f"Opening prices collected: {opening}")
    return opening

def place_buy_order(api: AlpacaREST, symbol: str, price: float) -> dict:
    """
    1) Submit market buy.
    2) Poll until filled.
    3) Return dict with symbol, qty, and actual filled_avg_price.
    """
    cash = float(api.get_account().cash)
    qty  = int(cash // price)
    logger.info(f"Submitting BUY {symbol} qty={qty} @ approx {price:.2f}")
    order = api.submit_order(
        symbol=symbol, qty=qty,
        side="buy", type="market", time_in_force="day"
    )

    # wait for fill
    while True:
        o = api.get_order(order.id)
        logger.debug(f"Order {order.id} status={o.status}")
        if o.status == "filled":
            filled_price = float(o.filled_avg_price)
            logger.info(f"Order {order.id} FILLED @ {filled_price:.2f}")
            return {"symbol": symbol, "qty": qty, "filled_avg_price": filled_price}
        time.sleep(0.2)

def evaluate_candidates_and_buy(
    api: AlpacaREST,
    fh: finnhub.Client,
    candidates: List[Dict[str, float]],
    prev_close: Dict[str, float]
) -> Optional[dict]:
    sorted_c = group_candidates_by_surprise(candidates)
    symbols = [c["symbol"] for c in sorted_c]
    logger.info("Streaming opening prices for all candidates")
    opening = asyncio.run(fetch_opening_prices_ws(symbols, FINNHUB_API_KEY))

    # super-surprise quick-buy
    top = sorted_c[0]["symbol"]
    if top in opening:
        pct = (opening[top] - prev_close[top]) / prev_close[top] * 100
        logger.debug(f"{top} open change {pct:.2f}%")
        if pct >= 6.0:
            return place_buy_order(api, top, opening[top])

    # fallback threshold-buy
    for c in sorted_c:
        s = c["symbol"]
        if s in opening:
            pct = (opening[s] - prev_close[s]) / prev_close[s] * 100
            logger.debug(f"{s} open change {pct:.2f}%")
            if pct >= 3.0:
                return place_buy_order(api, s, opening[s])

    logger.info("No open-day buy thresholds met")
    return None

async def monitor_trade_ws(
    api: AlpacaREST,
    symbol: str,
    qty: int,
    entry_price: float
):
    cutoff = get_cutoff_time(datetime.now(TZ).date())
    hit_target = False
    peak = entry_price

    uri = f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"
    logger.info(f"Starting WS-monitor for {symbol} entry={entry_price:.2f}")
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
        logger.debug(f"Subscribed WS-monitor {symbol}")
        async for raw in ws:
            data = json.loads(raw)
            if data.get("type") != "trade":
                continue
            price = data["data"][-1]["p"]
            now = datetime.now(TZ)
            logger.debug(f"Tick {symbol} @ {price:.2f} time={now.time()}")

            # time-exit
            if not hit_target and now >= cutoff:
                logger.info(f"Time-exit {symbol} @ {price:.2f}")
                await asyncio.to_thread(api.submit_order,
                    symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day"
                )
                break

            # stop-loss
            if price <= entry_price * STOP_LOSS_FACTOR:
                logger.info(f"Stop-loss {symbol} @ {price:.2f}")
                await asyncio.to_thread(api.submit_order,
                    symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day"
                )
                break

            # profit target
            if not hit_target and price >= entry_price * TARGET_FACTOR:
                hit_target = True
                peak = price
                logger.info(f"Hit-target {symbol} @ {price:.2f}")

            # trailing-stop
            if hit_target:
                if price > peak:
                    peak = price
                elif price < peak:
                    logger.info(f"Trailing-stop {symbol} @ {price:.2f} (peak {peak:.2f})")
                    await asyncio.to_thread(api.submit_order,
                        symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day"
                    )
                    break

    logger.info("WS-monitor completed")

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
            bars = api.get_bars(
                sym, timeframe="1Day",
                start=start_iso, end=end_iso,
                limit=1, adjustment="raw"
            )
            if bars:
                out[sym] = bars[0].c
                logger.debug(f"{sym} prev_close={bars[0].c:.2f}")
        except Exception as e:
            logger.warning(f"Failed to fetch prev_close for {sym}: {e}")
        time.sleep(0.1)
    return out

def main():
    if not all([FINNHUB_API_KEY, APCA_KEY, APCA_SECRET]):
        logger.error("Missing API keys—exiting")
        sys.exit(1)

    fh = finnhub.Client(api_key=FINNHUB_API_KEY)
    alp = AlpacaREST(APCA_KEY, APCA_SECRET, ALPACA_BASE_URL)
    logger.info("Finnhub and Alpaca clients initialized")

    today = datetime.now(TZ).date()
    prev_day = get_prev_business_day(today)
    start, end = prev_day.isoformat(), today.isoformat()

    entries = fetch_earnings_calendar(fh, start, end)
    candidates = filter_candidates(entries, fh, start, end)
    if not candidates:
        logger.error("No candidates—exiting")
        sys.exit(1)

    candidates = group_candidates_by_surprise(candidates)
    prev_close = preload_prev_closes(alp, candidates, start)
    if not prev_close:
        logger.error("No previous closes—exiting")
        sys.exit(1)

    clock = alp.get_clock()
    if not clock.is_open:
        sleep_until_market_open(alp)

    buy_order = evaluate_candidates_and_buy(alp, fh, candidates, prev_close)
    if not buy_order:
        logger.error("No buy executed—exiting")
        sys.exit(1)

    # Use the actual filled price as entry for monitoring
    sym   = buy_order["symbol"]
    qty   = buy_order["qty"]
    entry = float(buy_order["filled_avg_price"])

    asyncio.run(monitor_trade_ws(alp, sym, qty, entry))

if __name__ == "__main__":
    main()
