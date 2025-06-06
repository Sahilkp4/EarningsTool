import os
import sys
import time
import logging
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional

import finnhub
from alpaca_trade_api.rest import REST as AlpacaREST, APIError

# ─── Constants & Configuration ─────────────────────────────────────────────────
TZ = ZoneInfo("America/New_York")

# Load API keys from environment
FINNHUB_API_KEY       = os.getenv("FINNHUB_API_KEY")
APCA_KEY              = os.getenv("APCA_API_KEY_ID")
APCA_SECRET           = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL       = "https://paper-api.alpaca.markets"

# Trading parameters
STOP_LOSS_FACTOR      = 0.96    # 4% stop‐loss
TARGET_FACTOR         = 1.0267  # 2.67% profit target
MONITOR_END_HOUR      = 15
MONITOR_END_MINUTE    = 33
SURPRISE_THRESHOLD    = 10      # minimum EPS surprise (%) to consider
MAX_SURPRISE          = 500     # Max EPS surprise % to consider 
MC_THRESHOLD          = 10_000  # max market cap (in millions USD) to consider
GROUP_THRESHOLD       = 50      # % gap to split groups
# Alpaca API rate‐limit buffers
GROUP_SLEEP_SEC       = 1.0     # pause between trade‐polling bursts


# ─── Logging Configuration ─────────────────────────────────────────────────────
def configure_logging() -> logging.Logger:
    logger = logging.getLogger("earnings_trader")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S %Z"
        )
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # File handler
        log_path = os.path.join(os.path.dirname(__file__), "session_output.txt")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

logger = configure_logging()


# ─── Utilities ─────────────────────────────────────────────────────────────────
def get_prev_business_day(ref_date: date) -> date:
    d = ref_date
    while True:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            return d

def is_within_earnings_window(entry: dict, start_str: str, end_str: str) -> bool:
    e_date, e_hour = entry.get("date"), entry.get("hour")
    return ((e_date == start_str and e_hour == "amc") or
            (e_date == end_str   and e_hour == "bmo"))

def sleep_until_market_open(api: AlpacaREST, tz: ZoneInfo):
    """Poll Alpaca clock until market opens."""
    while True:
        try:
            clock = api.get_clock()
        except Exception as e:
            logger.warning(f"Failed to fetch Alpaca clock: {e}; retrying in 30s")
            time.sleep(30)
            continue

        if clock.is_open:
            logger.info("Market is OPEN at %s", clock.timestamp.astimezone(tz))
            return

        next_open = clock.next_open.astimezone(tz)
        now = datetime.now(tz)
        secs_to_open = (next_open - now).total_seconds()
        logger.debug(f"Market closed. Now: {now.time()}, Next open: {next_open.time()} (in {secs_to_open:.1f}s)")

        if secs_to_open > 300:
            logger.info("Sleeping 5m until ~%s", next_open)
            time.sleep(300)
        else:
            logger.info("Sleeping until ~%s", next_open)
            time.sleep(max(0, secs_to_open))

def get_cutoff_time(today: date, hour: int, minute: int, tz: ZoneInfo) -> datetime:
    return datetime(
        year=today.year,
        month=today.month,
        day=today.day,
        hour=hour,
        minute=minute,
        second=0,
        tzinfo=tz
    )


# ─── Core Logic ─────────────────────────────────────────────────────────────────
def fetch_earnings_calendar(fh_client: finnhub.Client,
                            start_str: str,
                            end_str: str) -> List[dict]:
    try:
        cal = fh_client.earnings_calendar(symbol=None, _from=start_str, to=end_str)
        entries = cal.get("earningsCalendar", [])
        logger.info(f"Fetched {len(entries)} earnings‐calendar entries")
        return entries
    except Exception as e:
        logger.exception("Failed to fetch earnings calendar: %s", e)
        sys.exit(1)

def filter_candidates(entries: List[dict], fh_client: finnhub.Client,
                      start_str: str, end_str: str) -> List[dict]:
    candidates = []
    logger.debug("Starting candidate filtering based on earnings surprise and market cap")
    for e in entries:
        sym = e.get("symbol")
        est = e.get("epsEstimate")
        act = e.get("epsActual")
        logger.debug(f"Evaluating {sym}: estimate={est}, actual={act}")

        # Skip if estimate or actual is missing, or if est == 0 (avoid divide‐by‐zero)
        if est is None or act is None or est == 0:
            logger.debug(f"Skipping {sym}: missing or zero estimate/actual")
            continue
       
        # ───── ADD THIS CHECK ─────
        if act < 0:
            logger.debug(f"Skipping {sym}: actual EPS is still negative")
            continue
        # ─────────────────────────

        # Now compute surprise over abs(est). 
        # If est < 0 but act ≥ 0, this measures “neg to pos” correctly.
        surprise = (act - est) / abs(est) * 100

        if surprise > MAX_SURPRISE:
            logger.debug(f"Skipping {sym}: surprise={surprise:.2f}% exceeds cap of {MAX_SURPRISE}%")
            continue

        if surprise <= SURPRISE_THRESHOLD or not is_within_earnings_window(e, start_str, end_str):
            logger.debug(f"Skipping {sym}: surprise={surprise:.2f}% below threshold or wrong window")
            continue

        try:
            profile = fh_client.company_profile2(symbol=sym)
            mc = float(profile.get("marketCapitalization", 0)) / 1e6
        except Exception as exc:
            logger.warning(f"Could not fetch profile for {sym}: {exc}")
            continue

        if mc <= MC_THRESHOLD:
            candidates.append({"symbol": sym, "surprise": surprise})
            logger.info(f"Candidate {sym}: surprise={surprise:.2f}% mc={mc}M")
        else:
            logger.debug(f"Skipping {sym}: market cap {mc}M above threshold")

    candidates.sort(key=lambda x: x["surprise"], reverse=True)
    logger.info(f"Total filtered candidates: {len(candidates)}")
    logger.debug(f"Filtered list: {[c['symbol'] for c in candidates]}")
    return candidates


def group_candidates_by_surprise(candidates: List[Dict[str, float]]) -> List[List[Dict[str, float]]]:
    """
    Group candidates so that within each group, the difference between
    highest and lowest surprise ≤ 15, and at most 5 symbols per group.
    """
    groups: List[List[Dict[str, float]]] = []
    remaining = candidates.copy()
    logger.debug("Starting grouping of candidates by surprise ranges")

    while remaining:
        base_entry = remaining.pop(0)
        base_surp = base_entry["surprise"]
        group = [base_entry]
        to_remove = []
        logger.debug(f"Forming new group with base {base_entry['symbol']} (surprise={base_surp:.2f}%)")
        for entry in remaining:
            if len(group) >= 5:
                break
            if base_surp - entry["surprise"] <= GROUP_THRESHOLD:
                group.append(entry)
                to_remove.append(entry)
                logger.debug(f"  Adding {entry['symbol']} (surprise={entry['surprise']:.2f}%) to group")
        for entry in to_remove:
            remaining.remove(entry)
        groups.append(group)

    logger.info(f"Formed {len(groups)} candidate groups")
    for i, grp in enumerate(groups, start=1):
        logger.debug(f" Group {i}: {[x['symbol'] for x in grp]} with surprises {[x['surprise'] for x in grp]}")
    return groups

def preload_prev_closes(api: AlpacaREST,
                        groups: List[List[Dict[str, float]]],
                        start_str: str) -> Dict[str, float]:
    """
    Before market open, fetch exactly the 'start_str' day's close for every symbol.
    Uses timeframe=1Day with start/end = start_str's midnight to 23:59:59, then takes .c.
    start_str is in "YYYY-MM-DD" form (previous business day).
    Returns a dict: {symbol: prev_close}.
    """
    symbols = { entry["symbol"] for group in groups for entry in group }
    prev_close_dict: Dict[str, float] = {}
    logger.info(f"Preloading previous close for {len(symbols)} symbols (date = {start_str})")

    # Build ISO timestamps for the entire 'start_str' day in EST
    day_start_iso = f"{start_str}T00:00:00-04:00"
    day_end_iso   = f"{start_str}T23:59:59-04:00"

    for sym in symbols:
        try:
            bars_yesterday = api.get_bars(
                sym,
                timeframe="1Day",
                start=day_start_iso,
                end=day_end_iso,
                limit=1,
                adjustment="raw"
            )
            if bars_yesterday:
                prev_close = bars_yesterday[0].c
                prev_close_dict[sym] = prev_close
                logger.debug(f"{sym}: {start_str} close={prev_close:.2f}")
            else:
                logger.warning(f"No 1Day bar found for {sym} on {start_str}; skipping symbol")
        except Exception as exc:
            logger.warning(f"Error fetching {start_str} bar for {sym}: {exc}")

        # Pace requests lightly (we're pre-open)
        time.sleep(0.11)

    logger.debug(f"Preloaded prev_close_dict: {prev_close_dict}")
    return prev_close_dict

def evaluate_groups_and_buy(api: AlpacaREST,
                            fh_client: finnhub.Client,
                            groups: List[List[Dict[str, float]]],
                            prev_close_dict: Dict[str, float]) -> Optional[dict]:
    """
    For each group, in turn:
      1) At or after 9:30, poll Finnhub quote for each symbol in a burst
         until at least one symbol shows a post-open timestamp.
      2) Among symbols with a post-open quote, compute pct change = (price / prev_close) - 1.
      3) If any pct_change > 0, buy the one with highest pct_change at market.
      4) If none positive, move to next group.
    """
    today = datetime.now(TZ).date()
    market_open_dt = datetime(today.year, today.month, today.day, 9, 30, 0, tzinfo=TZ)
    logger.debug(f"Market open datetime programmed as {market_open_dt}")

    # If running before 9:30, wait until then
    now = datetime.now(TZ)
    if now < market_open_dt:
        secs_to_open = (market_open_dt - now).total_seconds()
        logger.info(f"Waiting {secs_to_open:.1f}s until 9:30 AM")
        time.sleep(secs_to_open)

    for idx, group in enumerate(groups, start=1):
        symbols_in_group = [entry["symbol"] for entry in group]
        logger.info(f"Evaluating group {idx}/{len(groups)}: {symbols_in_group}")

        # 1) Poll Finnhub quote in a burst (no small sleep between calls)
        first_quotes: Dict[str, float] = {}
        while True:
            logger.debug(f"Burst polling for post-open quotes in group {idx}")
            for sym in symbols_in_group:
                prev_close = prev_close_dict.get(sym)
                if prev_close is None or prev_close <= 0:
                    logger.debug(f"No valid prev_close for {sym}; skipping this symbol")
                    continue

                try:
                    quote = fh_client.quote(sym)
                    trade_unix = quote.get('t')
                    if not trade_unix:
                        continue
                    trade_dt = datetime.fromtimestamp(trade_unix, tz=timezone.utc).astimezone(TZ)
                    trade_price = quote.get('c')
                    logger.debug(f"{sym} latest quote at {trade_dt.time()} price {trade_price:.2f}")
                    if trade_dt > market_open_dt:
                        if sym not in first_quotes:
                            first_quotes[sym] = trade_price
                            logger.info(f"Recorded first post-open quote for {sym} at {trade_dt.time()} → {trade_price:.2f}")
                except Exception as exc:
                    logger.warning(f"Error fetching Finnhub quote for {sym}: {exc}")

            if first_quotes:
                logger.debug(f"First post-open quotes received: {first_quotes}")
                break
            else:
                logger.debug(f"No post-open quotes yet for group {idx}; sleeping {GROUP_SLEEP_SEC}s before retry")
                time.sleep(GROUP_SLEEP_SEC)

        # 2) Compute percentage change vs prev_close for each symbol that quoted
        positive_moves: Dict[str, float] = {}
        for sym, quote_price in first_quotes.items():
            prev_close = prev_close_dict[sym]
            pct_change = (quote_price / prev_close) - 1.0
            logger.info(f"{sym}: post-open quote={quote_price:.2f}, prev_close={prev_close:.2f}, pct_change={pct_change*100:.2f}%")
            if pct_change > 0:
                positive_moves[sym] = pct_change

        logger.debug(f"Positive moves in group {idx}: {positive_moves}")
        if not positive_moves:
            logger.info(f"No positive moves in group {idx}; moving to next group")
            continue

        # 3) Choose symbol with highest positive pct_change
        chosen_sym = max(positive_moves, key=positive_moves.get)
        chosen_price = first_quotes[chosen_sym]
        logger.info(f"Chosen to buy {chosen_sym} with pct_change={positive_moves[chosen_sym]*100:.2f}% at reference price {chosen_price:.2f}")

        # 4) Submit market buy for as many shares as cash allows
        try:
            acct = api.get_account()
            cash_available = float(acct.cash)
            max_shares     = int(cash_available // chosen_price)
            logger.debug(f"cash_available=${cash_available:.2f} → max_shares={max_shares} for {chosen_sym}")
        except Exception as exc:
            logger.exception(f"Failed to fetch account data for {chosen_sym}: {exc}")
            return None

        if max_shares < 1:
            logger.error(f"Insufficient cash to buy even one share of {chosen_sym}")
            return None

        order_qty = max_shares
        while order_qty > 0:
            logger.debug(f"Attempting to buy {order_qty} shares of {chosen_sym}")
            try:
                order = api.submit_order(
                    symbol=chosen_sym,
                    qty=order_qty,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                logger.info(f"SUBMITTED BUY {chosen_sym} | qty={order_qty} | order_id={order.id}")
                actual_buy_price = chosen_price  # reference price = first post-open quote
                return {
                    "symbol":    chosen_sym,
                    "buy_time":  datetime.now(TZ).replace(microsecond=0).isoformat(),
                    "buy_price": actual_buy_price,
                    "qty":       order_qty,
                    "order_id":  order.id
                }
            except APIError as e:
                err_msg = str(e).lower()
                logger.warning(f"Buy {chosen_sym} qty={order_qty} failed: {e}")
                if "insufficient buying power" in err_msg:
                    order_qty -= 1
                    logger.debug(f"Retrying {chosen_sym} with qty={order_qty}")
                    time.sleep(1)
                    continue
                else:
                    logger.exception(f"Unexpected APIError for {chosen_sym}: {e}")
                    break
            except Exception as e:
                logger.exception(f"Unexpected exception when buying {chosen_sym}: {e}")
                break

        logger.error(f"Could not place any buy order for {chosen_sym}; moving to next group")
        time.sleep(GROUP_SLEEP_SEC)

    # If no purchase succeeded across all groups
    logger.info("No purchase executed across any group")
    return None

def monitor_trade(api: AlpacaREST, fh_client: finnhub.Client, buy_info: dict):
    symbol    = buy_info["symbol"]
    qty       = buy_info["qty"]
    buy_price = buy_info["buy_price"]

    stop_loss_factor = STOP_LOSS_FACTOR
    target_factor    = TARGET_FACTOR

    cutoff_time = get_cutoff_time(datetime.now(TZ).date(),
                                  MONITOR_END_HOUR, MONITOR_END_MINUTE, TZ)
    hit_target = False
    peak_price = None

    logger.info(f"Monitoring {symbol} | qty={qty} | entry_price={buy_price:.2f}")

    while True:
        try:
            quote = fh_client.quote(symbol)
            price = quote.get('c')
            trade_unix = quote.get('t')
            if price is None or trade_unix is None:
                logger.debug(f"No valid Finnhub quote for {symbol}; skipping iteration")
                time.sleep(0.30)
                continue
            trade_dt = datetime.fromtimestamp(trade_unix, tz=timezone.utc).astimezone(TZ)
            now = datetime.now(TZ)
            logger.debug(f"{symbol} latest quote at {trade_dt.time()} price {price:.2f}")

            # (1) Time-based exit (only if target not yet hit)
            if not hit_target and now >= cutoff_time:
                logger.info(f"{symbol} time cutoff reached at {now.time()}; selling at market")
                sell = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="day"
                )
                logger.info(f"TIME‐EXIT SELL {symbol} | order_id={sell.id}")
                break

            # (2) Hard stop-loss
            if price <= buy_price * stop_loss_factor:
                logger.info(f"{symbol} hit stop-loss at price {price:.2f} (<= {buy_price*stop_loss_factor:.2f}); selling")
                sell = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="day"
                )
                logger.info(f"STOP‐LOSS SELL {symbol} | order_id={sell.id}")
                break

            # (3) Initial profit target reached
            if not hit_target and price >= buy_price * target_factor:
                hit_target = True
                peak_price = price
                logger.info(f"{symbol} hit target at price {price:.2f}; peak_price set")

            # (4) Trailing stop once target is hit
            if hit_target:
                if price > peak_price:
                    logger.debug(f"{symbol} new peak price {price:.2f} (old peak {peak_price:.2f})")
                    peak_price = price
                elif price < peak_price:
                    logger.info(f"{symbol} price dropped below peak {peak_price:.2f} to {price:.2f}; trailing-stop sell")
                    sell = api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="sell",
                        type="market",
                        time_in_force="day"
                    )
                    logger.info(f"TRAILING‐STOP SELL {symbol} | order_id={sell.id}")
                    break

        except APIError as e:
            logger.warning(f"APIError in monitoring loop for {symbol}: {e}")
            time.sleep(0.30)
            continue
        except Exception as e:
            logger.exception(f"Unexpected error in monitoring loop for {symbol}: {e}")
            time.sleep(0.30)
            continue

        # Poll every ~0.30 seconds for the latest quote
        time.sleep(0.30)

    logger.info("Monitoring ended. Trading run completed.")

def main():
    # 1) Verify required environment variables
    if not FINNHUB_API_KEY:
        logger.error("Missing FINNHUB_API_KEY; exiting")
        sys.exit(1)
    if not APCA_KEY or not APCA_SECRET:
        logger.error("Missing Alpaca credentials; exiting")
        sys.exit(1)

    fh_client  = finnhub.Client(api_key=FINNHUB_API_KEY)
    alpaca_api = AlpacaREST(APCA_KEY, APCA_SECRET, ALPACA_BASE_URL)
    logger.info("Finnhub and Alpaca clients initialized")

    # 2) Compute today & previous business day
    today     = datetime.now(TZ).date()
    prev_bday = get_prev_business_day(today)
    start_str = prev_bday.isoformat()   # e.g. "2025-06-02"
    end_str   = today.isoformat()       # e.g. "2025-06-03"
    logger.info(f"Fetching earnings calendar from {start_str} to {end_str}")

    # 3) Fetch & filter candidates (AMC/BMO, surprise, market cap)
    entries  = fetch_earnings_calendar(fh_client, start_str, end_str)
    filtered = filter_candidates(entries, fh_client, start_str, end_str)
    if not filtered:
        logger.error("No candidates found; exiting")
        sys.exit(1)

    # 4) Group candidates by surprise‐range (each group sorted by surprise descending)
    groups = group_candidates_by_surprise(filtered)

    # 5) Preload exactly 'start_str' (June 2) close for every symbol in all groups
    prev_close_dict = preload_prev_closes(alpaca_api, groups, start_str)
    if not prev_close_dict:
        logger.error("Failed to preload any previous closes; exiting")
        sys.exit(1)

    # 6) Wait for market open if necessary
    try:
        clock = alpaca_api.get_clock()
    except Exception as e:
        logger.exception(f"Failed to fetch Alpaca clock: {e}")
        sys.exit(1)

    if not clock.is_open:
        sleep_until_market_open(alpaca_api, TZ)
    else:
        logger.info("Market already OPEN at %s", clock.timestamp.astimezone(TZ))

    # 7) Evaluate each group and attempt to buy
    buy_info = evaluate_groups_and_buy(alpaca_api, fh_client, groups, prev_close_dict)
    if not buy_info:
        logger.error("No purchase succeeded across all groups; exiting")
        sys.exit(1)

    # 8) Monitor the trade (polling every 0.30s) and potentially sell
    monitor_trade(alpaca_api, fh_client, buy_info)

if __name__ == "__main__":
    main()
