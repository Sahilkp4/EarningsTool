import os
import sys
import time
import logging
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional

import finnhub
from alpaca_trade_api.rest import REST as AlpacaREST, APIError

# ─── Constants & Configuration ─────────────────────────────────────────────────
TZ = ZoneInfo("America/New_York")

# Load API keys from environment (no hard-coding)
FINNHUB_API_KEY       = os.getenv("FINNHUB_API_KEY")
APCA_KEY              = os.getenv("APCA_API_KEY_ID")
APCA_SECRET           = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL       = "https://paper-api.alpaca.markets"

# Trading parameters (easily toggled at the top)
STOP_LOSS_FACTOR      = 0.96    # 4% stop-loss
TARGET_FACTOR         = 1.0267  # 2.67% profit target
MONITOR_END_HOUR      = 15
MONITOR_END_MINUTE    = 33
SURPRISE_THRESHOLD    = 50      # minimum EPS surprise (%) to consider
MC_THRESHOLD          = 10_000  # max market cap (in millions USD) to consider

# Price change filter: (open_price / prev_close) - 1 must be between MIN and MAX
MIN_PRICE_PCT_CHANGE  = -0.01   # −1% (i.e., open ≥ 99% of prev_close)
MAX_PRICE_PCT_CHANGE  =  0.20   # +20% (i.e., open ≤ 120% of prev_close)


# ─── Logging Configuration ─────────────────────────────────────────────────────
def configure_logging() -> logging.Logger:
    logger = logging.getLogger("earnings_trader")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S %Z")

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
    """Poll Alpaca clock every few minutes until market opens."""
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

        if secs_to_open > 300:
            logger.info("Market closed; sleeping 5m until ~%s", next_open)
            time.sleep(300)
        else:
            logger.info("Market opens in ~%.1f seconds at %s; sleeping until then",
                        secs_to_open, next_open)
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
        logger.info(f"Fetched {len(entries)} earnings-calendar entries")
        return entries
    except Exception as e:
        logger.exception("Failed to fetch earnings calendar: %s", e)
        sys.exit(1)


def filter_candidates(entries: List[dict], fh_client: finnhub.Client,
                      start_str: str, end_str: str) -> List[dict]:
    candidates = []
    for e in entries:
        sym = e.get("symbol")
        est = e.get("epsEstimate")
        act = e.get("epsActual")
        if est is None or act is None or est == 0:
            continue

        surprise = (act - est) / abs(est) * 100
        if surprise <= SURPRISE_THRESHOLD or not is_within_earnings_window(e, start_str, end_str):
            continue

        try:
            prof = fh_client.company_profile2(symbol=sym)
            mc = prof.get("marketCapitalization", 0)
        except Exception as exc:
            logger.warning("Could not fetch profile for %s: %s", sym, exc)
            continue

        if mc <= MC_THRESHOLD:
            candidates.append({"symbol": sym, "surprise": surprise})
            logger.info(f"Candidate {sym}: surprise={surprise:.2f}% mc={mc}M")

    candidates.sort(key=lambda x: x["surprise"], reverse=True)
    logger.info(f"Total filtered candidates: {len(candidates)}")
    return candidates


def place_buy_order(api: AlpacaREST, filtered: List[dict]) -> Optional[dict]:
    """
    Attempt to buy the top candidates in descending surprise order.
    Returns a buy_info dict on success, or None if all fail.
    """
    for entry in filtered:
        sym      = entry["symbol"]
        surprise = entry["surprise"]

        # 1) Retrieve today’s 1-minute bar at open (open_price)
        try:
            bars_1min = api.get_bars(
                sym,
                timeframe="1Min",
                limit=1,
                adjustment="raw"
            )
            if not bars_1min:
                logger.error("No 1-minute bar for %s at open", sym)
                continue
            open_price = bars_1min[0].o
            logger.info(f"Open price for {sym}: {open_price:.2f}")
        except Exception as exc:
            logger.warning(f"Error fetching 1-min bar for {sym}: {exc}")
            continue

        # 2) Fetch previous day's close and enforce MIN/MAX price-change filter
        try:
            bars_daily = api.get_bars(
                sym,
                timeframe="1Day",
                limit=2,
                adjustment="raw"
            )
            if len(bars_daily) < 2:
                logger.warning("Not enough daily bars to compute previous close for %s", sym)
                continue

            prev_close = bars_daily[-2].c
            pct_change = (open_price / prev_close) - 1.0
            logger.info(f"{sym}: prev_close={prev_close:.2f}, open={open_price:.2f}, "
                        f"pct_change={pct_change*100:.2f}%")
        except Exception as exc:
            logger.warning(f"Error fetching daily bars for {sym}: {exc}")
            continue

        if pct_change < MIN_PRICE_PCT_CHANGE or pct_change > MAX_PRICE_PCT_CHANGE:
            logger.info(
                f"Skipping {sym}: price move {pct_change*100:.2f}% ∉ "
                f"[{MIN_PRICE_PCT_CHANGE*100:.2f}%, {MAX_PRICE_PCT_CHANGE*100:.2f}%]"
            )
            continue

        # 3) Compute how many shares we can afford using all available cash
        try:
            acct = api.get_account()
            cash_available = float(acct.cash)
            max_shares     = int(cash_available // open_price)
            logger.debug(f"cash_available=${cash_available:.2f} → max_shares={max_shares}")
        except Exception as exc:
            logger.exception(f"Failed to fetch account data for {sym}: {exc}")
            continue

        if max_shares < 1:
            logger.error(f"Insufficient cash to buy even one share of {sym}")
            continue

        # 4) “Shave one share at a time” loop
        order_qty = max_shares
        while order_qty > 0:
            try:
                order = api.submit_order(
                    symbol=sym,
                    qty=order_qty,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                logger.info(f"SUBMITTED BUY {sym} | qty={order_qty} | order_id={order.id}")
                return {
                    "symbol":    sym,
                    "buy_time":  datetime.now(TZ).replace(microsecond=0).isoformat(),
                    "buy_price": open_price,
                    "qty":       order_qty,
                    "order_id":  order.id
                }

            except APIError as e:
                err_msg = str(e).lower()
                logger.warning(f"Buy {sym} qty={order_qty} failed: {e}")
                if "insufficient buying power" in err_msg:
                    order_qty -= 1
                    logger.debug(f"Retrying {sym} with qty={order_qty}")
                    time.sleep(1)
                    continue
                else:
                    logger.exception(f"Unexpected APIError for {sym}: {e}")
                    break

            except Exception as e:
                logger.exception(f"Unexpected exception when buying {sym}: {e}")
                break

        # If we exit the while-loop without a return, move to next candidate
        logger.error(f"Could not place any buy order for {sym}, trying next candidate")
        time.sleep(1)

    return None  # no successful buy


def monitor_trade(api: AlpacaREST, buy_info: dict):
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
            trade = api.get_latest_trade(symbol)
            price = trade.price
            if price is None:
                time.sleep(1)
                continue

            now = datetime.now(TZ)

            # (1) Time-based exit
            if not hit_target and now >= cutoff_time:
                sell = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="day"
                )
                logger.info(f"TIME-EXIT SELL {symbol} | order_id={sell.id}")
                break

            # (2) Hard stop-loss
            if price <= buy_price * stop_loss_factor:
                sell = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="day"
                )
                logger.info(f"STOP-LOSS SELL {symbol} | order_id={sell.id}")
                break

            # (3) Initial profit target reached
            if not hit_target and price >= buy_price * target_factor:
                hit_target = True
                peak_price = price
                logger.info(f"TARGET HIT {symbol} | peak={peak_price:.2f}")

            # (4) Trailing stop once target is hit
            if hit_target:
                if price > peak_price:
                    peak_price = price
                    logger.debug(f"New peak for {symbol}: {peak_price:.2f}")
                elif price < peak_price:
                    sell = api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="sell",
                        type="market",
                        time_in_force="day"
                    )
                    logger.info(f"TRAILING-STOP SELL {symbol} | order_id={sell.id}")
                    break

        except APIError as e:
            logger.warning(f"APIError in monitoring loop for {symbol}: {e}")
            time.sleep(1)
            continue
        except Exception as e:
            logger.exception(f"Unexpected error in monitoring loop for {symbol}: {e}")
            time.sleep(1)
            continue

        time.sleep(1)

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
    start_str = prev_bday.isoformat()
    end_str   = today.isoformat()
    logger.info(f"Fetching earnings calendar from {start_str} to {end_str}")

    # 3) Fetch & filter candidates (AMC/BMO, surprise, market cap)
    entries  = fetch_earnings_calendar(fh_client, start_str, end_str)
    filtered = filter_candidates(entries, fh_client, start_str, end_str)
    if not filtered:
        logger.error("No candidates found; exiting")
        sys.exit(1)

    # 4) Wait for market open if necessary (ensures start-of-day buy happens right at open)
    try:
        clock = alpaca_api.get_clock()
    except Exception as e:
        logger.exception(f"Failed to fetch Alpaca clock: {e}")
        sys.exit(1)

    if not clock.is_open:
        sleep_until_market_open(alpaca_api, TZ)
    else:
        logger.info("Market already OPEN at %s", clock.timestamp.astimezone(TZ))

    # 5) Place buy order (first candidate that meets price-change filter & affordability)
    buy_info = place_buy_order(alpaca_api, filtered)
    if not buy_info:
        logger.error("No purchase succeeded; exiting")
        sys.exit(1)

    # 6) Monitor the trade and potentially sell (stop-loss, target, trailing stop, time-exit)
    monitor_trade(alpaca_api, buy_info)


if __name__ == "__main__":
    main()






