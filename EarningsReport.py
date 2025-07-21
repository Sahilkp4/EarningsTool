import os
import sys
import time
import logging
import asyncio
import json
import requests
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
#fix these comments when running on ubuntu aaabbccdd
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
# NOTE: Market cap filtering removed since company_profile2 is not available on free tier
MIN_PCT_INCREASE   = 2.84  # Minimum threshold for % increase at open
TRAIL_PERCENT      = 0.1  # 0.1%- very tight trailing stop

ALPACA_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"  # or v2/crypto or v2/stocks, adjust as needed
ALPACA_MAX_SUBSCRIBE = 30
ALPACA_SUBSCRIBE_WAIT = 5  # seconds wait per batch before switching

# Finnhub rate limiting - free tier allows 60 calls/min
FINNHUB_RATE_LIMIT = 60  # calls per minute
FINNHUB_CALL_INTERVAL = 60.0 / FINNHUB_RATE_LIMIT  # seconds between calls

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
    filtered_count = {
        'missing_data': 0,
        'poor_surprise': 0,
        'wrong_window': 0,
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

        # Note: Market cap filtering removed since company_profile2 is not available on free tier
        candidates.append({"symbol": sym, "surprise": surprise})
        logger.info(f"✓ Candidate {sym}: surprise={surprise:.2f}%")
        filtered_count['passed'] += 1

    logger.info(f"Filtering results: {filtered_count}")
    logger.info(f"Total candidates after filter: {len(candidates)}")
    return candidates

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

# ─── Finnhub Quote Integration ───────────────────────────────────────────────
def get_finnhub_quote_data(fh: finnhub.Client, symbol: str) -> Dict:
    """
    Fetch quote data from Finnhub for a given symbol.
    Uses the quote endpoint which should be available on free tier.
    
    Args:
        fh: Finnhub client
        symbol: Stock symbol
    
    Returns:
        Dict with quote data or empty dict if no data found
    """
    try:
        logger.debug(f"Fetching Finnhub quote data for {symbol}")
        
        # Fetch quote
        quote = fh.quote(symbol)
        
        if quote and 'c' in quote:
            result = {
                'open_price': quote.get('o', 0),     # Open price of the day
                'high_price': quote.get('h', 0),     # High price of the day
                'low_price': quote.get('l', 0),      # Low price of the day
                'prev_close': quote.get('pc', 0),    # Previous close price
                'timestamp': quote.get('t', 0)       # Timestamp
            }
            
            logger.debug(f"Finnhub quote data for {symbol}: {result}")
            return result
        
        logger.warning(f"No quote data found for {symbol}")
        return {}
        
    except Exception as e:
        logger.error(f"Error fetching Finnhub quote data for {symbol}: {e}")
        return {}

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change from old_value to new_value."""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def generate_earnings_report(candidates: List[Dict], prev_closes: Dict[str, float], fh: finnhub.Client) -> str:
    """
    Generate comprehensive earnings report with quote data from Finnhub.
    Results are ordered by %change high (descending).
    """
    logger.info(f"Generating earnings report for {len(candidates)} candidates using Finnhub quote data")
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append(f"EARNINGS REPORT (Finnhub Quote) - {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    # Collect all data first for sorting
    report_data = []
    failed_reports = 0
    last_call_time = 0.0
    
    for candidate in candidates:
        symbol = candidate['symbol']
        surprise = candidate['surprise']
        
        # Get previous close from our existing data (Alpaca)
        alpaca_prev_close = prev_closes.get(symbol, 0.0)
        
        # Rate limiting for Finnhub free tier (60 calls/min)
        now = time.monotonic()
        if now - last_call_time < FINNHUB_CALL_INTERVAL:
            pause = FINNHUB_CALL_INTERVAL - (now - last_call_time)
            logger.debug(f"Rate limiting: sleeping {pause:.2f}s before Finnhub call for {symbol}")
            time.sleep(pause)
        
        # Get Finnhub quote data
        quote_data = get_finnhub_quote_data(fh, symbol)
        last_call_time = time.monotonic()
        
        if not quote_data:
            logger.warning(f"No Finnhub quote data available for {symbol}")
            failed_reports += 1
            # Add entry with N/A values and -999 as sort key (will appear at bottom)
            report_data.append({
                'symbol': symbol,
                'surprise': surprise,
                'alpaca_prev_close': alpaca_prev_close,
                'finnhub_prev_close': 'N/A',
                'open_price': 'N/A',
                'pct_change_open': 'N/A',
                'high_price': 'N/A',
                'low_price': 'N/A',
                'pct_change_high': 'N/A',
                'pct_change_low': 'N/A',
                'sort_key': -999  # Will sort to bottom
            })
            continue
        
        # Extract quote data
        open_price = quote_data['open_price']
        high_price = quote_data['high_price']
        low_price = quote_data['low_price']
        finnhub_prev_close = quote_data['prev_close']
        
        # Use Finnhub's previous close if available, otherwise use Alpaca's
        reference_prev_close = finnhub_prev_close if finnhub_prev_close > 0 else alpaca_prev_close
        
        # Calculate percentage changes
        pct_change_open = calculate_percentage_change(reference_prev_close, open_price) if reference_prev_close > 0 else 0.0
        pct_change_high = calculate_percentage_change(open_price, high_price) if open_price > 0 else 0.0
        pct_change_low = calculate_percentage_change(open_price, low_price) if open_price > 0 else 0.0
        
        # Add to report data with sort key
        report_data.append({
            'symbol': symbol,
            'surprise': surprise,
            'alpaca_prev_close': alpaca_prev_close,
            'finnhub_prev_close': finnhub_prev_close,
            'open_price': open_price,
            'pct_change_open': pct_change_open,
            'high_price': high_price,
            'low_price': low_price,
            'pct_change_high': pct_change_high,
            'pct_change_low': pct_change_low,
            'sort_key': pct_change_high if isinstance(pct_change_high, (int, float)) else -999
        })
    
    # Sort by %change high (descending) - highest gains first
    report_data.sort(key=lambda x: x['sort_key'], reverse=True)
    
    # Generate report header
    header = f"{'Symbol':<8} {'Surprise':<8} {'PrevClose':<10} {'Open':<10} {'%ChgOpen':<10} {'High':<10} {'Low':<10} {'%ChgHigh':<10} {'%ChgLow':<10}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    # Generate sorted report lines
    successful_reports = 0
    for data in report_data:
        if data['sort_key'] != -999:
            successful_reports += 1
            # Use Finnhub prev close if available, otherwise use Alpaca
            display_prev_close = data['finnhub_prev_close'] if data['finnhub_prev_close'] > 0 else data['alpaca_prev_close']
            line = f"{data['symbol']:<8} {data['surprise']:<8.2f} {display_prev_close:<10.2f} {data['open_price']:<10.2f} {data['pct_change_open']:<10.2f} {data['high_price']:<10.2f} {data['low_price']:<10.2f} {data['pct_change_high']:<10.2f} {data['pct_change_low']:<10.2f}"
        else:
            line = f"{data['symbol']:<8} {data['surprise']:<8.2f} {data['alpaca_prev_close']:<10.2f} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}"
        
        report_lines.append(line)
    
    # Summary
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append(f"Report Summary:")
    report_lines.append(f"  Total candidates: {len(candidates)}")
    report_lines.append(f"  Successful reports: {successful_reports}")
    report_lines.append(f"  Failed reports: {failed_reports}")
    report_lines.append(f"  Data source: Finnhub Quote API (Free Tier)")
    report_lines.append(f"  Rate limit: {FINNHUB_RATE_LIMIT} calls/min")
    report_lines.append("")
    report_lines.append("Results ordered by %Change High (highest to lowest)")
    report_lines.append("NOTE: Using real-time quote data instead of historical OHLC")
    report_lines.append("NOTE: Market cap filtering disabled (company_profile2 not available on free tier)")
    report_lines.append("=" * 100)
    
    report = "\n".join(report_lines)
    logger.info(f"Generated earnings report with {successful_reports} successful entries using Finnhub quote data")
    return report

def save_earnings_report(report: str, filename: str = "earningsreport.txt"):
    """Save the earnings report to a file, appending if file exists."""
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(report + "\n\n")
        logger.info(f"Earnings report saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save earnings report to {filename}: {e}")

def main():
    """Main function to run the earnings report generation."""
    logger.info("Starting earnings report generation with Finnhub OHLC data")
    
    # Initialize clients
    fh = finnhub.Client(api_key=FINNHUB_API_KEY)
    api = AlpacaREST(
        key_id=APCA_KEY,
        secret_key=APCA_SECRET,
        base_url=ALPACA_BASE_URL
    )
    
    # Set up date range
    today = date.today()
    yesterday = get_prev_business_day(today)
    
    frm = yesterday.strftime("%Y-%m-%d")
    to = today.strftime("%Y-%m-%d")
    
    logger.info(f"Analyzing earnings from {frm} to {to}")
    
    # Fetch earnings calendar
    entries = fetch_earnings_calendar(fh, frm, to)
    if not entries:
        logger.warning("No earnings entries found, exiting")
        return
    
    # Filter candidates (note: market cap filtering removed)
    candidates = filter_candidates(entries, fh, frm, to)
    if not candidates:
        logger.warning("No candidates found after filtering, exiting")
        return
    
    # Get previous closes
    prev_closes = preload_prev_closes(api, candidates, frm)
    
    # Generate report with Finnhub quote data
    report = generate_earnings_report(candidates, prev_closes, fh)
    
    # Save report
    save_earnings_report(report)
    
    # Also print to console
    print(report)
    
    logger.info("Earnings report generation completed using Finnhub OHLC data")

if __name__ == "__main__":
    main()