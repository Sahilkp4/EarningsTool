import os
import sys
import time
import logging
import asyncio
import json
import requests
import re
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
#fix these comments when running on ubuntu fixed report
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

def get_cutoff_time(today: date) -> datetime:
    cutoff = datetime(
        today.year, today.month, today.day,
        MONITOR_END_HOUR, MONITOR_END_MINUTE, tzinfo=TZ
    )
    logger.debug(f"Monitor cutoff time set to {cutoff.time()}")
    return cutoff

# ─── Daily Tickers Input Parsing ─────────────────────────────────────────────
def load_tickers_from_dailytickers(target_date: date) -> List[Dict[str, float]]:
    """
    Load tickers from dailytickers.txt for the specified date.
    Only processes log entries from the target date.
    
    Args:
        target_date: The date to look for ticker entries
        
    Returns:
        List of dictionaries with 'symbol' and 'surprise' keys
    """
    dailytickers_file = os.path.join(os.path.dirname(__file__), "dailytickers.txt")
    
    if not os.path.exists(dailytickers_file):
        logger.error(f"dailytickers.txt not found at {dailytickers_file}")
        return []
    
    logger.info(f"Loading tickers from dailytickers.txt for date {target_date}")
    
    try:
        with open(dailytickers_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into lines (handles both \n and \r\n)
        lines = content.splitlines()
        
        # Look for lines from the target date
        target_date_str = target_date.strftime("%Y-%m-%d")
        logger.debug(f"Looking for entries with date: {target_date_str}")
        
        candidates = []
        matching_lines_found = 0
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with target date
            if not line.startswith(target_date_str):
                continue
                
            # Look for the "Sorted candidates by surprise:" pattern
            if "Sorted candidates by surprise:" not in line:
                logger.debug(f"Line {line_num} has target date but no 'Sorted candidates by surprise:' pattern")
                continue
                
            matching_lines_found += 1
            logger.debug(f"Found matching line {line_num} (length: {len(line)} chars)")
            
            # Extract the ticker portion after "Sorted candidates by surprise:"
            pattern = r"Sorted candidates by surprise:\s*(.+)"
            match = re.search(pattern, line)
            
            if not match:
                logger.warning(f"Could not extract ticker data from line {line_num}: {line[:100]}...")
                continue
            
            ticker_string = match.group(1).strip()
            logger.debug(f"Extracted ticker string (length: {len(ticker_string)} chars): {ticker_string[:100]}...")
            
            # Parse individual tickers and surprises
            # Pattern: TICKER(XX.X%) - handles tickers with letters only
            ticker_pattern = r"([A-Z]+)\((\d+(?:\.\d+)?)%\)"
            matches = re.findall(ticker_pattern, ticker_string)
            
            if not matches:
                logger.warning(f"No ticker patterns found in line {line_num}")
                continue
            
            # Process each ticker found in this line
            line_candidates = []
            for symbol, surprise_str in matches:
                try:
                    surprise = float(surprise_str)
                    line_candidates.append({"symbol": symbol, "surprise": surprise})
                    logger.debug(f"Parsed {symbol}: surprise={surprise}%")
                except ValueError as e:
                    logger.warning(f"Could not parse surprise for {symbol}: {surprise_str} - {e}")
                    continue
            
            # Add to main candidates list
            candidates.extend(line_candidates)
            logger.info(f"Added {len(line_candidates)} candidates from line {line_num}")
        
        if matching_lines_found == 0:
            logger.warning(f"No lines found with date {target_date_str} and 'Sorted candidates by surprise:' pattern")
            return []
        
        if not candidates:
            logger.warning(f"No ticker entries could be parsed for date {target_date_str}")
            return []
        
        # Remove duplicates while preserving order (in case same ticker appears multiple times)
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            symbol = candidate['symbol']
            if symbol not in seen:
                seen.add(symbol)
                unique_candidates.append(candidate)
            else:
                logger.debug(f"Duplicate ticker {symbol} removed")
        
        logger.info(f"Successfully loaded {len(unique_candidates)} unique tickers for {target_date_str} from {matching_lines_found} log entries")
        return unique_candidates
        
    except Exception as e:
        logger.error(f"Error reading dailytickers.txt: {e}")
        return []

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

# ─── Market Data Integration (Alpaca + Finnhub Fallback) ──────────────────────
def get_alpaca_quote_data(api: AlpacaREST, symbol: str, today: str) -> Dict:
    """
    Fetch current day OHLC data from Alpaca for a given symbol.
    Only includes regular market hours (9:30 AM - 4:00 PM ET).
    Now includes timestamps for high and low prices, and closing price.
    
    Args:
        api: Alpaca REST client
        symbol: Stock symbol
        today: Today's date in YYYY-MM-DD format
    
    Returns:
        Dict with OHLC data, high/low times, and closing price or empty dict if no data found
    """
    try:
        logger.debug(f"Fetching Alpaca intraday data for {symbol} (regular market hours only)")
        
        # Regular market hours: 9:30 AM - 4:00 PM ET
        start_iso = f"{today}T09:30:00-04:00"  # Market open
        end_iso = f"{today}T16:00:00-04:00"    # Market close
        
        bars = api.get_bars(
            symbol,
            timeframe="1Min",
            start=start_iso,
            end=end_iso,
            limit=500,  # ~390 minutes in trading day, plus buffer
            adjustment="raw"
        )
        
        if not bars:
            logger.warning(f"No Alpaca bars found for {symbol} on {today} during market hours")
            return {}
        
        # Filter bars to ensure they're within market hours (double-check)
        market_bars = []
        for bar in bars:
            bar_time = bar.t.astimezone(TZ)
            bar_hour = bar_time.hour
            bar_minute = bar_time.minute
            
            # Include bars from 9:30 AM to 3:59 PM ET
            if (bar_hour == 9 and bar_minute >= 30) or (10 <= bar_hour <= 15):
                market_bars.append(bar)
            elif bar_hour == 16 and bar_minute == 0:  # Include 4:00 PM close
                market_bars.append(bar)
        
        if not market_bars:
            logger.warning(f"No bars found during market hours for {symbol}")
            return {}
        
        # Initialize tracking variables
        open_price = market_bars[0].o
        high_price = market_bars[0].h
        low_price = market_bars[0].l
        close_price = market_bars[-1].c  # Last available close price
        high_time = market_bars[0].t.astimezone(TZ)
        low_time = market_bars[0].t.astimezone(TZ)
        
        # Find the actual high and low prices with their timestamps
        for bar in market_bars:
            bar_time = bar.t.astimezone(TZ)
            
            # Check for new high
            if bar.h > high_price:
                high_price = bar.h
                high_time = bar_time
            
            # Check for new low
            if bar.l < low_price:
                low_price = bar.l
                low_time = bar_time
        
        result = {
            'open_price': open_price,
            'high_price': high_price,
            'low_price': low_price,
            'close_price': close_price,
            'high_time': high_time.strftime('%H:%M'),
            'low_time': low_time.strftime('%H:%M'),
            'bar_count': len(market_bars),
            'total_bars': len(bars),
            'time_range': f"{market_bars[0].t.astimezone(TZ).strftime('%H:%M')} - {market_bars[-1].t.astimezone(TZ).strftime('%H:%M')}"
        }
        
        logger.debug(f"Alpaca market hours OHLC for {symbol}: O=${result['open_price']:.2f}, H=${result['high_price']:.2f}@{result['high_time']}, L=${result['low_price']:.2f}@{result['low_time']}, C=${result['close_price']:.2f}")
        logger.debug(f"  {result['bar_count']} market bars ({result['time_range']}) from {result['total_bars']} total bars")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching Alpaca data for {symbol}: {e}")
        return {}

def get_finnhub_quote_data(fh: finnhub.Client, symbol: str) -> Dict:
    """
    Fetch quote data from Finnhub for a given symbol as fallback.
    Uses the quote endpoint which should be available on free tier.
    Note: Finnhub doesn't provide intraday timestamps for high/low prices.
    
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
                'close_price': quote.get('c', 0),    # Current close price
                'timestamp': quote.get('t', 0),      # Timestamp
                'high_time': 'N/A',                  # Finnhub doesn't provide intraday timestamps
                'low_time': 'N/A'                    # Finnhub doesn't provide intraday timestamps
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

def generate_earnings_report(candidates: List[Dict], prev_closes: Dict[str, float], api: AlpacaREST, fh: finnhub.Client, today: str) -> str:
    """
    Generate comprehensive earnings report with OHLC data from Alpaca (primary) and Finnhub (fallback).
    Now includes high/low times and closing prices.
    Results are ordered by %change high (descending).
    """
    logger.info(f"Generating earnings report for {len(candidates)} candidates using Alpaca + Finnhub data")
    
    report_lines = []
    report_lines.append("=" * 130)
    report_lines.append(f"EARNINGS REPORT (Alpaca + Finnhub) - {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    report_lines.append("=" * 130)
    report_lines.append("")
    
    # Collect all data first for sorting
    report_data = []
    failed_reports = 0
    alpaca_success = 0
    finnhub_fallback = 0
    last_finnhub_call = 0.0
    
    for candidate in candidates:
        symbol = candidate['symbol']
        surprise = candidate['surprise']
        
        # Get previous close from our existing data (Alpaca)
        alpaca_prev_close = prev_closes.get(symbol, 0.0)
        
        # Try Alpaca first for OHLC data
        quote_data = get_alpaca_quote_data(api, symbol, today)
        data_source = "Alpaca"
        
        if quote_data:
            alpaca_success += 1
            logger.debug(f"Using Alpaca data for {symbol}")
        else:
            # Fallback to Finnhub
            logger.debug(f"Alpaca failed for {symbol}, trying Finnhub fallback")
            
            # Rate limiting for Finnhub free tier (60 calls/min)
            now = time.monotonic()
            if now - last_finnhub_call < FINNHUB_CALL_INTERVAL:
                pause = FINNHUB_CALL_INTERVAL - (now - last_finnhub_call)
                logger.debug(f"Rate limiting: sleeping {pause:.2f}s before Finnhub call for {symbol}")
                time.sleep(pause)
            
            quote_data = get_finnhub_quote_data(fh, symbol)
            last_finnhub_call = time.monotonic()
            data_source = "Finnhub"
            
            if quote_data:
                finnhub_fallback += 1
        
        if not quote_data:
            logger.warning(f"No data available for {symbol} from either source")
            failed_reports += 1
            # Add entry with N/A values and -999 as sort key (will appear at bottom)
            report_data.append({
                'symbol': symbol,
                'surprise': surprise,
                'alpaca_prev_close': alpaca_prev_close,
                'data_source': 'N/A',
                'open_price': 'N/A',
                'pct_change_open': 'N/A',
                'high_price': 'N/A',
                'high_time': 'N/A',
                'low_price': 'N/A',
                'low_time': 'N/A',
                'close_price': 'N/A',
                'pct_change_high': 'N/A',
                'pct_change_low': 'N/A',
                'pct_change_close': 'N/A',
                'sort_key': -999  # Will sort to bottom
            })
            continue
        
        # Extract quote data
        open_price = quote_data['open_price']
        high_price = quote_data['high_price']
        low_price = quote_data['low_price']
        close_price = quote_data['close_price']
        high_time = quote_data.get('high_time', 'N/A')
        low_time = quote_data.get('low_time', 'N/A')
        
        # For previous close, prefer Finnhub if available and from Finnhub source, otherwise use Alpaca
        if data_source == "Finnhub" and quote_data.get('prev_close', 0) > 0:
            reference_prev_close = quote_data['prev_close']
        else:
            reference_prev_close = alpaca_prev_close
        
        # Calculate percentage changes
        pct_change_open = calculate_percentage_change(reference_prev_close, open_price) if reference_prev_close > 0 else 0.0
        pct_change_high = calculate_percentage_change(open_price, high_price) if open_price > 0 else 0.0
        pct_change_low = calculate_percentage_change(open_price, low_price) if open_price > 0 else 0.0
        pct_change_close = calculate_percentage_change(reference_prev_close, close_price) if reference_prev_close > 0 else 0.0
        
        # Add to report data with sort key
        report_data.append({
            'symbol': symbol,
            'surprise': surprise,
            'alpaca_prev_close': alpaca_prev_close,
            'data_source': data_source,
            'reference_prev_close': reference_prev_close,
            'open_price': open_price,
            'pct_change_open': pct_change_open,
            'high_price': high_price,
            'high_time': high_time,
            'low_price': low_price,
            'low_time': low_time,
            'close_price': close_price,
            'pct_change_high': pct_change_high,
            'pct_change_low': pct_change_low,
            'pct_change_close': pct_change_close,
            'sort_key': pct_change_high if isinstance(pct_change_high, (int, float)) else -999
        })
    
    # Sort by %change high (descending) - highest gains first
    report_data.sort(key=lambda x: x['sort_key'], reverse=True)
    
    # Generate report header - extended to include new columns
    header = f"{'Symbol':<8} {'Surprise':<8} {'Src':<7} {'PrevCls':<8} {'Open':<8} {'%Open':<7} {'High':<8} {'HTime':<6} {'Low':<8} {'LTime':<6} {'Close':<8} {'%High':<7} {'%Low':<7} {'%Close':<7}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    # Generate sorted report lines
    successful_reports = 0
    for data in report_data:
        if data['sort_key'] != -999:
            successful_reports += 1
            line = f"{data['symbol']:<8} {data['surprise']:<8.2f} {data['data_source']:<7} {data['reference_prev_close']:<8.2f} {data['open_price']:<8.2f} {data['pct_change_open']:<7.2f} {data['high_price']:<8.2f} {data['high_time']:<6} {data['low_price']:<8.2f} {data['low_time']:<6} {data['close_price']:<8.2f} {data['pct_change_high']:<7.2f} {data['pct_change_low']:<7.2f} {data['pct_change_close']:<7.2f}"
        else:
            line = f"{data['symbol']:<8} {data['surprise']:<8.2f} {data['data_source']:<7} {data['alpaca_prev_close']:<8.2f} {'N/A':<8} {'N/A':<7} {'N/A':<8} {'N/A':<6} {'N/A':<8} {'N/A':<6} {'N/A':<8} {'N/A':<7} {'N/A':<7} {'N/A':<7}"
        
        report_lines.append(line)
    
    # Summary
    report_lines.append("")
    report_lines.append("-" * 130)
    report_lines.append(f"Report Summary:")
    report_lines.append(f"  Total candidates: {len(candidates)}")
    report_lines.append(f"  Successful reports: {successful_reports}")
    report_lines.append(f"  Failed reports: {failed_reports}")
    report_lines.append(f"  Alpaca primary source: {alpaca_success}")
    report_lines.append(f"  Finnhub fallback used: {finnhub_fallback}")
    report_lines.append("")
    report_lines.append("=" * 130)
    
    report = "\n".join(report_lines)
    logger.info(f"Generated earnings report with {successful_reports} successful entries")
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
    logger.info("Starting earnings report generation with dailytickers.txt input")
    
    # Initialize clients
    fh = finnhub.Client(api_key=FINNHUB_API_KEY)
    api = AlpacaREST(
        key_id=APCA_KEY,
        secret_key=APCA_SECRET,
        base_url=ALPACA_BASE_URL
    )
    
    # Get today's date and load tickers from dailytickers.txt
    today = date.today()
    logger.info(f"Looking for tickers from dailytickers.txt for date: {today}")
    
    candidates = load_tickers_from_dailytickers(today)
    if not candidates:
        logger.warning(f"No candidates loaded for {today}, exiting")
        print(f"No valid tickers found for {today} in dailytickers.txt. Exiting.")
        return
    
    logger.info(f"Loaded {len(candidates)} tickers for {today}:")
    for candidate in candidates:
        logger.debug(f"  {candidate['symbol']}: {candidate['surprise']:.1f}%")
    
    # Set up date range for previous closes
    yesterday = get_prev_business_day(today)
    frm = yesterday.strftime("%Y-%m-%d")
    
    logger.info(f"Using {frm} for previous close data")
    
    # Get previous closes
    logger.info(f"Fetching previous close data for {frm}...")
    prev_closes = preload_prev_closes(api, candidates, frm)
    
    # Generate report with Alpaca + Finnhub data
    logger.info("Generating earnings report...")
    report = generate_earnings_report(candidates, prev_closes, api, fh, today.strftime("%Y-%m-%d"))
    
    # Save report
    save_earnings_report(report)
    
    # Also print to console (for cron logging)
    print("\n" + report)
    
    logger.info(f"Earnings report generation completed for {today} with {len(candidates)} tickers from dailytickers.txt")

if __name__ == "__main__":
    main()