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
MAX_SURPRISE       = 600 # max suprise
MC_THRESHOLD       = 900_000_000_000_000 # market cap threshold: 900,000,000 million = 900 trillion
TRAIL_PERCENT      = 0.1  # 0.1%- very tight trailing stop

ALPACA_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"  # or v2/crypto or v2/stocks, adjust as needed
ALPACA_MAX_SUBSCRIBE = 30
ALPACA_SUBSCRIBE_WAIT = 5  # seconds wait per batch before switching

# Finnhub rate limiting - free tier allows 60 calls/min
FINNHUB_RATE_LIMIT = 60  # calls per minute
FINNHUB_CALL_INTERVAL = 60.0 / FINNHUB_RATE_LIMIT  # seconds between calls

# Pre-market hours (4:00 AM - 9:30 AM ET)
PREMARKET_START_HOUR = 4
PREMARKET_END_HOUR = 9
PREMARKET_END_MINUTE = 30

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

def is_within_earnings_window(entry: dict, start: str, end: str) -> bool:
    ok = ((entry.get("date") == start and entry.get("hour") == "amc") or
          (entry.get("date") == end   and entry.get("hour") == "bmo"))
    logger.debug(f"{entry.get('symbol')} window check: {ok} (date={entry.get('date')}, hour={entry.get('hour')})")
    return ok

# ─── Earnings Calendar & Candidate Filtering ─────────────────────────────────
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
            logger.debug(f"Skipping {sym}: MC {mc:.1f}M > {MC_THRESHOLD/1_000_000:.0f}M")
            filtered_count['high_mc'] += 1

    logger.info(f"Filtering results: {filtered_count}")
    logger.info(f"Total candidates after filter: {len(candidates)}")
    return candidates

def group_candidates_by_surprise(
    cands: List[Dict[str, float]]
) -> List[Dict[str, float]]:
    logger.debug(f"Sorting {len(cands)} candidates by surprise")
    sorted_c = sorted(cands, key=lambda x: x["surprise"], reverse=True)
    
    # Log message showing sorted candidates
    message = "Sorted candidates by surprise: " + \
              ", ".join(f"{c['symbol']}({c['surprise']:.1f}%)" for c in sorted_c)
    logger.info(message)
    
    return sorted_c

def generate_daily_tickers(fh: finnhub.Client, target_date: date) -> List[Dict[str, float]]:
    """
    Generate daily tickers by fetching and filtering earnings calendar data.
    
    Args:
        fh: Finnhub client
        target_date: The date to generate tickers for
        
    Returns:
        List of dictionaries with 'symbol' and 'surprise' keys
    """
    logger.info(f"Generating daily tickers for {target_date}")
    
    # Set date range for earnings calendar
    yesterday = get_prev_business_day(target_date)
    frm = yesterday.strftime("%Y-%m-%d")
    to = target_date.strftime("%Y-%m-%d")
    
    logger.info(f"Fetching earnings calendar from {frm} to {to}")
    
    # Fetch earnings calendar
    entries = fetch_earnings_calendar(fh, frm, to)
    if not entries:
        logger.warning(f"No earnings entries found for date range {frm} to {to}")
        return []
    
    # Filter candidates
    candidates = filter_candidates(entries, fh, frm, to)
    if not candidates:
        logger.warning(f"No candidates passed filtering for {target_date}")
        return []
    
    # Sort by surprise
    sorted_candidates = group_candidates_by_surprise(candidates)
    
    logger.info(f"Generated {len(sorted_candidates)} daily tickers for {target_date}")
    return sorted_candidates

def get_extended_stock_data(fh: finnhub.Client, symbol: str, api: AlpacaREST, today: str, last_finnhub_call: float) -> tuple:
    """
    Get extended stock data including market cap, sector, revenue surprise, and pre-market data.
    Returns tuple: (extended_data_dict, updated_last_finnhub_call_time)
    """
    extended_data = {
        'market_cap': 'N/A',
        'revenue_surprise': 'N/A', 
        'sector': 'N/A',
        'industry': 'N/A',
        'earnings_time': 'N/A',
        'premarket_high': 'N/A',
        'premarket_low': 'N/A',
        'premarket_volume': 'N/A',
        'opening_minute_volume': 'N/A',
        'volume_comparison': 'N/A',
        'pct_1min': 'N/A',
        'pct_5min': 'N/A', 
        'pct_15min': 'N/A',
        'pct_30min': 'N/A',
        'pct_1hr': 'N/A',
        'high_before_low': 'N/A'
    }
    
    # Rate limit for Finnhub calls
    def rate_limit_finnhub():
        nonlocal last_finnhub_call
        now = time.monotonic()
        if now - last_finnhub_call < FINNHUB_CALL_INTERVAL:
            pause = FINNHUB_CALL_INTERVAL - (now - last_finnhub_call)
            logger.debug(f"Rate limiting: sleeping {pause:.2f}s before Finnhub call")
            time.sleep(pause)
        last_finnhub_call = time.monotonic()
        return last_finnhub_call
    
    try:
        # Get company profile for market cap and sector info
        logger.debug(f"Fetching company profile for {symbol}")
        last_finnhub_call = rate_limit_finnhub()
        profile = fh.company_profile2(symbol=symbol)
        
        if profile:
            extended_data['market_cap'] = profile.get('marketCapitalization', 'N/A')
            # Finnhub free tier doesn't provide sector in company_profile2
            extended_data['sector'] = profile.get('finnhubIndustry', 'N/A')
            extended_data['industry'] = profile.get('finnhubIndustry', 'N/A')
            
    except Exception as e:
        logger.warning(f"Failed to fetch company profile for {symbol}: {e}")
    
    try:
        # Get earnings calendar entry for revenue surprise and timing
        yesterday = get_prev_business_day(date.fromisoformat(today))
        frm = yesterday.strftime("%Y-%m-%d")
        
        logger.debug(f"Fetching earnings calendar for {symbol}")
        last_finnhub_call = rate_limit_finnhub()
        cal = fh.earnings_calendar(symbol=symbol, _from=frm, to=today)
        
        earnings_entries = cal.get("earningsCalendar", [])
        for entry in earnings_entries:
            if entry.get('symbol') == symbol:
                # Revenue surprise calculation
                rev_est = entry.get('revenueEstimate')
                rev_act = entry.get('revenueActual') 
                if rev_est and rev_act and rev_est != 0:
                    extended_data['revenue_surprise'] = ((rev_act - rev_est) / abs(rev_est)) * 100
                
                # Earnings release time
                extended_data['earnings_time'] = entry.get('hour', 'N/A')
                break
                
    except Exception as e:
        logger.warning(f"Failed to fetch earnings calendar for {symbol}: {e}")
    
    try:
        # Get pre-market and intraday data from Alpaca
        premarket_data = get_premarket_data(api, symbol, today)
        extended_data.update(premarket_data)
        
        # Get intraday percentage changes and high/low timing
        intraday_data = get_intraday_analysis(api, symbol, today)
        extended_data.update(intraday_data)
        
    except Exception as e:
        logger.warning(f"Failed to fetch market data for {symbol}: {e}")
    
    return extended_data, last_finnhub_call

def get_premarket_data(api: AlpacaREST, symbol: str, today: str) -> dict:
    """Get pre-market high/low/volume data from Alpaca"""
    try:
        # Pre-market hours: 4:00 AM - 9:30 AM ET
        premarket_start = f"{today}T04:00:00-04:00"
        premarket_end = f"{today}T09:30:00-04:00"
        
        bars = api.get_bars(
            symbol,
            timeframe="1Min",
            start=premarket_start,
            end=premarket_end,
            limit=400,  # ~5.5 hours * 60 minutes
            adjustment="raw"
        )
        
        if not bars:
            return {'premarket_high': 'N/A', 'premarket_low': 'N/A', 'premarket_volume': 'N/A'}
        
        premarket_bars = []
        total_volume = 0
        
        for bar in bars:
            bar_time = bar.t.astimezone(TZ)
            bar_hour = bar_time.hour
            bar_minute = bar_time.minute
            
            # Include bars from 4:00 AM to 9:29 AM ET
            if PREMARKET_START_HOUR <= bar_hour < PREMARKET_END_HOUR or (bar_hour == PREMARKET_END_HOUR and bar_minute < PREMARKET_END_MINUTE):
                premarket_bars.append(bar)
                total_volume += bar.v
        
        if not premarket_bars:
            return {'premarket_high': 'N/A', 'premarket_low': 'N/A', 'premarket_volume': 'N/A'}
        
        premarket_high = max(bar.h for bar in premarket_bars)
        premarket_low = min(bar.l for bar in premarket_bars)
        
        return {
            'premarket_high': premarket_high,
            'premarket_low': premarket_low, 
            'premarket_volume': total_volume
        }
        
    except Exception as e:
        logger.warning(f"Failed to fetch pre-market data for {symbol}: {e}")
        return {'premarket_high': 'N/A', 'premarket_low': 'N/A', 'premarket_volume': 'N/A'}

def get_intraday_analysis(api: AlpacaREST, symbol: str, today: str) -> dict:
    """Get intraday percentage changes and opening minute volume analysis"""
    try:
        # Get market hours data
        market_start = f"{today}T09:30:00-04:00"
        market_end = f"{today}T16:00:00-04:00"
        
        bars = api.get_bars(
            symbol,
            timeframe="1Min", 
            start=market_start,
            end=market_end,
            limit=500,
            adjustment="raw"
        )
        
        if not bars:
            return {
                'opening_minute_volume': 'N/A',
                'volume_comparison': 'N/A',
                'pct_1min': 'N/A',
                'pct_5min': 'N/A',
                'pct_15min': 'N/A', 
                'pct_30min': 'N/A',
                'pct_1hr': 'N/A',
                'high_before_low': 'N/A'
            }
        
        # Filter to market hours only
        market_bars = []
        for bar in bars:
            bar_time = bar.t.astimezone(TZ)
            bar_hour = bar_time.hour
            bar_minute = bar_time.minute
            
            if (bar_hour == 9 and bar_minute >= 30) or (10 <= bar_hour <= 15) or (bar_hour == 16 and bar_minute == 0):
                market_bars.append(bar)
        
        if not market_bars:
            return {
                'opening_minute_volume': 'N/A',
                'volume_comparison': 'N/A', 
                'pct_1min': 'N/A',
                'pct_5min': 'N/A',
                'pct_15min': 'N/A',
                'pct_30min': 'N/A',
                'pct_1hr': 'N/A',
                'high_before_low': 'N/A'
            }
        
        open_price = market_bars[0].o
        opening_minute_volume = market_bars[0].v
        
        # Calculate percentage changes at different time intervals
        def get_price_at_minutes(minutes):
            if len(market_bars) > minutes:
                return market_bars[minutes].c
            return market_bars[-1].c
        
        pct_changes = {}
        intervals = [1, 5, 15, 30, 60]
        for interval in intervals:
            price = get_price_at_minutes(interval - 1)  # 0-indexed
            pct_changes[f'pct_{interval}min' if interval < 60 else 'pct_1hr'] = calculate_percentage_change(open_price, price)
        
        # Determine if high came before low
        high_price = max(bar.h for bar in market_bars)
        low_price = min(bar.l for bar in market_bars)
        
        high_time = None
        low_time = None
        
        for bar in market_bars:
            if bar.h == high_price and high_time is None:
                high_time = bar.t
            if bar.l == low_price and low_time is None:
                low_time = bar.t
        
        high_before_low = 1 if high_time and low_time and high_time < low_time else 0
        
        # Get average volume for comparison (simplified - using current volume as baseline)
        avg_volume = sum(bar.v for bar in market_bars[:10]) / min(10, len(market_bars)) if market_bars else 1
        volume_comparison = opening_minute_volume / avg_volume if avg_volume > 0 else 'N/A'
        
        return {
            'opening_minute_volume': opening_minute_volume,
            'volume_comparison': f"{volume_comparison:.2f}x" if isinstance(volume_comparison, (int, float)) else 'N/A',
            **pct_changes,
            'high_before_low': high_before_low
        }
        
    except Exception as e:
        logger.warning(f"Failed to fetch intraday analysis for {symbol}: {e}")
        return {
            'opening_minute_volume': 'N/A',
            'volume_comparison': 'N/A',
            'pct_1min': 'N/A', 
            'pct_5min': 'N/A',
            'pct_15min': 'N/A',
            'pct_30min': 'N/A',
            'pct_1hr': 'N/A',
            'high_before_low': 'N/A'
        }

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
    Now includes extended data: market cap, revenue surprise, sector, pre-market data, and intraday analysis.
    Results are ordered by %change high (descending).
    """
    logger.info(f"Generating comprehensive earnings report for {len(candidates)} candidates")
    
    report_lines = []
    report_lines.append("=" * 120)
    report_lines.append(f"EARNINGS REPORT - {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    report_lines.append("=" * 120)
    report_lines.append("")
    
    # Collect all data first for sorting
    report_data = []
    failed_reports = 0
    alpaca_success = 0
    finnhub_fallback = 0
    last_finnhub_call = 0.0
    
    for i, candidate in enumerate(candidates, 1):
        symbol = candidate['symbol']
        surprise = candidate['surprise']
        
        logger.info(f"Processing {i}/{len(candidates)}: {symbol}")
        
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
        
        # Get extended data regardless of quote_data success
        extended_data, last_finnhub_call = get_extended_stock_data(fh, symbol, api, today, last_finnhub_call)
        
        if not quote_data:
            logger.warning(f"No basic quote data available for {symbol} from either source")
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
                'sort_key': -999,  # Will sort to bottom
                **extended_data  # Include extended data even if basic quote failed
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
            'sort_key': pct_change_high if isinstance(pct_change_high, (int, float)) else -999,
            **extended_data  # Include all extended data
        })
    
    # Sort by %change high (descending) - highest gains first
    report_data.sort(key=lambda x: x['sort_key'], reverse=True)
    
    # Generate report with cleaner formatting
    successful_reports = 0
    for rank, data in enumerate(report_data, 1):
        if data['sort_key'] == -999:
            continue
            
        successful_reports += 1
        symbol = data['symbol']
        
        # Main header line with key metrics
        report_lines.append(f"#{rank:2d} {symbol:<6} | EPS:{data['surprise']:6.1f}% | Source:{data['data_source']:<7} | PrevClose:${data['reference_prev_close']:7.2f}")
        
        # Price action line
        if data['sort_key'] != -999:
            report_lines.append(f"    OHLC: ${data['open_price']:7.2f} ${data['high_price']:7.2f}@{data['high_time']} ${data['low_price']:7.2f}@{data['low_time']} ${data['close_price']:7.2f}")
            report_lines.append(f"    Chg%: Open:{data['pct_change_open']:+6.2f}% High:{data['pct_change_high']:+6.2f}% Low:{data['pct_change_low']:+6.2f}% Close:{data['pct_change_close']:+6.2f}%")
        else:
            report_lines.append(f"    OHLC: NO DATA AVAILABLE")
            report_lines.append(f"    Chg%: NO DATA AVAILABLE")
        
        # Extended data line
        mc_str = f"{data['market_cap']:.0f}M" if isinstance(data['market_cap'], (int, float)) else str(data['market_cap'])
        rev_str = f"{data['revenue_surprise']:+.1f}%" if isinstance(data['revenue_surprise'], (int, float)) else str(data['revenue_surprise'])
        sector_str = str(data['sector'])[:15] if data['sector'] != 'N/A' else 'N/A'
        
        report_lines.append(f"    Info: MCap:{mc_str:<8} Rev:{rev_str:<6} Sector:{sector_str:<15} Time:{data['earnings_time']}")
        
        # Pre-market data
        pm_high = f"${data['premarket_high']:.2f}" if isinstance(data['premarket_high'], (int, float)) else str(data['premarket_high'])
        pm_low = f"${data['premarket_low']:.2f}" if isinstance(data['premarket_low'], (int, float)) else str(data['premarket_low'])
        pm_vol = f"{data['premarket_volume']:,.0f}" if isinstance(data['premarket_volume'], (int, float)) else str(data['premarket_volume'])
        
        report_lines.append(f"    PM  : High:{pm_high:<8} Low:{pm_low:<8} Volume:{pm_vol}")
        
        # Intraday progression
        om_vol = f"{data['opening_minute_volume']:,.0f}" if isinstance(data['opening_minute_volume'], (int, float)) else str(data['opening_minute_volume'])
        vol_cmp = str(data['volume_comparison'])
        
        pct_1min = f"{data['pct_1min']:+.2f}%" if isinstance(data['pct_1min'], (int, float)) else str(data['pct_1min'])
        pct_5min = f"{data['pct_5min']:+.2f}%" if isinstance(data['pct_5min'], (int, float)) else str(data['pct_5min'])
        pct_15min = f"{data['pct_15min']:+.2f}%" if isinstance(data['pct_15min'], (int, float)) else str(data['pct_15min'])
        pct_30min = f"{data['pct_30min']:+.2f}%" if isinstance(data['pct_30min'], (int, float)) else str(data['pct_30min'])
        pct_1hr = f"{data['pct_1hr']:+.2f}%" if isinstance(data['pct_1hr'], (int, float)) else str(data['pct_1hr'])
        
        report_lines.append(f"    Vol : OpenMin:{om_vol:<10} Ratio:{vol_cmp:<8} H>L:{data['high_before_low']}")
        report_lines.append(f"    Time: 1m:{pct_1min:<7} 5m:{pct_5min:<7} 15m:{pct_15min:<7} 30m:{pct_30min:<7} 1h:{pct_1hr}")
        
        report_lines.append("")  # Separator line between stocks
    
    # Add failed reports at the end
    if failed_reports > 0:
        report_lines.append("FAILED REPORTS:")
        report_lines.append("-" * 40)
        for data in report_data:
            if data['sort_key'] == -999:
                report_lines.append(f"{data['symbol']:<6} | EPS:{data['surprise']:6.1f}% | NO PRICE DATA")
        report_lines.append("")
    
    # Summary
    report_lines.append("=" * 120)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 120)
    report_lines.append(f"Total candidates: {len(candidates)}")
    report_lines.append(f"Successful reports: {successful_reports}")
    report_lines.append(f"Failed reports: {failed_reports}")
    report_lines.append(f"Data sources - Alpaca: {alpaca_success}, Finnhub fallback: {finnhub_fallback}")
    report_lines.append("")
    report_lines.append("LEGEND:")
    report_lines.append("EPS%: Earnings Per Share surprise percentage")
    report_lines.append("Rev%: Revenue surprise percentage") 
    report_lines.append("MCap: Market capitalization in millions")
    report_lines.append("PM: Pre-market data (4:00-9:30 AM ET)")
    report_lines.append("Vol: Volume data and ratios")
    report_lines.append("Time: Price change percentages at intervals from market open")
    report_lines.append("H>L: 1 if daily high occurred before daily low, 0 otherwise")
    report_lines.append("=" * 120)
    
    report = "\n".join(report_lines)
    logger.info(f"Generated comprehensive earnings report with {successful_reports} successful entries")
    return report

def save_earnings_report(report: str, filename: str = "earningsreport.txt"):
    """Save the earnings report to a file, appending if file exists."""
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(report + "\n\n")
        logger.info(f"Earnings report saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save earnings report to {filename}: {e}")

def save_earnings_report_json(report_data: List[Dict], filename: str = "earnings_data.json"):
    """Save the earnings data as JSON, appending to existing data."""
    import json
    import os
    
    try:
        # Clean data for JSON serialization
        json_data = []
        for data in report_data:
            clean_data = {}
            for key, value in data.items():
                # Convert datetime objects to strings
                if hasattr(value, 'strftime'):
                    clean_data[key] = value.strftime('%H:%M')
                elif value == 'N/A' or (isinstance(value, str) and value.strip() == ''):
                    clean_data[key] = None
                else:
                    clean_data[key] = value
            json_data.append(clean_data)
        
        # Create new entry with timestamp
        new_entry = {
            'timestamp': datetime.now(TZ).isoformat(),
            'date': date.today().strftime('%Y-%m-%d'),
            'data': json_data
        }
        
        # Read existing data if file exists
        existing_data = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    file_content = f.read().strip()
                    if file_content:
                        existing_data = json.loads(file_content)
                        # Handle both single entry and array formats
                        if isinstance(existing_data, dict):
                            existing_data = [existing_data]
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning(f"Could not read existing JSON data from {filename}, starting fresh")
                existing_data = []
        
        # Append new data
        existing_data.append(new_entry)
        
        # Write back to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, default=str)
        
        logger.info(f"Earnings data appended to {filename} (total entries: {len(existing_data)})")
    except Exception as e:
        logger.error(f"Failed to save JSON data to {filename}: {e}")

def save_earnings_report_csv(report_data: List[Dict], filename: str = "earnings_data.csv"):
    """Save the earnings data as CSV, appending to existing data."""
    import csv
    import os
    
    try:
        if not report_data:
            return
        
        # Add date column to each record
        current_date = date.today().strftime('%Y-%m-%d')
        current_timestamp = datetime.now(TZ).isoformat()
        
        for data in report_data:
            data['report_date'] = current_date
            data['report_timestamp'] = current_timestamp
            
        # Get all possible field names
        fieldnames = set()
        for data in report_data:
            fieldnames.update(data.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header only if file is new or empty
            if not file_exists or os.path.getsize(filename) == 0:
                writer.writeheader()
            
            for data in report_data:
                # Clean data for CSV
                clean_row = {}
                for field in fieldnames:
                    value = data.get(field, '')
                    if hasattr(value, 'strftime'):
                        clean_row[field] = value.strftime('%H:%M')
                    elif value == 'N/A':
                        clean_row[field] = ''
                    else:
                        clean_row[field] = value
                writer.writerow(clean_row)
        
        logger.info(f"Earnings data appended to {filename} ({len(report_data)} rows added)")
    except Exception as e:
        logger.error(f"Failed to save CSV data to {filename}: {e}")

def main():
    """Main function to run the earnings report generation."""
    logger.info("Starting earnings report generation with integrated daily tickers generation")
    
    # Initialize clients
    fh = finnhub.Client(api_key=FINNHUB_API_KEY)
    api = AlpacaREST(
        key_id=APCA_KEY,
        secret_key=APCA_SECRET,
        base_url=ALPACA_BASE_URL
    )
    
    # Get today's date and generate tickers
    today = date.today()
    logger.info(f"Generating daily tickers for date: {today}")
    
    # Generate candidates using integrated functionality
    candidates = generate_daily_tickers(fh, today)
    if not candidates:
        logger.warning(f"No candidates generated for {today}, exiting")
        print(f"No valid tickers found for {today}. Exiting.")
        return
    
    logger.info(f"Generated {len(candidates)} tickers for {today}:")
    for candidate in candidates:
        logger.debug(f"  {candidate['symbol']}: {candidate['surprise']:.1f}%")
    
    # Set up date range for previous closes
    yesterday = get_prev_business_day(today)
    frm = yesterday.strftime("%Y-%m-%d")
    
    logger.info(f"Using {frm} for previous close data")
    
    # Get previous closes
    logger.info(f"Fetching previous close data for {frm}...")
    prev_closes = preload_prev_closes(api, candidates, frm)
    
    # Generate report with Alpaca + Finnhub data - MODIFIED TO RETURN BOTH REPORT AND DATA
    logger.info("Generating earnings report...")
    report, report_data = generate_earnings_report_with_data(candidates, prev_closes, api, fh, today.strftime("%Y-%m-%d"))
    
    # Save reports in multiple formats - NOW APPENDING TO SAME FILES
    # Save human-readable text report (appends automatically)
    save_earnings_report(report, "earnings_data.txt")
    
    # Save structured data for algorithms (appends to existing data)
    save_earnings_report_json(report_data, "earnings_data.json")
    save_earnings_report_csv(report_data, "earnings_data.csv")
    
    # Also print to console (for cron logging)
    print("\n" + report)
    
    logger.info(f"Earnings report generation completed for {today} with {len(candidates)} tickers")
    logger.info(f"Data appended to: earnings_data.txt, earnings_data.json, earnings_data.csv")

def generate_earnings_report_with_data(candidates: List[Dict], prev_closes: Dict[str, float], api: AlpacaREST, fh: finnhub.Client, today: str) -> tuple:
    """
    Generate comprehensive earnings report AND return structured data.
    Returns: (report_string, report_data_list)
    """
    logger.info(f"Generating comprehensive earnings report for {len(candidates)} candidates")
    
    report_lines = []
    report_lines.append("=" * 120)
    report_lines.append(f"EARNINGS REPORT - {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    report_lines.append("=" * 120)
    report_lines.append("")
    
    # Collect all data first for sorting
    report_data = []
    failed_reports = 0
    alpaca_success = 0
    finnhub_fallback = 0
    last_finnhub_call = 0.0
    
    for i, candidate in enumerate(candidates, 1):
        symbol = candidate['symbol']
        surprise = candidate['surprise']
        
        logger.info(f"Processing {i}/{len(candidates)}: {symbol}")
        
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
        
        # Get extended data regardless of quote_data success
        extended_data, last_finnhub_call = get_extended_stock_data(fh, symbol, api, today, last_finnhub_call)
        
        if not quote_data:
            logger.warning(f"No basic quote data available for {symbol} from either source")
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
                'sort_key': -999,  # Will sort to bottom
                **extended_data  # Include extended data even if basic quote failed
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
            'sort_key': pct_change_high if isinstance(pct_change_high, (int, float)) else -999,
            **extended_data  # Include all extended data
        })
    
    # Sort by %change high (descending) - highest gains first
    report_data.sort(key=lambda x: x['sort_key'], reverse=True)
    
    # Generate report with cleaner formatting (same as before)
    successful_reports = 0
    for rank, data in enumerate(report_data, 1):
        if data['sort_key'] == -999:
            continue
            
        successful_reports += 1
        symbol = data['symbol']
        
        # Main header line with key metrics
        report_lines.append(f"#{rank:2d} {symbol:<6} | EPS:{data['surprise']:6.1f}% | Source:{data['data_source']:<7} | PrevClose:${data['reference_prev_close']:7.2f}")
        
        # Price action line
        if data['sort_key'] != -999:
            report_lines.append(f"    OHLC: ${data['open_price']:7.2f} ${data['high_price']:7.2f}@{data['high_time']} ${data['low_price']:7.2f}@{data['low_time']} ${data['close_price']:7.2f}")
            report_lines.append(f"    Chg%: Open:{data['pct_change_open']:+6.2f}% High:{data['pct_change_high']:+6.2f}% Low:{data['pct_change_low']:+6.2f}% Close:{data['pct_change_close']:+6.2f}%")
        else:
            report_lines.append(f"    OHLC: NO DATA AVAILABLE")
            report_lines.append(f"    Chg%: NO DATA AVAILABLE")
        
        # Extended data line
        mc_str = f"{data['market_cap']:.0f}M" if isinstance(data['market_cap'], (int, float)) else str(data['market_cap'])
        rev_str = f"{data['revenue_surprise']:+.1f}%" if isinstance(data['revenue_surprise'], (int, float)) else str(data['revenue_surprise'])
        sector_str = str(data['sector'])[:15] if data['sector'] != 'N/A' else 'N/A'
        
        report_lines.append(f"    Info: MCap:{mc_str:<8} Rev:{rev_str:<6} Sector:{sector_str:<15} Time:{data['earnings_time']}")
        
        # Pre-market data
        pm_high = f"${data['premarket_high']:.2f}" if isinstance(data['premarket_high'], (int, float)) else str(data['premarket_high'])
        pm_low = f"${data['premarket_low']:.2f}" if isinstance(data['premarket_low'], (int, float)) else str(data['premarket_low'])
        pm_vol = f"{data['premarket_volume']:,.0f}" if isinstance(data['premarket_volume'], (int, float)) else str(data['premarket_volume'])
        
        report_lines.append(f"    PM  : High:{pm_high:<8} Low:{pm_low:<8} Volume:{pm_vol}")
        
        # Intraday progression
        om_vol = f"{data['opening_minute_volume']:,.0f}" if isinstance(data['opening_minute_volume'], (int, float)) else str(data['opening_minute_volume'])
        vol_cmp = str(data['volume_comparison'])
        
        pct_1min = f"{data['pct_1min']:+.2f}%" if isinstance(data['pct_1min'], (int, float)) else str(data['pct_1min'])
        pct_5min = f"{data['pct_5min']:+.2f}%" if isinstance(data['pct_5min'], (int, float)) else str(data['pct_5min'])
        pct_15min = f"{data['pct_15min']:+.2f}%" if isinstance(data['pct_15min'], (int, float)) else str(data['pct_15min'])
        pct_30min = f"{data['pct_30min']:+.2f}%" if isinstance(data['pct_30min'], (int, float)) else str(data['pct_30min'])
        pct_1hr = f"{data['pct_1hr']:+.2f}%" if isinstance(data['pct_1hr'], (int, float)) else str(data['pct_1hr'])
        
        report_lines.append(f"    Vol : OpenMin:{om_vol:<10} Ratio:{vol_cmp:<8} H>L:{data['high_before_low']}")
        report_lines.append(f"    Time: 1m:{pct_1min:<7} 5m:{pct_5min:<7} 15m:{pct_15min:<7} 30m:{pct_30min:<7} 1h:{pct_1hr}")
        
        report_lines.append("")  # Separator line between stocks
    
    # Add failed reports at the end
    if failed_reports > 0:
        report_lines.append("FAILED REPORTS:")
        report_lines.append("-" * 40)
        for data in report_data:
            if data['sort_key'] == -999:
                report_lines.append(f"{data['symbol']:<6} | EPS:{data['surprise']:6.1f}% | NO PRICE DATA")
        report_lines.append("")
    
    # Summary
    report_lines.append("=" * 120)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 120)
    report_lines.append(f"Total candidates: {len(candidates)}")
    report_lines.append(f"Successful reports: {successful_reports}")
    report_lines.append(f"Failed reports: {failed_reports}")
    report_lines.append(f"Data sources - Alpaca: {alpaca_success}, Finnhub fallback: {finnhub_fallback}")
    report_lines.append("")
    report_lines.append("LEGEND:")
    report_lines.append("EPS%: Earnings Per Share surprise percentage")
    report_lines.append("Rev%: Revenue surprise percentage") 
    report_lines.append("MCap: Market capitalization in millions")
    report_lines.append("PM: Pre-market data (4:00-9:30 AM ET)")
    report_lines.append("Vol: Volume data and ratios")
    report_lines.append("Time: Price change percentages at intervals from market open")
    report_lines.append("H>L: 1 if daily high occurred before daily low, 0 otherwise")
    report_lines.append("=" * 120)
    
    report = "\n".join(report_lines)
    logger.info(f"Generated comprehensive earnings report with {successful_reports} successful entries")
    
    # Return both the formatted report and the raw data
    return report, report_data

if __name__ == "__main__":
    main()