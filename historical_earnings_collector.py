import os
import sys
import time
import logging
import asyncio
import requests
import re
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python 3.8 fallback
from typing import List, Dict, Optional, Set
import csv

import websockets
import finnhub
from alpaca_trade_api.rest import REST as AlpacaREST
from typing import List, Dict, Set, Optional

# ── Constants & Configuration ─────────────────────────────────────────────────────────────────────────
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

# Historical data collection settings
HISTORICAL_DAYS = 30  # Number of days to go back
TODAY = date(2025, 8, 20)  # August 20, 2025

# ── Logging Setup ─────────────────────────────────────────────────────────────────────────────────────
def configure_logging() -> logging.Logger:
    logger = logging.getLogger("historical_earnings_trader")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s",
            "%Y-%m-%d %H:%M:%S %Z"
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # Less verbose console output for historical runs
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "historical_session_output.txt"),
            mode="a", encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

logger = configure_logging()

# ── Utils ─────────────────────────────────────────────────────────────────────────────────────────────
def get_prev_business_day(ref: date) -> date:
    d = ref
    while True:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            logger.debug(f"Previous business day for {ref} is {d}")
            return d

def is_business_day(d: date) -> bool:
    """Check if a date is a business day (Monday-Friday)"""
    return d.weekday() < 5

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

def safe_float(value, default=-1.0):
    """Convert value to float, return default if invalid"""
    if value is None or value == 'N/A' or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=-1):
    """Convert value to int, return default if invalid"""
    if value is None or value == 'N/A' or value == '':
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# ── Earnings Calendar & Candidate Filtering ──────────────────────────────────────────────────────────
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

# NEW: Volume calculation functions
def get_30_day_avg_volume(api: AlpacaREST, symbol: str, ref_date: str) -> float:
    """Get 30-day average volume for a symbol"""
    try:
        # Calculate date range for 30 days prior (excluding weekends)
        ref_dt = datetime.strptime(ref_date, "%Y-%m-%d").date()
        start_date = ref_dt - timedelta(days=45)  # Go back further to ensure 30 business days
        end_date = ref_dt - timedelta(days=1)  # Up to day before reference date
        
        logger.debug(f"Fetching 30-day volume data for {symbol} from {start_date} to {end_date}")
        
        bars = api.get_bars(
            symbol,
            timeframe="1Day",
            start=f"{start_date}T00:00:00-04:00",
            end=f"{end_date}T23:59:59-04:00",
            limit=50,
            adjustment="raw"
        )
        
        if not bars or len(bars) < 5:  # Need at least 5 days of data
            logger.warning(f"Insufficient volume data for {symbol}: {len(bars) if bars else 0} days")
            return -1.0
        
        # Take last 30 business days (or whatever we have)
        recent_bars = bars[-30:]
        total_volume = sum(bar.v for bar in recent_bars)
        avg_volume = total_volume / len(recent_bars)
        
        logger.debug(f"{symbol} 30-day avg volume: {avg_volume:,.0f} (from {len(recent_bars)} days)")
        return avg_volume
        
    except Exception as e:
        logger.warning(f"Failed to get 30-day volume for {symbol}: {e}")
        return -1.0

def get_7_day_avg_volume(api: AlpacaREST, symbol: str, ref_date: str) -> float:
    """Get 7-day (1 week) average volume for a symbol"""
    try:
        # Calculate date range for 7 days prior
        ref_dt = datetime.strptime(ref_date, "%Y-%m-%d").date()
        start_date = ref_dt - timedelta(days=10)  # Go back 10 days to ensure 7 business days
        end_date = ref_dt - timedelta(days=1)  # Up to day before reference date
        
        logger.debug(f"Fetching 7-day volume data for {symbol} from {start_date} to {end_date}")
        
        bars = api.get_bars(
            symbol,
            timeframe="1Day",
            start=f"{start_date}T00:00:00-04:00",
            end=f"{end_date}T23:59:59-04:00",
            limit=15,
            adjustment="raw"
        )
        
        if not bars or len(bars) < 3:  # Need at least 3 days of data
            logger.warning(f"Insufficient 7-day volume data for {symbol}: {len(bars) if bars else 0} days")
            return -1.0
        
        # Take last 7 business days (or whatever we have)
        recent_bars = bars[-7:]
        total_volume = sum(bar.v for bar in recent_bars)
        avg_volume = total_volume / len(recent_bars)
        
        logger.debug(f"{symbol} 7-day avg volume: {avg_volume:,.0f} (from {len(recent_bars)} days)")
        return avg_volume
        
    except Exception as e:
        logger.warning(f"Failed to get 7-day volume for {symbol}: {e}")
        return -1.0

def calculate_volume_ratios(opening_minute_volume: float, avg_30_day: float, avg_7_day: float) -> tuple:
    """Calculate volume ratios vs 30-day and 7-day averages"""
    ratio_30_day = opening_minute_volume / avg_30_day if avg_30_day > 0 else -1.0
    ratio_7_day = opening_minute_volume / avg_7_day if avg_7_day > 0 else -1.0
    
    return ratio_30_day, ratio_7_day

# Modified extended stock data function
def get_extended_stock_data(fh: finnhub.Client, symbol: str, api: AlpacaREST, today: str, last_finnhub_call: float) -> tuple:
    """
    Get extended stock data with numerical values only.
    Returns tuple: (extended_data_dict, updated_last_finnhub_call_time)
    """
    extended_data = {
        'market_cap': -1.0,
        'revenue_surprise': -999.0, 
        'sector': 'Unknown',
        'industry': 'Unknown',
        'earnings_time': -1,  # 1 for AMC, 0 for BMO, -1 for unknown
        'premarket_high': -1.0,
        'premarket_low': -1.0,
        'premarket_volume': -1,
        'opening_minute_volume': -1,
        'volume_30_day_avg': -1.0,
        'volume_7_day_avg': -1.0,
        'volume_ratio_30_day': -1.0,
        'volume_ratio_7_day': -1.0,
        'pct_1min': -999.0,
        'pct_5min': -999.0, 
        'pct_15min': -999.0,
        'pct_30min': -999.0,
        'pct_1hr': -999.0,
        'high_before_low': -1  # 1 if high before low, 0 if low before high, -1 if unknown
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
            raw_mc = profile.get('marketCapitalization', 0)
            if raw_mc and raw_mc > 0:
                extended_data['market_cap'] = float(raw_mc)
                logger.debug(f"Market cap for {symbol}: {raw_mc:.1f}M")
            
            # Sector information
            sector = profile.get('finnhubIndustry', 'Unknown')
            extended_data['sector'] = sector if sector else 'Unknown'
            extended_data['industry'] = sector if sector else 'Unknown'
            
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
                
                # Convert earnings time to numerical: 1 for AMC, 0 for BMO
                earnings_hour = entry.get('hour', '').lower()
                if earnings_hour == 'amc':
                    extended_data['earnings_time'] = 1
                elif earnings_hour == 'bmo':
                    extended_data['earnings_time'] = 0
                break
                
    except Exception as e:
        logger.warning(f"Failed to fetch earnings calendar for {symbol}: {e}")
    
    try:
        # Get pre-market data
        premarket_data = get_premarket_data(api, symbol, today)
        extended_data['premarket_high'] = safe_float(premarket_data.get('premarket_high'))
        extended_data['premarket_low'] = safe_float(premarket_data.get('premarket_low'))
        extended_data['premarket_volume'] = safe_int(premarket_data.get('premarket_volume'))
        
        # Get intraday data with NEW volume analysis
        intraday_data = get_intraday_analysis(api, symbol, today)
        extended_data.update(intraday_data)
        
    except Exception as e:
        logger.warning(f"Failed to fetch market data for {symbol}: {e}")
    
    return extended_data, last_finnhub_call

# Modified intraday analysis with new volume comparison
def get_intraday_analysis(api: AlpacaREST, symbol: str, today: str) -> dict:
    """Get intraday percentage changes and NEW volume analysis"""
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
                'opening_minute_volume': -1,
                'volume_30_day_avg': -1.0,
                'volume_7_day_avg': -1.0,
                'volume_ratio_30_day': -1.0,
                'volume_ratio_7_day': -1.0,
                'pct_1min': -999.0,
                'pct_5min': -999.0,
                'pct_15min': -999.0,
                'pct_30min': -999.0,
                'pct_1hr': -999.0,
                'high_before_low': -1
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
                'opening_minute_volume': -1,
                'volume_30_day_avg': -1.0,
                'volume_7_day_avg': -1.0,
                'volume_ratio_30_day': -1.0,
                'volume_ratio_7_day': -1.0,
                'pct_1min': -999.0,
                'pct_5min': -999.0,
                'pct_15min': -999.0,
                'pct_30min': -999.0,
                'pct_1hr': -999.0,
                'high_before_low': -1
            }
        
        open_price = market_bars[0].o
        opening_minute_volume = market_bars[0].v
        
        # NEW: Get 30-day and 7-day average volumes
        avg_30_day = get_30_day_avg_volume(api, symbol, today)
        avg_7_day = get_7_day_avg_volume(api, symbol, today)
        
        # Calculate volume ratios
        ratio_30_day, ratio_7_day = calculate_volume_ratios(opening_minute_volume, avg_30_day, avg_7_day)
        
        # Calculate percentage changes at different time intervals
        def get_price_at_minutes(minutes):
            if len(market_bars) > minutes:
                return market_bars[minutes].c
            return market_bars[-1].c
        
        pct_changes = {}
        intervals = [1, 5, 15, 30, 60]
        for interval in intervals:
            price = get_price_at_minutes(interval - 1)  # 0-indexed
            pct_change = calculate_percentage_change(open_price, price)
            pct_changes[f'pct_{interval}min' if interval < 60 else 'pct_1hr'] = pct_change
        
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
        
        return {
            'opening_minute_volume': opening_minute_volume,
            'volume_30_day_avg': avg_30_day,
            'volume_7_day_avg': avg_7_day,
            'volume_ratio_30_day': ratio_30_day,
            'volume_ratio_7_day': ratio_7_day,
            **pct_changes,
            'high_before_low': high_before_low
        }
        
    except Exception as e:
        logger.warning(f"Failed to fetch intraday analysis for {symbol}: {e}")
        return {
            'opening_minute_volume': -1,
            'volume_30_day_avg': -1.0,
            'volume_7_day_avg': -1.0,
            'volume_ratio_30_day': -1.0,
            'volume_ratio_7_day': -1.0,
            'pct_1min': -999.0,
            'pct_5min': -999.0,
            'pct_15min': -999.0,
            'pct_30min': -999.0,
            'pct_1hr': -999.0,
            'high_before_low': -1
        }

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
            return {'premarket_high': -1.0, 'premarket_low': -1.0, 'premarket_volume': -1}
        
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
            return {'premarket_high': -1.0, 'premarket_low': -1.0, 'premarket_volume': -1}
        
        premarket_high = max(bar.h for bar in premarket_bars)
        premarket_low = min(bar.l for bar in premarket_bars)
        
        return {
            'premarket_high': premarket_high,
            'premarket_low': premarket_low, 
            'premarket_volume': total_volume
        }
        
    except Exception as e:
        logger.warning(f"Failed to fetch pre-market data for {symbol}: {e}")
        return {'premarket_high': -1.0, 'premarket_low': -1.0, 'premarket_volume': -1}

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

# ── Market Data Integration (Alpaca + Finnhub Fallback) ──────────────────────────────────────────────
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

def generate_earnings_report_with_data(candidates: List[Dict], prev_closes: Dict[str, float], api: AlpacaREST, fh: finnhub.Client, today: str) -> tuple:
    """
    Generate comprehensive earnings report AND return structured data.
    Returns: (report_string, report_data_list)
    """
    logger.info(f"Generating comprehensive earnings report for {len(candidates)} candidates")
    
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
            # Add entry with default numerical values
            report_data.append({
                'symbol': symbol,
                'surprise': surprise,
                'alpaca_prev_close': alpaca_prev_close,
                'data_source': 'FAILED',
                'open_price': -1.0,
                'pct_change_open': -999.0,
                'high_price': -1.0,
                'high_time': 'N/A',
                'low_price': -1.0,
                'low_time': 'N/A',
                'close_price': -1.0,
                'pct_change_high': -999.0,
                'pct_change_low': -999.0,
                'pct_change_close': -999.0,
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
        pct_change_open = calculate_percentage_change(reference_prev_close, open_price) if reference_prev_close > 0 else -999.0
        pct_change_high = calculate_percentage_change(open_price, high_price) if open_price > 0 else -999.0
        pct_change_low = calculate_percentage_change(open_price, low_price) if open_price > 0 else -999.0
        pct_change_close = calculate_percentage_change(reference_prev_close, close_price) if reference_prev_close > 0 else -999.0
        
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
            'sort_key': pct_change_high if isinstance(pct_change_high, (int, float)) and pct_change_high > -999 else -999,
            **extended_data  # Include all extended data
        })
    
    # Sort by %change high (descending) - highest gains first
    report_data.sort(key=lambda x: x['sort_key'], reverse=True)
    
    logger.info(f"Generated data for {len(report_data)} stocks on {today}")
    
    # Return the raw data (no formatted report needed for historical collection)
    return "", report_data

def save_historical_earnings_csv(all_report_data: List[Dict], filename: str = "historical_earnings_data.csv"):
    """Save all historical earnings data as CSV."""
    import csv
    import os
    
    try:
        if not all_report_data:
            logger.warning("No data to save")
            return
        
        # Get all possible field names
        fieldnames = set()
        for data in all_report_data:
            fieldnames.update(data.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for data in all_report_data:
                # Clean data for CSV - all numerical now
                clean_row = {}
                for field in fieldnames:
                    value = data.get(field, '')
                    if hasattr(value, 'strftime'):
                        clean_row[field] = value.strftime('%H:%M')
                    elif field in ['sector', 'industry', 'symbol', 'data_source', 'high_time', 'low_time', 'report_date', 'report_timestamp']:
                        # Keep text fields as-is
                        clean_row[field] = str(value) if value is not None else ''
                    else:
                        # Numerical fields - use the value directly
                        clean_row[field] = value if value is not None else -1
                writer.writerow(clean_row)
        
        logger.info(f"Historical earnings data saved to {filename} ({len(all_report_data)} rows)")
        print(f"\n✓ Historical data collection complete!")
        print(f"✓ {len(all_report_data)} records saved to {filename}")
        print(f"✓ You can now copy this data into your existing earnings_data.csv file")
        
    except Exception as e:
        logger.error(f"Failed to save CSV data to {filename}: {e}")

def get_historical_dates() -> List[date]:
    """Get list of business days for the past 30 days (excluding today)"""
    historical_dates = []
    current_date = TODAY - timedelta(days=1)  # Start from yesterday
    
    days_collected = 0
    while days_collected < HISTORICAL_DAYS:
        if is_business_day(current_date):
            historical_dates.append(current_date)
            days_collected += 1
        current_date -= timedelta(days=1)
    
    # Reverse to process oldest to newest
    historical_dates.reverse()
    return historical_dates

def main():
    """Main function to collect historical earnings data."""
    logger.info("Starting historical earnings data collection")
    logger.info(f"Collecting data for past {HISTORICAL_DAYS} business days excluding {TODAY}")
    
    # Initialize clients
    fh = finnhub.Client(api_key=FINNHUB_API_KEY)
    api = AlpacaREST(
        key_id=APCA_KEY,
        secret_key=APCA_SECRET,
        base_url=ALPACA_BASE_URL
    )
    
    # Get historical dates
    historical_dates = get_historical_dates()
    logger.info(f"Processing {len(historical_dates)} business days: {historical_dates[0]} to {historical_dates[-1]}")
    
    all_report_data = []
    total_candidates = 0
    processed_days = 0
    
    for i, target_date in enumerate(historical_dates, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing day {i}/{len(historical_dates)}: {target_date}")
        logger.info(f"{'='*60}")
        
        try:
            # Generate candidates for this date
            candidates = generate_daily_tickers(fh, target_date)
            if not candidates:
                logger.info(f"No candidates found for {target_date}, skipping")
                continue
            
            logger.info(f"Found {len(candidates)} candidates for {target_date}")
            total_candidates += len(candidates)
            
            # Get previous closes for this date
            yesterday = get_prev_business_day(target_date)
            prev_closes = preload_prev_closes(api, candidates, yesterday.strftime("%Y-%m-%d"))
            
            # Generate report data for this date
            _, report_data = generate_earnings_report_with_data(
                candidates, 
                prev_closes, 
                api, 
                fh, 
                target_date.strftime("%Y-%m-%d")
            )
            
            # Add date and timestamp to each record
            current_timestamp = datetime.now(TZ).isoformat()
            for data in report_data:
                data['report_date'] = target_date.strftime('%Y-%m-%d')
                data['report_timestamp'] = current_timestamp
            
            all_report_data.extend(report_data)
            processed_days += 1
            
            logger.info(f"✓ Processed {len(report_data)} records for {target_date}")
            print(f"Day {i}/{len(historical_dates)} ({target_date}): {len(candidates)} candidates, {len(report_data)} records")
            
        except Exception as e:
            logger.error(f"Error processing {target_date}: {e}")
            print(f"✗ Error processing {target_date}: {e}")
            continue
        
        # Add small delay between days to be respectful to APIs
        if i < len(historical_dates):  # Don't sleep after the last day
            time.sleep(2)
    
    # Save all data
    if all_report_data:
        save_historical_earnings_csv(all_report_data, "historical_earnings_data.csv")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"HISTORICAL DATA COLLECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Processed days: {processed_days}/{len(historical_dates)}")
        print(f"Total candidates found: {total_candidates}")
        print(f"Total records collected: {len(all_report_data)}")
        print(f"Date range: {historical_dates[0]} to {historical_dates[-1]}")
        print(f"Output file: historical_earnings_data.csv")
        print(f"{'='*60}")
        
        logger.info(f"Historical data collection completed: {len(all_report_data)} total records from {processed_days} days")
    else:
        logger.warning("No historical data collected")
        print("⚠ No historical data was collected")

if __name__ == "__main__":
    main()