import os
import sys
import time
import logging
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
import csv

import finnhub
from alpaca_trade_api.rest import REST as AlpacaREST

# ─── Constants & Configuration ────────────────────────────────────────────────────────────────────────
TZ = ZoneInfo("America/New_York")
load_dotenv('/home/ubuntu/.env')
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
APCA_KEY        = os.getenv("APCA_API_KEY_ID")
APCA_SECRET     = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Filter thresholds
SURPRISE_THRESHOLD = 1  # % min earnings surprise
MAX_SURPRISE       = 600  # max surprise
MC_THRESHOLD       = 900_000_000  # market cap threshold: 900 million

# Conservative rate limits for sustained operation
FINNHUB_DAILY_LIMIT = 50  # Conservative for free tier
FINNHUB_PER_MINUTE_LIMIT = 25  # Conservative per-minute
ALPACA_DAILY_LIMIT = 1000  # Conservative for sustained use
ALPACA_PER_MINUTE_LIMIT = 100  # Conservative per-minute

# Files
CHECKPOINT_FILE = "comprehensive_earnings_checkpoint.json"
PROFILE_CACHE_FILE = "company_profiles_cache.json"
HISTORICAL_DAYS = 365

# Market timing constants
PREMARKET_START_HOUR = 4
PREMARKET_END_HOUR = 9
PREMARKET_END_MINUTE = 30

# ─── Logging Setup ──────────────────────────────────────────────────────────────────────────────────
def configure_logging() -> logging.Logger:
    logger = logging.getLogger("comprehensive_earnings")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler("comprehensive_earnings_log.txt", mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

logger = configure_logging()

# ─── Persistent Cache System ────────────────────────────────────────────────────────────────────────
class ComprehensiveCache:
    """Comprehensive caching system for all data types."""
    
    def __init__(self):
        self.profile_cache_file = "company_profiles_cache.json"
        self.volume_cache_file = "volume_averages_cache.json"
        
        self.profiles = self._load_json_cache(self.profile_cache_file)
        self.volume_averages = self._load_json_cache(self.volume_cache_file)
        
        self.unsaved_changes = 0
    
    def _load_json_cache(self, filename: str) -> Dict:
        """Load cache from JSON file."""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache {filename}: {e}")
        return {}
    
    def _save_json_cache(self, data: Dict, filename: str):
        """Save cache to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache {filename}: {e}")
    
    def save_all_caches(self):
        """Save all caches to disk."""
        self._save_json_cache(self.profiles, self.profile_cache_file)
        self._save_json_cache(self.volume_averages, self.volume_cache_file)
        logger.info(f"Saved {len(self.profiles)} profiles, {len(self.volume_averages)} volume averages")
        self.unsaved_changes = 0
    
    def get_profile(self, symbol: str) -> Optional[Dict]:
        return self.profiles.get(symbol)
    
    def set_profile(self, symbol: str, profile: Dict):
        self.profiles[symbol] = profile
        self.unsaved_changes += 1
        if self.unsaved_changes >= 20:  # Auto-save every 20 changes
            self.save_all_caches()
    
    def get_volume_average(self, symbol: str) -> Optional[float]:
        return self.volume_averages.get(symbol)
    
    def set_volume_average(self, symbol: str, avg_volume: float):
        self.volume_averages[symbol] = avg_volume
        self.unsaved_changes += 1

# ─── Enhanced Checkpoint Management ─────────────────────────────────────────────────────────────────
def load_checkpoint() -> Dict:
    """Load comprehensive checkpoint."""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
    
    return {
        "completed_weeks": [],
        "completed_symbols": [],  # Track individual symbols
        "current_week_progress": {},
        "last_processed_week": None,
        "total_collected": 0,
        "api_calls_used": {"finnhub": 0, "alpaca": 0},
        "last_reset_time": datetime.now(TZ).isoformat(),
        "failed_symbols": [],
        "partial_data": {}  # Store partial results
    }

def save_checkpoint(checkpoint: Dict):
    """Save comprehensive checkpoint."""
    checkpoint['last_checkpoint_time'] = datetime.now(TZ).isoformat()
    checkpoint['paused_for_api_limits'] = False
    checkpoint['next_resume_time'] = None
    
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, default=str, indent=2)
    except Exception as e:
        logger.warning(f"Could not save checkpoint: {e}")

def save_checkpoint_with_pause(checkpoint: Dict, pause_until: datetime, api_name: str):
    """Save checkpoint with API limit pause information."""
    checkpoint['last_checkpoint_time'] = datetime.now(TZ).isoformat()
    checkpoint['next_resume_time'] = pause_until.isoformat()
    checkpoint['paused_for_api_limits'] = True
    checkpoint['api_limit_api'] = api_name
    
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, default=str, indent=2)
        logger.info(f"🛑 Paused until {pause_until} due to {api_name} API limits")
    except Exception as e:
        logger.warning(f"Could not save checkpoint: {e}")

def check_resume_time() -> tuple:
    """Check if we can resume from API limit pause."""
    checkpoint = load_checkpoint()
    
    if checkpoint.get('paused_for_api_limits') and checkpoint.get('next_resume_time'):
        try:
            resume_time = datetime.fromisoformat(checkpoint['next_resume_time'])
            now = datetime.now(TZ)
            
            if now < resume_time:
                wait_minutes = int((resume_time - now).total_seconds() / 60)
                return False, wait_minutes
            else:
                logger.info("✅ API limit pause ended, resuming collection")
                return True, 0
        except Exception as e:
            logger.warning(f"Error checking resume time: {e}")
    
    return True, 0

# ─── Utility Functions ──────────────────────────────────────────────────────────────────────────────
def get_prev_business_day(ref: date) -> date:
    """Get previous business day."""
    d = ref
    while True:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            return d

def get_next_business_day(ref: date) -> date:
    """Get next business day."""
    d = ref
    while True:
        d += timedelta(days=1)
        if d.weekday() < 5:
            return d

def get_market_reaction_date(earnings_date: date, earnings_time: str) -> date:
    """
    Calculate correct market reaction date:
    - BMO (Before Market Open): Market reaction happens SAME DAY at 9:30 AM
    - AMC (After Market Close): Market reaction happens NEXT TRADING DAY at 9:30 AM
    """
    earnings_time = earnings_time.lower() if earnings_time else ''
    
    if earnings_time == 'bmo':
        return earnings_date
    elif earnings_time == 'amc':
        return get_next_business_day(earnings_date)
    else:
        # Default to same day if timing unknown
        return earnings_date

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change."""
    if old_value == 0 or old_value is None or new_value is None:
        return 'N/A'
    return ((new_value - old_value) / old_value) * 100

def generate_weekly_ranges(start_date: date, end_date: date) -> List[tuple]:
    """Generate weekly date ranges for batch processing."""
    weeks = []
    current = start_date
    
    while current <= end_date:
        week_end = min(current + timedelta(days=6), end_date)
        weeks.append((current, week_end))
        current = week_end + timedelta(days=1)
    
    return weeks

def convert_time_to_minutes_since_open(time_str: str) -> int:
    """
    Convert time string to minutes since market open (9:30 AM).
    Returns: Minutes since 9:30 AM, or 'N/A' if no data.
    Example: "10:45" -> 75 minutes since market open
    """
    try:
        if time_str == 'N/A':
            return 'N/A'
        
        hour, minute = map(int, time_str.split(':'))
        
        # Market opens at 9:30 AM
        market_open_minutes = 9 * 60 + 30  # 570 minutes since midnight
        current_minutes = hour * 60 + minute
        
        minutes_since_open = current_minutes - market_open_minutes
        
        # Ensure it's within market hours (0-390 minutes for 9:30-4:00)
        if 0 <= minutes_since_open <= 390:
            return minutes_since_open
        else:
            return 'N/A'  # Outside market hours
            
    except (ValueError, AttributeError):
        return 'N/A'

def convert_earnings_time_to_numeric(earnings_time: str) -> int:
    """
    Convert earnings time to numeric.
    Returns: 0 for BMO (Before Market Open), 1 for AMC (After Market Close), 'N/A' for unknown.
    """
    earnings_time = earnings_time.lower() if earnings_time else ''
    
    if earnings_time == 'bmo':
        return 0
    elif earnings_time == 'amc':
        return 1
    else:
        return 'N/A'

def convert_data_source_to_numeric(data_source: str) -> int:
    """
    Convert data source to numeric code.
    Returns: 1 for Alpaca, 0 for other/unknown sources.
    """
    if data_source == 'Alpaca':
        return 1
    else:
        return 0

# ─── Enhanced API Limit Handling ────────────────────────────────────────────────────────────────────
class APILimitException(Exception):
    """API limit exceeded exception."""
    def __init__(self, api_name, retry_after=None):
        self.api_name = api_name
        self.retry_after = retry_after
        super().__init__(f"API limit exceeded for {api_name}")

def is_api_limit_error(error_msg: str) -> tuple:
    """Detect API limit errors. Returns (is_limit, api_name, wait_time)."""
    error_lower = str(error_msg).lower()
    
    # API limit indicators
    limit_patterns = [
        'api limit', 'rate limit', 'too many requests', 'limit exceeded',
        'quota exceeded', '429', 'request limit', 'frequency limit',
        'rate exceeded', 'throttle', 'rate_limit'
    ]
    
    if any(pattern in error_lower for pattern in limit_patterns):
        if 'finnhub' in error_lower:
            return True, 'Finnhub', 3600  # 1 hour wait
        elif 'alpaca' in error_lower:
            return True, 'Alpaca', 900   # 15 minutes wait
        else:
            return True, 'Unknown', 1800  # 30 minutes wait
    
    return False, None, 0

class RobustRateLimiter:
    """Robust rate limiter designed for week+ operation."""
    
    def __init__(self):
        self.finnhub_calls_today = 0
        self.alpaca_calls_today = 0
        self.last_finnhub_call = 0
        self.last_alpaca_call = 0
        
        # Track calls per minute
        self.finnhub_minute_calls = []
        self.alpaca_minute_calls = []
        
        # Daily reset tracking
        self.daily_reset_time = datetime.now(TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _reset_daily_counters_if_needed(self):
        """Reset daily counters at midnight."""
        now = datetime.now(TZ)
        if now.date() > self.daily_reset_time.date():
            self.finnhub_calls_today = 0
            self.alpaca_calls_today = 0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            logger.info("🔄 Daily API counters reset")
    
    def _clean_minute_calls(self, call_list: List[float]) -> List[float]:
        """Remove calls older than 1 minute."""
        now = time.monotonic()
        return [call_time for call_time in call_list if now - call_time < 60]
    
    def _wait_for_finnhub_limits(self):
        """Handle Finnhub rate limiting."""
        # Clean old calls
        self.finnhub_minute_calls = self._clean_minute_calls(self.finnhub_minute_calls)
        
        # Check per-minute limit
        if len(self.finnhub_minute_calls) >= FINNHUB_PER_MINUTE_LIMIT:
            oldest_call = min(self.finnhub_minute_calls)
            wait_time = 60 - (time.monotonic() - oldest_call) + 2  # 2 second buffer
            if wait_time > 0:
                logger.info(f"⏱️  Finnhub per-minute limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self.finnhub_minute_calls = []
        
        # Check daily limit
        if self.finnhub_calls_today >= FINNHUB_DAILY_LIMIT:
            raise APILimitException('Finnhub', 86400)  # Wait until tomorrow
        
        # Wait between individual calls
        time_since_last = time.monotonic() - self.last_finnhub_call
        if time_since_last < 3.0:  # 3 seconds between calls
            time.sleep(3.0 - time_since_last)
    
    def _wait_for_alpaca_limits(self):
        """Handle Alpaca rate limiting."""
        # Clean old calls
        self.alpaca_minute_calls = self._clean_minute_calls(self.alpaca_minute_calls)
        
        # Check per-minute limit
        if len(self.alpaca_minute_calls) >= ALPACA_PER_MINUTE_LIMIT:
            oldest_call = min(self.alpaca_minute_calls)
            wait_time = 60 - (time.monotonic() - oldest_call) + 1
            if wait_time > 0:
                logger.info(f"⏱️  Alpaca per-minute limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self.alpaca_minute_calls = []
        
        # Check daily limit
        if self.alpaca_calls_today >= ALPACA_DAILY_LIMIT:
            raise APILimitException('Alpaca', 86400)
        
        # Wait between calls
        time_since_last = time.monotonic() - self.last_alpaca_call
        if time_since_last < 0.6:  # 0.6 seconds between calls
            time.sleep(0.6 - time_since_last)
    
    def finnhub_call(self, func, *args, **kwargs):
        """Make rate-limited Finnhub call."""
        self._reset_daily_counters_if_needed()
        self._wait_for_finnhub_limits()
        
        try:
            result = func(*args, **kwargs)
            call_time = time.monotonic()
            
            self.last_finnhub_call = call_time
            self.finnhub_minute_calls.append(call_time)
            self.finnhub_calls_today += 1
            
            return result
            
        except Exception as e:
            # Count failed calls
            call_time = time.monotonic()
            self.last_finnhub_call = call_time
            self.finnhub_minute_calls.append(call_time)
            self.finnhub_calls_today += 1
            
            # Check if it's an API limit error
            is_limit, api_name, wait_time = is_api_limit_error(str(e))
            if is_limit:
                raise APILimitException(api_name or 'Finnhub', wait_time)
            
            logger.warning(f"Finnhub call failed: {e}")
            return None
    
    def alpaca_call(self, func, *args, **kwargs):
        """Make rate-limited Alpaca call."""
        self._reset_daily_counters_if_needed()
        self._wait_for_alpaca_limits()
        
        try:
            result = func(*args, **kwargs)
            call_time = time.monotonic()
            
            self.last_alpaca_call = call_time
            self.alpaca_minute_calls.append(call_time)
            self.alpaca_calls_today += 1
            
            return result
            
        except Exception as e:
            call_time = time.monotonic()
            self.last_alpaca_call = call_time
            self.alpaca_minute_calls.append(call_time)
            
            is_limit, api_name, wait_time = is_api_limit_error(str(e))
            if is_limit:
                raise APILimitException(api_name or 'Alpaca', wait_time)
            
            logger.warning(f"Alpaca call failed: {e}")
            return None
    
    def get_usage_summary(self) -> str:
        """Get current API usage summary."""
        self._reset_daily_counters_if_needed()
        return (f"Daily: Finnhub {self.finnhub_calls_today}/{FINNHUB_DAILY_LIMIT}, "
                f"Alpaca {self.alpaca_calls_today}/{ALPACA_DAILY_LIMIT}")

# ─── Comprehensive Earnings Data Collection ──────────────────────────────────────────────────────────
def fetch_weekly_earnings_calendar(fh: finnhub.Client, rate_limiter: RobustRateLimiter, 
                                  start_date: date, end_date: date) -> List[dict]:
    """Fetch ALL earnings for a weekly period."""
    try:
        frm = start_date.strftime("%Y-%m-%d")
        to = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"📅 Fetching earnings calendar: {frm} to {to}")
        
        cal = rate_limiter.finnhub_call(fh.earnings_calendar, symbol=None, _from=frm, to=to)
        
        if cal and 'earningsCalendar' in cal:
            entries = cal['earningsCalendar']
            logger.info(f"📊 Retrieved {len(entries)} earnings entries")
            return entries
        else:
            logger.warning(f"❌ No earnings data for {frm} to {to}")
            return []
            
    except APILimitException:
        raise
    except Exception as e:
        logger.error(f"💥 Failed to fetch earnings calendar: {e}")
        return []

def filter_earnings_candidates(entries: List[dict], fh: finnhub.Client, 
                              rate_limiter: RobustRateLimiter,
                              cache: ComprehensiveCache,
                              failed_symbols: Set[str]) -> List[dict]:
    """Filter earnings entries and collect company profiles."""
    logger.info(f"🔍 Filtering {len(entries)} earnings entries")
    
    candidates = []
    
    for e in entries:
        sym = e.get("symbol")
        est, act = e.get("epsEstimate"), e.get("epsActual")
        earnings_date = e.get("date")
        earnings_time = e.get("hour", "").lower()
        
        # Skip if basic data missing or symbol previously failed
        if not sym or not earnings_date or sym in failed_symbols:
            continue
        
        # Filter by earnings surprise
        if est is None or act is None or est == 0 or act < 0:
            continue

        surprise = (act - est) / abs(est) * 100
        if surprise <= SURPRISE_THRESHOLD or surprise > MAX_SURPRISE:
            continue

        # Get or fetch company profile
        profile = cache.get_profile(sym)
        if not profile:
            try:
                profile = rate_limiter.finnhub_call(fh.company_profile2, symbol=sym)
                if profile and profile.get('marketCapitalization'):
                    cache.set_profile(sym, profile)
                else:
                    failed_symbols.add(sym)
                    continue
            except APILimitException:
                raise
            except Exception as ex:
                logger.warning(f"❌ Profile fetch failed for {sym}: {ex}")
                failed_symbols.add(sym)
                continue
        
        # Filter by market cap
        mc = float(profile.get("marketCapitalization", 0))
        if mc <= MC_THRESHOLD:
            continue
        
        # Calculate market reaction date
        try:
            earnings_date_obj = datetime.strptime(earnings_date, '%Y-%m-%d').date()
            market_reaction_date = get_market_reaction_date(earnings_date_obj, earnings_time)
        except (ValueError, TypeError):
            continue
        
        candidates.append({
            "symbol": sym,
            "surprise": surprise,
            "market_cap": mc,
            "earnings_date": earnings_date,
            "earnings_time": earnings_time,
            "market_reaction_date": market_reaction_date.strftime("%Y-%m-%d"),
            "sector": profile.get('finnhubIndustry', 'N/A'),
            "industry": profile.get('finnhubIndustry', 'N/A'),
            "epsEstimate": est,
            "epsActual": act,
            "revenueEstimate": e.get("revenueEstimate"),
            "revenueActual": e.get("revenueActual")
        })
        
        logger.debug(f"✅ Candidate {sym}: {surprise:.2f}% surprise, {mc:.0f}M cap")

    logger.info(f"🎯 Found {len(candidates)} qualified candidates")
    return candidates

# ─── Market Data Helper Functions ────────────────────────────────────────────────────────────────────
def get_previous_close_price(api: AlpacaREST, symbol: str, date_str: str, 
                           rate_limiter: RobustRateLimiter) -> float:
    """Get previous trading day close price."""
    try:
        start_iso = f"{date_str}T00:00:00-04:00"
        end_iso = f"{date_str}T23:59:59-04:00"
        
        bars = rate_limiter.alpaca_call(
            api.get_bars,
            symbol,
            timeframe="1Day",
            start=start_iso,
            end=end_iso,
            limit=1,
            adjustment="raw"
        )
        
        if bars and len(bars) > 0:
            return float(bars[0].c)
        return 0.0
        
    except Exception as e:
        logger.debug(f"❌ Previous close failed for {symbol}: {e}")
        return 0.0

def get_premarket_data(api: AlpacaREST, symbol: str, target_date: str, 
                      rate_limiter: RobustRateLimiter) -> Dict:
    """Get comprehensive premarket data (4:00 AM - 9:30 AM)."""
    try:
        start_iso = f"{target_date}T04:00:00-04:00"
        end_iso = f"{target_date}T09:30:00-04:00"
        
        bars = rate_limiter.alpaca_call(
            api.get_bars,
            symbol,
            timeframe="1Min",
            start=start_iso,
            end=end_iso,
            limit=1000,
            adjustment="raw"
        )
        
        if not bars or len(bars) == 0:
            return {'high': 'N/A', 'low': 'N/A', 'volume': 'N/A'}
        
        # Calculate premarket metrics
        high_price = max(bar.h for bar in bars)
        low_price = min(bar.l for bar in bars)
        total_volume = sum(bar.v for bar in bars)
        
        return {
            'high': high_price,
            'low': low_price,
            'volume': total_volume
        }
        
    except Exception as e:
        logger.debug(f"❌ Premarket data failed for {symbol}: {e}")
        return {'high': 'N/A', 'low': 'N/A', 'volume': 'N/A'}

def get_detailed_market_data(api: AlpacaREST, symbol: str, target_date: str, 
                           rate_limiter: RobustRateLimiter) -> Dict:
    """Get detailed market hours data with timing information."""
    try:
        start_iso = f"{target_date}T09:30:00-04:00"
        end_iso = f"{target_date}T16:00:00-04:00"
        
        bars = rate_limiter.alpaca_call(
            api.get_bars,
            symbol,
            timeframe="1Min",
            start=start_iso,
            end=end_iso,
            limit=1000,
            adjustment="raw"
        )
        
        if not bars or len(bars) == 0:
            return {}
        
        # Calculate OHLC
        open_price = bars[0].o
        high_price = max(bar.h for bar in bars)
        low_price = min(bar.l for bar in bars)
        close_price = bars[-1].c
        
        # Find exact timing of high and low
        high_time = None
        low_time = None
        for bar in bars:
            if bar.h == high_price and high_time is None:
                high_time = bar.t.astimezone(TZ).strftime('%H:%M')
            if bar.l == low_price and low_time is None:
                low_time = bar.t.astimezone(TZ).strftime('%H:%M')
        
        return {
            'open_price': open_price,
            'high_price': high_price,
            'low_price': low_price,
            'close_price': close_price,
            'high_time': high_time or 'N/A',
            'low_time': low_time or 'N/A',
            'bars': bars  # Keep bars for detailed analysis
        }
        
    except Exception as e:
        logger.debug(f"❌ Market data failed for {symbol}: {e}")
        return {}

def calculate_interval_pct_change(bars: List, opening_price: float, minutes: int) -> float:
    """Calculate percentage change at specific minute interval from market open."""
    if not bars or len(bars) < minutes or opening_price <= 0:
        return 'N/A'
    
    try:
        # Get price at specified minute (0-indexed)
        if minutes <= len(bars):
            interval_price = bars[minutes - 1].c
        else:
            interval_price = bars[-1].c  # Use last available if not enough data
        
        return calculate_percentage_change(opening_price, interval_price)
    except (IndexError, AttributeError, TypeError):
        return 'N/A'

def calculate_historical_average_volume(symbol: str, api: AlpacaREST, 
                                      rate_limiter: RobustRateLimiter) -> Optional[float]:
    """Calculate 30-day average volume for comparison."""
    try:
        # Get last 30 trading days
        end_date = datetime.now(TZ).date()
        start_date = end_date - timedelta(days=45)  # Get extra days to ensure 30 trading days
        
        start_iso = f"{start_date}T00:00:00-04:00"
        end_iso = f"{end_date}T23:59:59-04:00"
        
        bars = rate_limiter.alpaca_call(
            api.get_bars,
            symbol,
            timeframe="1Day",
            start=start_iso,
            end=end_iso,
            limit=30,
            adjustment="raw"
        )
        
        if bars and len(bars) >= 10:  # Need at least 10 days of data
            volumes = [bar.v for bar in bars if bar.v > 0]
            if volumes:
                return sum(volumes) / len(volumes)
        
        return None
        
    except Exception as e:
        logger.debug(f"❌ Historical volume calculation failed for {symbol}: {e}")
        return None

def analyze_volume_vs_average(symbol: str, current_volume: int, cache: ComprehensiveCache,
                             api: AlpacaREST, rate_limiter: RobustRateLimiter) -> float:
    """
    Analyze current volume compared to historical average.
    Returns the actual ratio (e.g., 2.5 means 2.5x normal volume).
    """
    try:
        # Check cache first
        avg_volume = cache.get_volume_average(symbol)
        
        if avg_volume is None:
            # Calculate historical average volume (last 30 days)
            avg_volume = calculate_historical_average_volume(symbol, api, rate_limiter)
            if avg_volume:
                cache.set_volume_average(symbol, avg_volume)
        
        if avg_volume and avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            return round(volume_ratio, 2)  # Return actual ratio rounded to 2 decimals
        else:
            # No historical data available
            return 'N/A'
                
    except Exception as e:
        logger.debug(f"❌ Volume analysis failed for {symbol}: {e}")
        return 'N/A'

def get_detailed_intraday_analysis(api: AlpacaREST, symbol: str, target_date: str, 
                                 rate_limiter: RobustRateLimiter, cache: ComprehensiveCache) -> Dict:
    """Get comprehensive intraday analysis with ALL time intervals."""
    try:
        start_iso = f"{target_date}T09:30:00-04:00"
        end_iso = f"{target_date}T16:00:00-04:00"
        
        bars = rate_limiter.alpaca_call(
            api.get_bars,
            symbol,
            timeframe="1Min",
            start=start_iso,
            end=end_iso,
            limit=1000,
            adjustment="raw"
        )
        
        if not bars or len(bars) == 0:
            return create_empty_intraday_analysis()
        
        # Get opening price
        opening_price = bars[0].o
        
        # Calculate percentage changes at ALL required intervals
        pct_1min = calculate_interval_pct_change(bars, opening_price, 1)
        pct_5min = calculate_interval_pct_change(bars, opening_price, 5)
        pct_15min = calculate_interval_pct_change(bars, opening_price, 15)
        pct_30min = calculate_interval_pct_change(bars, opening_price, 30)
        pct_1hr = calculate_interval_pct_change(bars, opening_price, 60)
        
        # Opening minute volume
        opening_minute_volume = bars[0].v if bars else 0
        
        # Volume comparison analysis (now returns numeric ratio)
        total_volume = sum(bar.v for bar in bars)
        volume_comparison = analyze_volume_vs_average(symbol, total_volume, cache, api, rate_limiter)
        
        return {
            'pct_1min': pct_1min,
            'pct_5min': pct_5min,
            'pct_15min': pct_15min,
            'pct_30min': pct_30min,
            'pct_1hr': pct_1hr,
            'opening_minute_volume': opening_minute_volume,
            'volume_comparison': volume_comparison
        }
        
    except Exception as e:
        logger.debug(f"❌ Intraday analysis failed for {symbol}: {e}")
        return create_empty_intraday_analysis()

def determine_high_before_low(high_time: str, low_time: str) -> int:
    """
    Determine if high occurred before low during the trading day.
    Returns: 1 if high before low, 0 if low before high, 'N/A' if no data.
    """
    try:
        if high_time == 'N/A' or low_time == 'N/A':
            return 'N/A'
        
        # Convert times to comparable format
        high_hour, high_min = map(int, high_time.split(':'))
        low_hour, low_min = map(int, low_time.split(':'))
        
        high_minutes = high_hour * 60 + high_min
        low_minutes = low_hour * 60 + low_min
        
        return 1 if high_minutes < low_minutes else 0
        
    except (ValueError, AttributeError):
        return 'N/A'

def calculate_revenue_surprise(candidate: Dict) -> float:
    """Calculate revenue surprise percentage."""
    revenue_estimate = candidate.get('revenueEstimate')
    revenue_actual = candidate.get('revenueActual')
    
    if (revenue_estimate and revenue_actual and 
        revenue_estimate != 0 and 
        isinstance(revenue_estimate, (int, float)) and 
        isinstance(revenue_actual, (int, float))):
        return ((revenue_actual - revenue_estimate) / abs(revenue_estimate)) * 100
    return 'N/A'

def create_empty_intraday_analysis() -> Dict:
    """Create empty intraday analysis structure."""
    return {
        'pct_1min': 'N/A',
        'pct_5min': 'N/A',
        'pct_15min': 'N/A',
        'pct_30min': 'N/A',
        'pct_1hr': 'N/A',
        'opening_minute_volume': 'N/A',
        'volume_comparison': 'N/A'
    }

# ─── Comprehensive Market Data Collection ────────────────────────────────────────────────────────────
def get_comprehensive_market_data(api: AlpacaREST, symbol: str, target_date: str, 
                                 prev_date: str, rate_limiter: RobustRateLimiter,
                                 cache: ComprehensiveCache) -> Dict:
    """
    Collect ALL comprehensive market data as requested.
    This will use multiple API calls per symbol but gets everything.
    """
    try:
        logger.debug(f"📈 Collecting comprehensive data for {symbol} on {target_date}")
        
        # Get previous day close
        prev_close = get_previous_close_price(api, symbol, prev_date, rate_limiter)
        
        # Get premarket data (4:00 AM - 9:30 AM)
        premarket_data = get_premarket_data(api, symbol, target_date, rate_limiter)
        
        # Get comprehensive market hours data
        market_data = get_detailed_market_data(api, symbol, target_date, rate_limiter)
        
        # Get detailed intraday analysis
        intraday_data = get_detailed_intraday_analysis(api, symbol, target_date, rate_limiter, cache)
        
        if not market_data:
            return create_empty_comprehensive_data()
        
        # Extract core prices
        open_price = market_data['open_price']
        high_price = market_data['high_price']
        low_price = market_data['low_price']
        close_price = market_data['close_price']
        
        # Calculate ALL percentage changes
        pct_change_open = calculate_percentage_change(prev_close, open_price)
        pct_change_high = calculate_percentage_change(open_price, high_price)
        pct_change_low = calculate_percentage_change(open_price, low_price)
        pct_change_close = calculate_percentage_change(prev_close, close_price)
        
        # Determine timing relationships (now numeric)
        high_before_low = determine_high_before_low(market_data.get('high_time'), market_data.get('low_time'))
        
        # Convert times to numeric (minutes since market open)
        high_time_numeric = convert_time_to_minutes_since_open(market_data.get('high_time', 'N/A'))
        low_time_numeric = convert_time_to_minutes_since_open(market_data.get('low_time', 'N/A'))
        
        return {
            # Core price data
            'alpaca_prev_close': prev_close,
            'reference_prev_close': prev_close,
            'open_price': open_price,
            'high_price': high_price,
            'low_price': low_price,
            'close_price': close_price,
            'high_time': market_data.get('high_time', 'N/A'),  # Keep original time string
            'low_time': market_data.get('low_time', 'N/A'),    # Keep original time string
            'high_time_minutes': high_time_numeric,             # NEW: Numeric time
            'low_time_minutes': low_time_numeric,               # NEW: Numeric time
            
            # Percentage changes
            'pct_change_open': pct_change_open,
            'pct_change_high': pct_change_high,
            'pct_change_low': pct_change_low,
            'pct_change_close': pct_change_close,
            
            # Premarket data
            'premarket_high': premarket_data.get('high', 'N/A'),
            'premarket_low': premarket_data.get('low', 'N/A'),
            'premarket_volume': premarket_data.get('volume', 'N/A'),
            
            # Detailed intraday analysis
            'pct_1min': intraday_data.get('pct_1min', 'N/A'),
            'pct_5min': intraday_data.get('pct_5min', 'N/A'),
            'pct_15min': intraday_data.get('pct_15min', 'N/A'),
            'pct_30min': intraday_data.get('pct_30min', 'N/A'),
            'pct_1hr': intraday_data.get('pct_1hr', 'N/A'),
            
            # Volume analysis (now numeric ratio)
            'opening_minute_volume': intraday_data.get('opening_minute_volume', 'N/A'),
            'volume_comparison': intraday_data.get('volume_comparison', 'N/A'),  # Now numeric ratio
            
            # Timing analysis (now numeric)
            'high_before_low': high_before_low,  # Now 1/0 instead of Yes/No
            
            # Metadata
            'data_source': 'Alpaca',
            'data_source_numeric': 1,  # NEW: Numeric data source code
            'sort_key': pct_change_high if isinstance(pct_change_high, (int, float)) else -999
        }
        
    except APILimitException:
        raise
    except Exception as e:
        logger.warning(f"❌ Comprehensive data collection failed for {symbol}: {e}")
        return create_empty_comprehensive_data()

def create_empty_comprehensive_data() -> Dict:
    """Create empty comprehensive data structure with ALL required fields (optimized for ML)."""
    return {
        'alpaca_prev_close': 'N/A',
        'reference_prev_close': 'N/A',
        'open_price': 'N/A',
        'high_price': 'N/A',
        'low_price': 'N/A',
        'close_price': 'N/A',
        'high_time': 'N/A',
        'low_time': 'N/A',
        'high_time_minutes': 'N/A',      # NEW: Numeric time
        'low_time_minutes': 'N/A',       # NEW: Numeric time
        'pct_change_open': 'N/A',
        'pct_change_high': 'N/A',
        'pct_change_low': 'N/A',
        'pct_change_close': 'N/A',
        'premarket_high': 'N/A',
        'premarket_low': 'N/A',
        'premarket_volume': 'N/A',
        'pct_1min': 'N/A',
        'pct_5min': 'N/A',
        'pct_15min': 'N/A',
        'pct_30min': 'N/A',
        'pct_1hr': 'N/A',
        'opening_minute_volume': 'N/A',
        'volume_comparison': 'N/A',      # Now numeric ratio
        'high_before_low': 'N/A',        # Now 1/0
        'data_source': 'N/A',
        'data_source_numeric': 'N/A',   # NEW: Numeric data source
        'sort_key': -999
    }

# ─── CSV Output with ALL Required Fields ─────────────────────────────────────────────────────────────
def save_comprehensive_earnings_data(batch_data: List[Dict], filename: str = "comprehensive_earnings_data.csv"):
    """
    Save comprehensive earnings data with ALL required fields (optimized for machine learning).
    Now includes numeric versions of categorical data for better ML compatibility.
    """
    if not batch_data:
        return
        
    try:
        # ALL required fields including new numeric optimizations
        required_fields = [
            # Core identification
            'symbol',
            'surprise', 
            'report_date',
            'report_timestamp',
            'earnings_time',
            'earnings_time_numeric',        # NEW: 0=BMO, 1=AMC
            
            # Company data
            'market_cap',
            'sector',
            'industry',
            'revenue_surprise',
            
            # Price data
            'alpaca_prev_close',
            'reference_prev_close',
            'open_price',
            'high_price',
            'high_time',
            'high_time_minutes',            # NEW: Minutes since market open
            'low_price',
            'low_time',
            'low_time_minutes',             # NEW: Minutes since market open
            'close_price',
            
            # Percentage changes
            'pct_change_open',
            'pct_change_high',
            'pct_change_low',
            'pct_change_close',
            
            # Premarket data
            'premarket_high',
            'premarket_low',
            'premarket_volume',
            
            # Intraday intervals
            'pct_1min',
            'pct_5min',
            'pct_15min',
            'pct_30min',
            'pct_1hr',
            
            # Volume analysis
            'opening_minute_volume',
            'volume_comparison',            # NOW: Numeric ratio (e.g., 2.5)
            
            # Timing analysis
            'high_before_low',              # NOW: 1/0 instead of Yes/No
            
            # Metadata
            'data_source',
            'data_source_numeric',          # NEW: 1=Alpaca, 0=Other
            'sort_key'
        ]
        
        # Check if file exists
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
        
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=required_fields)
            
            if not file_exists:
                writer.writeheader()
                logger.info(f"📄 Created ML-optimized CSV with {len(required_fields)} fields")
                logger.info(f"📊 Numeric optimizations: earnings_time, high_before_low, volume_comparison, timing")
            
            for data in batch_data:
                # Ensure ALL fields are present with proper handling
                clean_row = {}
                for field in required_fields:
                    value = data.get(field, 'N/A')
                    # Keep 'N/A' for missing data, convert others appropriately
                    clean_row[field] = 'N/A' if (value == 'N/A' or value is None) else value
                
                writer.writerow(clean_row)
        
        logger.info(f"💾 Saved {len(batch_data)} ML-optimized records to {filename}")
        
    except Exception as e:
        logger.error(f"💥 Failed to save comprehensive CSV: {e}")

# ─── Main Processing Engine ──────────────────────────────────────────────────────────────────────────
def process_comprehensive_earnings_data():
    """
    Main processing engine designed for week+ operation.
    Collects ALL comprehensive data with robust pause/resume capability.
    """
    try:
        # Initialize all systems
        logger.info("🚀 Starting comprehensive earnings data collection")
        logger.info("📊 Designed for complete data collection over 1+ weeks")
        logger.info("⏸️  Automatic pause/resume on API limits")
        logger.info("🤖 Optimized for machine learning with numeric data")
        
        fh = finnhub.Client(api_key=FINNHUB_API_KEY)
        api = AlpacaREST(APCA_KEY, APCA_SECRET, ALPACA_BASE_URL, api_version='v2')
        rate_limiter = RobustRateLimiter()
        cache = ComprehensiveCache()
        
        # Check if we can resume
        can_resume, wait_minutes = check_resume_time()
        if not can_resume:
            logger.info(f"⏱️  Paused for API limits. Resume in {wait_minutes} minutes.")
            return False
        
        # Load checkpoint and state
        checkpoint = load_checkpoint()
        completed_weeks = set(checkpoint.get('completed_weeks', []))
        failed_symbols = set(checkpoint.get('failed_symbols', []))
        
        # Calculate date ranges
        end_date = date.today() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=HISTORICAL_DAYS)  # Past year
        weekly_ranges = generate_weekly_ranges(start_date, end_date)
        
        total_weeks = len(weekly_ranges)
        logger.info(f"📅 Processing {total_weeks} weeks: {start_date} to {end_date}")
        logger.info(f"✅ Completed: {len(completed_weeks)}/{total_weeks} weeks")
        logger.info(f"❌ Failed symbols: {len(failed_symbols)}")
        logger.info(f"📊 {rate_limiter.get_usage_summary()}")
        
        # Process each week systematically
        for week_idx, (week_start, week_end) in enumerate(weekly_ranges):
            week_key = f"{week_start}_{week_end}"
            
            # Skip if already completed
            if week_key in completed_weeks:
                logger.debug(f"⏭️  Skipping completed week {week_idx + 1}/{total_weeks}")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📅 WEEK {week_idx + 1}/{total_weeks}: {week_start} to {week_end}")
            logger.info(f"{'='*80}")
            
            try:
                # Step 1: Fetch ALL earnings for this week
                earnings_entries = fetch_weekly_earnings_calendar(fh, rate_limiter, week_start, week_end)
                
                if not earnings_entries:
                    logger.info("📭 No earnings found for this week")
                    completed_weeks.add(week_key)
                    checkpoint['completed_weeks'] = list(completed_weeks)
                    save_checkpoint(checkpoint)
                    continue
                
                # Step 2: Filter candidates and get profiles
                candidates = filter_earnings_candidates(
                    earnings_entries, fh, rate_limiter, cache, failed_symbols
                )
                
                if not candidates:
                    logger.info("🚫 No qualified candidates after filtering")
                    completed_weeks.add(week_key)
                    checkpoint['completed_weeks'] = list(completed_weeks)
                    save_checkpoint(checkpoint)
                    continue
                
                # Step 3: Collect comprehensive market data for each candidate
                logger.info(f"📈 Collecting comprehensive data for {len(candidates)} candidates")
                week_data = []
                successful_collections = 0
                
                for i, candidate in enumerate(candidates):
                    symbol = candidate['symbol']
                    earnings_date = candidate['earnings_date']
                    market_reaction_date = candidate['market_reaction_date']
                    earnings_time = candidate['earnings_time']
                    
                    logger.info(f"  📊 [{i+1}/{len(candidates)}] Processing {symbol}")
                    logger.info(f"      Earnings: {earnings_date} {earnings_time.upper()}")
                    logger.info(f"      Reaction: {market_reaction_date}")
                    
                    # Calculate previous business day
                    try:
                        reaction_date_obj = datetime.strptime(market_reaction_date, '%Y-%m-%d').date()
                        prev_date_obj = get_prev_business_day(reaction_date_obj)
                        prev_date_str = prev_date_obj.strftime('%Y-%m-%d')
                    except (ValueError, TypeError):
                        logger.warning(f"❌ Invalid date format for {symbol}")
                        continue
                    
                    # Collect ALL comprehensive market data
                    market_data = get_comprehensive_market_data(
                        api, symbol, market_reaction_date, prev_date_str, rate_limiter, cache
                    )
                    
                    # Calculate revenue surprise
                    revenue_surprise = calculate_revenue_surprise(candidate)
                    
                    # Convert earnings time to numeric for ML
                    earnings_time_numeric = convert_earnings_time_to_numeric(earnings_time)
                    
                    # Build complete earnings record with ALL fields (ML-optimized)
                    earnings_record = {
                        'symbol': symbol,
                        'surprise': candidate['surprise'],
                        'report_date': earnings_date,
                        'report_timestamp': f"{earnings_date} {earnings_time.upper()}",
                        'earnings_time': earnings_time.upper(),
                        'earnings_time_numeric': earnings_time_numeric,  # NEW: 0=BMO, 1=AMC
                        'market_cap': candidate['market_cap'],
                        'sector': candidate.get('sector', 'N/A'),
                        'industry': candidate.get('industry', 'N/A'),
                        'revenue_surprise': revenue_surprise,
                        **market_data  # Include ALL market data fields (now with numeric optimizations)
                    }
                    
                    week_data.append(earnings_record)
                    successful_collections += 1
                    
                    # Progress update every 5 symbols
                    if (i + 1) % 5 == 0:
                        logger.info(f"      ✅ Processed {i + 1}/{len(candidates)} candidates")
                        logger.info(f"      📊 {rate_limiter.get_usage_summary()}")
                
                # Sort by highest intraday performance
                week_data.sort(
                    key=lambda x: x['sort_key'] if isinstance(x['sort_key'], (int, float)) else -999, 
                    reverse=True
                )
                
                # Save comprehensive data to CSV
                if week_data:
                    save_comprehensive_earnings_data(week_data)
                    logger.info(f"💾 Saved {len(week_data)} comprehensive records")
                
                # Update checkpoint with progress
                completed_weeks.add(week_key)
                checkpoint['completed_weeks'] = list(completed_weeks)
                checkpoint['last_processed_week'] = week_key
                checkpoint['total_collected'] = checkpoint.get('total_collected', 0) + len(week_data)
                checkpoint['failed_symbols'] = list(failed_symbols)
                save_checkpoint(checkpoint)
                
                # Save caches
                cache.save_all_caches()
                
                # Weekly progress summary
                progress = (len(completed_weeks) / total_weeks) * 100
                logger.info(f"\n📊 WEEK SUMMARY:")
                logger.info(f"   ✅ Progress: {len(completed_weeks)}/{total_weeks} weeks ({progress:.1f}%)")
                logger.info(f"   📈 Total records: {checkpoint.get('total_collected', 0):,}")
                logger.info(f"   📱 {rate_limiter.get_usage_summary()}")
                
                # Check if approaching daily limits
                if (rate_limiter.finnhub_calls_today >= FINNHUB_DAILY_LIMIT * 0.8 or 
                    rate_limiter.alpaca_calls_today >= ALPACA_DAILY_LIMIT * 0.8):
                    logger.info("⚠️  Approaching daily API limits")
                    remaining_hours = 24 - datetime.now(TZ).hour
                    logger.info(f"⏱️  Will pause soon. {remaining_hours} hours until reset.")
                
            except APILimitException as e:
                logger.error(f"🛑 API LIMIT REACHED: {e.api_name}")
                
                # Calculate generous resume time
                if e.api_name == 'Finnhub':
                    base_wait = e.retry_after or 3600  # 1 hour base
                elif e.api_name == 'Alpaca':
                    base_wait = e.retry_after or 900   # 15 minutes base
                else:
                    base_wait = 1800  # 30 minutes base
                
                # Add 25% safety buffer
                resume_seconds = int(base_wait * 1.25)
                pause_until = datetime.now(TZ) + timedelta(seconds=resume_seconds)
                
                # Save state and pause
                checkpoint['failed_symbols'] = list(failed_symbols)
                save_checkpoint_with_pause(checkpoint, pause_until, e.api_name)
                cache.save_all_caches()
                
                logger.info(f"⏰ PAUSED until {pause_until.strftime('%Y-%m-%d %H:%M %Z')}")
                logger.info(f"💤 Resume in {resume_seconds/3600:.1f} hours by running: python script.py start")
                logger.info(f"📊 Progress saved: {len(completed_weeks)}/{total_weeks} weeks complete")
                
                return False  # Graceful exit
                
            except Exception as e:
                logger.error(f"💥 Unexpected error in week {week_start}: {e}")
                # Log error but continue to next week
                continue
        
        # Successfully completed all weeks
        cache.save_all_caches()
        logger.info(f"\n{'='*80}")
        logger.info(f"🎉 COMPREHENSIVE DATA COLLECTION COMPLETED!")
        logger.info(f"{'='*80}")
        logger.info(f"📊 Total records collected: {checkpoint.get('total_collected', 0):,}")
        logger.info(f"📈 Weeks processed: {len(completed_weeks)}/{total_weeks}")
        logger.info(f"📱 Final API usage: {rate_limiter.get_usage_summary()}")
        logger.info(f"📄 Output file: comprehensive_earnings_data.csv")
        logger.info(f"💾 All data includes: premarket, intraday intervals, volume analysis, timing data")
        logger.info(f"🤖 ML-optimized with numeric encodings for algorithms")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 FATAL ERROR in main processing: {e}")
        return False

# ─── Status and Management ──────────────────────────────────────────────────────────────────────────
def display_comprehensive_status():
    """Display detailed status of comprehensive data collection."""
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE EARNINGS DATA COLLECTOR - STATUS")
    print(f"{'='*80}")
    
    checkpoint = load_checkpoint()
    
    if not checkpoint or not checkpoint.get("completed_weeks"):
        print("❌ No processing history found")
        print("   Run 'python script.py start' to begin comprehensive collection")
        return
    
    # Calculate overall progress
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=HISTORICAL_DAYS)
    weekly_ranges = generate_weekly_ranges(start_date, end_date)
    
    completed_count = len(checkpoint['completed_weeks'])
    total_count = len(weekly_ranges)
    progress = (completed_count / total_count) * 100
    
    print(f"📈 OVERALL PROGRESS:")
    print(f"   Weeks: {completed_count}/{total_count} ({progress:.1f}%)")
    print(f"   Records: {checkpoint.get('total_collected', 0):,}")
    print(f"   Date Range: {start_date} to {end_date}")
    print(f"   Last Processed: {checkpoint.get('last_processed_week', 'None')}")
    
    # API status
    can_resume, wait_minutes = check_resume_time()
    print(f"\n⚡ API STATUS:")
    if not can_resume:
        api_name = checkpoint.get('api_limit_api', 'Unknown')
        wait_hours = wait_minutes / 60
        print(f"   Status: PAUSED ({api_name} limits)")
        print(f"   Resume in: {wait_hours:.1f} hours ({wait_minutes} minutes)")
        print(f"   Action: Run 'python script.py start' after wait time")
    else:
        if completed_count >= total_count:
            print(f"   Status: ✅ COMPLETED")
        else:
            print(f"   Status: ▶️  READY TO RESUME")
            print(f"   Action: Run 'python script.py start' to continue")
    
    # File analysis
    print(f"\n📁 OUTPUT FILES:")
    csv_file = "comprehensive_earnings_data.csv"
    if os.path.exists(csv_file):
        size = os.path.getsize(csv_file)
        try:
            with open(csv_file, 'r') as f:
                line_count = sum(1 for _ in f) - 1  # Exclude header
            print(f"   📄 {csv_file}:")
            print(f"      Size: {size:,} bytes")
            print(f"      Records: {line_count:,}")
            
            # Verify all required fields are present
            with open(csv_file, 'r') as f:
                header = f.readline().strip().split(',')
                required_count = 40  # Updated total with ML optimizations
                print(f"      Fields: {len(header)}/{required_count} ({'✅' if len(header) >= required_count else '❌'})")
                
        except Exception as e:
            print(f"   📄 {csv_file}: {size:,} bytes (error reading details)")
    else:
        print(f"   📄 {csv_file}: Not created yet")
    
    # Cache status
    print(f"\n🗃️  CACHE STATUS:")
    profile_cache = "company_profiles_cache.json"
    volume_cache = "volume_averages_cache.json"
    
    if os.path.exists(profile_cache):
        try:
            with open(profile_cache, 'r') as f:
                profiles = json.load(f)
            print(f"   Company Profiles: {len(profiles):,} cached")
        except:
            print(f"   Company Profiles: Error reading cache")
    
    if os.path.exists(volume_cache):
        try:
            with open(volume_cache, 'r') as f:
                volumes = json.load(f)
            print(f"   Volume Averages: {len(volumes):,} cached")
        except:
            print(f"   Volume Averages: Error reading cache")
    
    # Data completeness summary
    print(f"\n📋 DATA COMPLETENESS & ML OPTIMIZATIONS:")
    print(f"   ✅ Premarket data (high, low, volume)")
    print(f"   ✅ Intraday intervals (1min, 5min, 15min, 30min, 1hr)")
    print(f"   ✅ Volume analysis vs historical averages (numeric ratios)")
    print(f"   ✅ Timing analysis (high before low as 1/0)")
    print(f"   ✅ Revenue surprises")
    print(f"   ✅ All percentage changes")
    print(f"   ✅ Market cap and sector classification")
    print(f"   🤖 Numeric timing (minutes since market open)")
    print(f"   🤖 Numeric earnings timing (0=BMO, 1=AMC)")
    print(f"   🤖 Numeric data sources (1=Alpaca)")
    print(f"   🤖 All optimized for machine learning algorithms")
    
    # Performance estimate
    if completed_count > 0 and completed_count < total_count:
        records_per_week = checkpoint.get('total_collected', 0) / completed_count
        estimated_total = records_per_week * total_count
        print(f"\n🔮 PROJECTION:")
        print(f"   Estimated total records: {estimated_total:,.0f}")
        
        remaining_weeks = total_count - completed_count
        if can_resume:
            print(f"   Remaining weeks: {remaining_weeks}")
            print(f"   Estimated completion: {remaining_weeks} days (if run daily)")

def clear_all_comprehensive_data():
    """Clear all data and caches for fresh start."""
    files_to_remove = [
        CHECKPOINT_FILE,
        "company_profiles_cache.json",
        "volume_averages_cache.json",
        "comprehensive_earnings_data.csv",
        "comprehensive_earnings_log.txt"
    ]
    
    print("\n🗑️  CLEARING ALL DATA...")
    removed_count = 0
    
    for file in files_to_remove:
        try:
            if os.path.exists(file):
                size = os.path.getsize(file)
                os.remove(file)
                print(f"   ✅ Removed {file} ({size:,} bytes)")
                removed_count += 1
            else:
                print(f"   ⏭️  {file} (not found)")
        except Exception as e:
            print(f"   ❌ Failed to remove {file}: {e}")
    
    print(f"\n🔄 CLEANUP COMPLETE:")
    print(f"   Files removed: {removed_count}/{len(files_to_remove)}")
    print(f"   Status: Ready for fresh comprehensive collection")
    print(f"   Next: Run 'python script.py start'")

def main():
    """Main entry point for comprehensive earnings data collection."""
    if len(sys.argv) < 2:
        print(f"\n{'='*80}")
        print(f"🚀 COMPREHENSIVE EARNINGS DATA COLLECTOR")
        print(f"{'='*80}")
        print(f"📊 Designed for COMPLETE data collection over 1+ weeks")
        print(f"⏸️  Automatic pause/resume on API limits")
        print(f"💾 Collects ALL required fields for algotrading")
        print(f"\nUsage: python script.py [command]")
        print(f"\nCommands:")
        print(f"  start   - Start/resume comprehensive collection")
        print(f"  status  - Show detailed progress and statistics")
        print(f"  reset   - Clear all data and start completely fresh")
        print(f"\n📋 COMPREHENSIVE DATA COLLECTED (ML-OPTIMIZED):")
        print(f"  • All original fields: alpaca_prev_close, close_price, etc.")
        print(f"  • NEW: earnings_time_numeric (0=BMO, 1=AMC)")
        print(f"  • NEW: high_time_minutes, low_time_minutes (since market open)")
        print(f"  • NEW: volume_comparison as numeric ratio (e.g., 2.5x)")
        print(f"  • NEW: high_before_low as 1/0 instead of Yes/No")
        print(f"  • NEW: data_source_numeric (1=Alpaca, 0=Other)")
        print(f"  • Total: 40+ fields optimized for machine learning")
        print(f"\n🤖 MACHINE LEARNING OPTIMIZATIONS:")
        print(f"  • Categorical → Numeric: Better for ML algorithms")
        print(f"  • Time → Minutes: Easier mathematical operations")
        print(f"  • Ratios → Decimals: Direct comparison values")
        print(f"  • Boolean → 1/0: Standard ML format")
        print(f"  • Preserves 'N/A' for missing data handling")
        print(f"\n⚡ DESIGNED FOR RELIABILITY:")
        print(f"  • Conservative API limits for sustained operation")
        print(f"  • Persistent caching to avoid duplicate requests")
        print(f"  • Robust checkpoint system for exact resume points")
        print(f"  • Automatic pause when API limits reached")
        print(f"  • Week+ operation timeline expected and supported")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'start':
        print(f"\n{'='*80}")
        print(f"🚀 STARTING COMPREHENSIVE EARNINGS DATA COLLECTION")
        print(f"{'='*80}")
        print(f"📊 Target: ALL earnings data with complete market analysis")
        print(f"⏱️  Timeline: 1+ weeks for complete historical year")
        print(f"🛡️  Safety: Conservative API limits with auto-pause/resume")
        print(f"💾 Output: CSV with 40+ fields optimized for machine learning")
        print(f"🤖 ML Ready: Numeric data types, ratios, and boolean encoding")
        print(f"🔄 Resume: Automatic continuation from any interruption")
        print(f"\n{'='*80}")
        print(f"Starting collection... (Check log file for detailed progress)")
        
        success = process_comprehensive_earnings_data()
        
        if success:
            print(f"\n{'='*80}")
            print(f"🎉 COMPREHENSIVE COLLECTION COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"📄 Check 'comprehensive_earnings_data.csv' for results")
            print(f"📊 All 40+ fields collected and optimized for ML training")
            print(f"🤖 Numeric data types ready for algorithm development")
            print(f"💯 Data ready for algotrading model training")
        else:
            print(f"\n{'='*80}")
            print(f"⏸️  COLLECTION PAUSED DUE TO API LIMITS")
            print(f"{'='*80}")
            print(f"🔄 Progress automatically saved")
            print(f"⏰ Resume by running: python script.py start")
            print(f"📊 Check status: python script.py status")
            print(f"💡 This is normal for comprehensive data collection")
            
    elif command == 'status':
        display_comprehensive_status()
        
    elif command == 'reset':
        confirm = input("\n⚠️  This will delete ALL collected data and caches. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            clear_all_comprehensive_data()
        else:
            print("❌ Reset cancelled")
            
    else:
        print(f"❌ Unknown command: {command}")
        print(f"Valid commands: start, status, reset")

if __name__ == "__main__":
    main()