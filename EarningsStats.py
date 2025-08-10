#!/usr/bin/env python3
"""
EarningsStats.py - Optimized bin & SL/TP optimizer for EarningsBot handoff

Reads an appended earningsreport.txt (format shown in user's message),
finds quantile-based opening-% bins, tests for significance vs rest,
requires bin mean >= GOAL_RETURN, vectorized SL/TP grid search,
computes recommended daily purchase count, and writes output JSON
for EarningsBot.py to consume.

Usage:
    python EarningsStats.py
Outputs:
    earnings_bins.json   -- main output file used by EarningsBot.py


Date: 2025-08-08
"""
from __future__ import annotations
import re
import json
import math
import time
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps

# -------------------------
# CONFIGURATION - TUNE THESE
# -------------------------
# Goal: segments must have mean IntraDayReturn >= GOAL_RETURN (%) to be considered
GOAL_RETURN = 0.1  # percent

# Statistical settings
MIN_SIGNIFICANCE_LEVEL = 0.05
MIN_DATA_POINTS_STATS = 10
OUTLIER_STD_THRESHOLD = 3.0

# Performance thresholds for recommended segments
MIN_TRADES_PER_SEGMENT = 30
MIN_WIN_RATE = 0.45
MIN_SHARPE_RATIO = 0.5

# Stop/Target grid (percent)
STOP_LOSS_MIN = -10.0
STOP_LOSS_MAX = -0.1
STOP_LOSS_STEP = 0.25

TARGET_PROFIT_MIN = 2.0
TARGET_PROFIT_MAX = 30.0
TARGET_PROFIT_STEP = 0.25

MIN_RISK_REWARD_RATIO = 0.4

# Segment settings
DEFAULT_NUM_SEGMENTS = 8
MAX_WORKERS = 6

# Sensitivity analysis (used for metadata & optional robustness)
SENSITIVITY_VARIATIONS = 5
SENSITIVITY_RANGE = 0.2
ROBUSTNESS_THRESHOLD = 0.2

# I/O
INPUT_FILE = "earningsreport.txt"
OUTPUT_JSON = "earnings_bins.json"

# API placeholders (if you plug APIs later)
ALPACA_RATE_LIMIT_PER_MIN = 200  # adjust to your account
FINNHUB_RATE_LIMIT_PER_MIN = 60

# -------------------------
# Data classes
# -------------------------
@dataclass
class SegmentResult:
    range_min: float
    range_max: float
    stop_loss: float
    target_profit: float
    win_rate: float
    avg_return: float
    trade_count: int
    sharpe_ratio: float
    effect_size: float
    p_value: float
    segment_mean: float
    is_robust: Optional[bool] = None

# -------------------------
# Utility: simple rate limiter decorator for optional API wrappers
# -------------------------
def rate_limited(wait_seconds: float):
    """Simple sleep-based rate limiter decorator for non-concurrent API calls."""
    def deco(func):
        last = {"t": 0.0}
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last["t"]
            to_sleep = max(0.0, wait_seconds - elapsed)
            if to_sleep > 0:
                time.sleep(to_sleep)
            last["t"] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return deco

# Example placeholders (no network calls done by default)
@lru_cache(maxsize=2048)
@rate_limited(60.0 / FINNHUB_RATE_LIMIT_PER_MIN)
def fetch_finnhub_stub(ticker: str) -> Dict:
    """Stub - replace if you need to fetch intraday metrics. Currently unused."""
    # Return an empty dict to indicate no network call in default flow
    return {}

@lru_cache(maxsize=2048)
@rate_limited(60.0 / ALPACA_RATE_LIMIT_PER_MIN)
def fetch_alpaca_stub(ticker: str) -> Dict:
    """Stub for Alpaca; replace body with actual calls if needed."""
    return {}

# -------------------------
# Parsing logic for earningsreport.txt format
# -------------------------
def parse_earningsreport_text(path: str) -> pd.DataFrame:
    """
    Parse the earningsreport.txt file that contains multiple appended reports.
    Expected block format (header lines present) as in the user's example.

    Returns a DataFrame with columns:
      Symbol, Surprise, Src, PrevCls, Open, PercentOpen, High, HTime, Low, LTime, Close, PercentHigh, PercentLow, PercentClose, Date
    Percent columns are numeric percentages (no trailing '%').
    """
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()

    # Split by report blocks using the big separator lines that repeat
    blocks = re.split(r"\n=+\n", content)
    rows = []

    # Regex for header line that contains "EARNINGS REPORT (... ) - YYYY-MM-DD hh:mm:ss TZ"
    header_re = re.compile(r"EARNINGS REPORT .* - (\d{4}-\d{2}-\d{2})")

    # For each block, find data table lines
    for block in blocks:
        if not block.strip():
            continue
            
        # Try to find the report date within the block
        m = header_re.search(block)
        block_date_str = m.group(1) if m else None

        lines = block.splitlines()
        
        # Find the header table start (line containing 'Symbol' and 'Surprise' and 'PrevCls' etc)
        header_idx = None
        for i, line in enumerate(lines):
            if "Symbol" in line and "PrevCls" in line and "%Open" in line:
                header_idx = i
                break
                
        if header_idx is None:
            continue

        # Data lines start after dashed line (---)
        data_start = None
        for j in range(header_idx + 1, min(len(lines), header_idx + 6)):
            if re.match(r"^-{3,}", lines[j].strip()):
                data_start = j + 1
                break
                
        if data_start is None:
            # if dashed line not found, assume header+1
            data_start = header_idx + 1

        # Process each data line
        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue
                
            # Stop at report summary or end-of-block markers
            if line.startswith("Report Summary") or line.startswith("===") or line.startswith("--"):
                break
                
            # Parse the line using fixed-width or whitespace splitting
            # The format appears to be space-separated with consistent column alignment
            parts = re.split(r'\s+', line)
            
            if len(parts) < 10:  # Need at least symbol + 9 data fields
                continue
                
            try:
                # Expected format based on header:
                # Symbol   Surprise Src     PrevCls  Open     %Open   High     HTime  Low      LTime  Close    %High   %Low    %Close
                symbol = parts[0]
                surprise = float(parts[1])
                src = parts[2] if parts[2] in ("Alpaca", "Finnhub") else None
                prevcls = float(parts[3])
                open_price = float(parts[4])
                percent_open = float(parts[5])
                high_price = float(parts[6])
                
                # Find time field (HH:MM format) - this helps us locate remaining fields
                time_indices = []
                for i, part in enumerate(parts):
                    if re.match(r'^\d{1,2}:\d{2}$', part):
                        time_indices.append(i)
                
                # Based on the data format, after High comes HTime, then Low, LTime, Close, %High, %Low, %Close
                if len(time_indices) >= 1:
                    htime_idx = time_indices[0]
                    htime = parts[htime_idx]
                    
                    # Low should be right after HTime
                    low_price = float(parts[htime_idx + 1])
                    
                    # LTime should be next time field or next after Low
                    if len(time_indices) >= 2:
                        ltime_idx = time_indices[1]
                        ltime = parts[ltime_idx]
                        close_idx = ltime_idx + 1
                    else:
                        # No LTime found, assume Low is followed by Close
                        ltime = ""
                        close_idx = htime_idx + 2
                    
                    close_price = float(parts[close_idx])
                    
                    # Remaining fields should be %High, %Low, %Close
                    percent_high = float(parts[close_idx + 1]) if close_idx + 1 < len(parts) else np.nan
                    percent_low = float(parts[close_idx + 2]) if close_idx + 2 < len(parts) else np.nan
                    percent_close = float(parts[close_idx + 3]) if close_idx + 3 < len(parts) else np.nan
                    
                else:
                    # Fallback: try to parse without relying on time fields
                    # Assume: Symbol Surprise Src PrevCls Open %Open High HTime Low LTime Close %High %Low %Close
                    htime = parts[7] if len(parts) > 7 else ""
                    low_price = float(parts[8]) if len(parts) > 8 else np.nan
                    ltime = parts[9] if len(parts) > 9 else ""
                    close_price = float(parts[10]) if len(parts) > 10 else np.nan
                    percent_high = float(parts[11]) if len(parts) > 11 else np.nan
                    percent_low = float(parts[12]) if len(parts) > 12 else np.nan
                    percent_close = float(parts[13]) if len(parts) > 13 else np.nan

                row = {
                    "Symbol": symbol,
                    "Surprise": surprise,
                    "Src": src,
                    "PrevCls": prevcls,
                    "Open": open_price,
                    "PercentOpen": percent_open,
                    "High": high_price,
                    "HTime": htime,
                    "Low": low_price,
                    "LTime": ltime,
                    "Close": close_price,
                    "PercentHigh": percent_high,
                    "PercentLow": percent_low,
                    "PercentClose": percent_close,
                    "ReportDate": block_date_str
                }
                
                # Basic validation - skip rows with critical missing data
                if not math.isnan(percent_open) and not math.isnan(percent_close):
                    rows.append(row)
                    
            except (ValueError, IndexError) as e:
                # Skip malformed lines
                print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
                continue

    if not rows:
        print("No valid data rows found!")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    
    print(f"Parsed {len(df)} rows from earnings report")

    # Clean column types
    numeric_cols = ["Surprise", "PrevCls", "Open", "PercentOpen",
                    "High", "Low", "Close", "PercentHigh", "PercentLow", "PercentClose"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

# Compute IntraDayReturn - CRITICAL: must be from OPEN price (your actual entry) to close
    if "Close" in df.columns and "Open" in df.columns:
        # Direct calculation: your actual P&L from buying at open
        df["IntraDayReturn"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0
    elif "PercentClose" in df.columns and "PercentOpen" in df.columns:
        # Convert from prev-close basis to open basis using PercentOpen
        # IntraReturn from open = ((PrevClose * (1 + PercentClose/100)) - (PrevClose * (1 + PercentOpen/100))) / (PrevClose * (1 + PercentOpen/100)) * 100
        # Simplifies to: IntraReturn = (PercentClose - PercentOpen) / (1 + PercentOpen/100)
        open_multiplier = (1 + df["PercentOpen"] / 100.0)
        df["IntraDayReturn"] = (df["PercentClose"] - df["PercentOpen"]) / open_multiplier
    elif "PercentClose" in df.columns:
        # Fallback to prev-close basis (less accurate but better than nothing)
        df["IntraDayReturn"] = df["PercentClose"]
        print("WARNING: Using prev-close basis for IntraDayReturn - results may be less accurate")
    else:
        # Last resort fallback if Close and PrevCls exist
        if {"Close", "PrevCls"}.issubset(df.columns):
            df["IntraDayReturn"] = (df["Close"] - df["PrevCls"]) / df["PrevCls"] * 100.0
            print("WARNING: Using prev-close basis for IntraDayReturn - results may be less accurate")
        else:
            df["IntraDayReturn"] = np.nan

    # MaxGain and MaxLoss used by optimizer - CRITICAL: must be relative to OPEN price for buy-at-open strategy
    # PercentHigh/PercentLow are from previous close, but we need from open price
    if all(col in df.columns for col in ["High", "Open", "Low"]):
        # Calculate intraday max gain/loss from open price (what we actually buy at)
        df["MaxGain"] = (df["High"] - df["Open"]) / df["Open"] * 100.0
        df["MaxLoss"] = (df["Low"] - df["Open"]) / df["Open"] * 100.0
    elif "PercentHigh" in df.columns and "PercentLow" in df.columns and "PercentOpen" in df.columns:
        # Fallback: convert from prev-close basis to open basis using PercentOpen
        # MaxGain from open = ((PrevClose * (1 + PercentHigh/100)) - (PrevClose * (1 + PercentOpen/100))) / (PrevClose * (1 + PercentOpen/100)) * 100
        # Simplifies to: MaxGain = (PercentHigh - PercentOpen) / (1 + PercentOpen/100) * 100
        open_multiplier = (1 + df["PercentOpen"] / 100.0)
        df["MaxGain"] = (df["PercentHigh"] - df["PercentOpen"]) / open_multiplier
        df["MaxLoss"] = (df["PercentLow"] - df["PercentOpen"]) / open_multiplier
    else:
        df["MaxGain"] = np.nan
        df["MaxLoss"] = np.nan

    # Normalize ReportDate into Date column
    if "ReportDate" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["ReportDate"])
        except Exception:
            df["Date"] = pd.NaT
    else:
        df["Date"] = pd.NaT

    # Drop rows missing key metrics
    df = df.dropna(subset=["PercentOpen", "IntraDayReturn"], how="any").reset_index(drop=True)
    
    # Debug: Print some statistics about the parsed data
    print(f"PercentOpen range: {df['PercentOpen'].min():.2f} to {df['PercentOpen'].max():.2f}")
    print(f"IntraDayReturn range: {df['IntraDayReturn'].min():.2f} to {df['IntraDayReturn'].max():.2f}")
    print(f"Positive returns: {(df['IntraDayReturn'] > 0).sum()}, Negative returns: {(df['IntraDayReturn'] < 0).sum()}")
    
    return df

# -------------------------
# Segment identification
# -------------------------
def identify_segments(df: pd.DataFrame, num_segments: int = DEFAULT_NUM_SEGMENTS) -> List[Tuple[float, float]]:
    """
    Create quantile-based segments over PercentOpen after removing outliers.
    Returns list of (min, max) tuples.
    """
    arr = df["PercentOpen"].astype(float)
    mean = arr.mean()
    std = arr.std(ddof=0) if arr.std(ddof=0) > 0 else 0.0
    if std > 0:
        filtered = df[np.abs(df["PercentOpen"] - mean) <= OUTLIER_STD_THRESHOLD * std]
    else:
        filtered = df
    quantiles = np.linspace(0.0, 1.0, num_segments + 1)
    bounds = filtered["PercentOpen"].quantile(quantiles).values
    segments = []
    for i in range(len(bounds) - 1):
        lo, hi = float(bounds[i]), float(bounds[i + 1])
        # avoid degenerate zero-width segments
        if math.isclose(lo, hi):
            # bump hi slightly
            hi = lo + 1e-6
        segments.append((lo, hi))
    return segments

# -------------------------
# Statistical test + goal check
# -------------------------
def test_statistical_significance(df: pd.DataFrame, segment: Tuple[float, float]) -> Dict:
    lo, hi = segment
    mask = (df["PercentOpen"] >= lo) & (df["PercentOpen"] <= hi)
    seg_returns = df.loc[mask, "IntraDayReturn"].dropna().values
    ctrl_returns = df.loc[~mask, "IntraDayReturn"].dropna().values

    if len(seg_returns) < MIN_DATA_POINTS_STATS or len(ctrl_returns) < MIN_DATA_POINTS_STATS:
        return {"significant": False, "p_value": 1.0, "segment_mean": float(np.nan),
                "segment_size": len(seg_returns), "control_size": len(ctrl_returns), "meets_goal": False, "effect_size": 0.0}

    # Two-sample t-test (Welch)
    try:
        t_stat, p_value = stats.ttest_ind(seg_returns, ctrl_returns, equal_var=False, nan_policy="omit")
    except Exception:
        p_value = 1.0

    # Normality check - if non-normal use Mann-Whitney
    use_nonparam = False
    try:
        if len(seg_returns) >= 8 and len(ctrl_returns) >= 8:  # normaltest needs >=8
            _, p_seg = stats.normaltest(seg_returns)
            _, p_ctrl = stats.normaltest(ctrl_returns)
            if p_seg < MIN_SIGNIFICANCE_LEVEL or p_ctrl < MIN_SIGNIFICANCE_LEVEL:
                use_nonparam = True
        else:
            # small samples -> use nonparam safe path
            use_nonparam = True
    except Exception:
        use_nonparam = True

    if use_nonparam:
        try:
            _, p_mw = stats.mannwhitneyu(seg_returns, ctrl_returns, alternative="two-sided")
            p_value = float(min(p_value if not math.isnan(p_value) else 1.0, p_mw if not math.isnan(p_mw) else 1.0))
        except Exception:
            pass

    seg_mean = float(np.mean(seg_returns))
    ctrl_mean = float(np.mean(ctrl_returns)) if len(ctrl_returns) > 0 else 0.0
    is_significant = p_value < MIN_SIGNIFICANCE_LEVEL
    meets_goal = seg_mean >= GOAL_RETURN
    effect_size = seg_mean - ctrl_mean
    return {
        "significant": is_significant,
        "p_value": float(p_value),
        "segment_mean": seg_mean,
        "segment_size": len(seg_returns),
        "control_size": len(ctrl_returns),
        "meets_goal": meets_goal,
        "effect_size": effect_size
    }

# -------------------------
# Vectorized SL/TP optimization
# -------------------------
def optimize_exit_strategy(segment_df: pd.DataFrame) -> Optional[Dict]:
    """
    Vectorized grid search over stop losses and target profits.
    Returns dict with best params (max Sharpe w/ penalties) or None if insufficient rows.
    """
    n = len(segment_df)
    if n < MIN_TRADES_PER_SEGMENT:
        return None

    max_gain = np.asarray(segment_df["MaxGain"].fillna(-np.inf), dtype=float)
    max_loss = np.asarray(segment_df["MaxLoss"].fillna(np.inf), dtype=float)
    final_ret = np.asarray(segment_df["IntraDayReturn"].fillna(0.0), dtype=float)

    stop_losses = np.arange(STOP_LOSS_MIN, STOP_LOSS_MAX + 1e-9, STOP_LOSS_STEP)
    targets = np.arange(TARGET_PROFIT_MIN, TARGET_PROFIT_MAX + 1e-9, TARGET_PROFIT_STEP)

    best_score = float("inf")
    best_stats = None

    # iterate over smaller dimension (grid), but compute exits vectorized per grid cell
    for sl in stop_losses:
        hit_stop_mask = (max_loss <= sl)  # True if stop loss would have been hit
        for tp in targets:
            rr = (tp / abs(sl)) if sl != 0 else 0.0
            if rr < MIN_RISK_REWARD_RATIO:
                continue
            hit_target_mask = (max_gain >= tp)
            # exit returns vectorized:
            exits = np.where(hit_stop_mask, sl, np.where(hit_target_mask, tp, final_ret))
            wins = np.sum(exits > 0)
            losses = np.sum(exits <= 0)
            total = wins + losses
            if total == 0:
                continue
            win_rate = wins / total
            avg_return = float(exits.mean())
            vol = float(np.std(exits, ddof=0))
            sharpe = float((exits.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0)
            # penalize very low win rate
            penalty = 0.0
            if win_rate < 0.4:
                penalty = (0.4 - win_rate) * 10.0
            score = -(sharpe - penalty)  # minimize negative (maximize sharpe-penalty)
            if score < best_score:
                best_score = score
                best_stats = {
                    "stop_loss": float(sl),
                    "target_profit": float(tp),
                    "expected_return": avg_return,
                    "win_rate": float(win_rate),
                    "sharpe_ratio": float(sharpe),
                    "total_trades": int(total),
                    "risk_reward_ratio": float(rr)
                }
    return best_stats

# -------------------------
# Sensitivity (optional robustness check)
# -------------------------
def conduct_sensitivity(segment_df: pd.DataFrame, strategy: SegmentResult, variations: int = SENSITIVITY_VARIATIONS) -> Dict:
    lo, hi = strategy.range_min, strategy.range_max
    seg = segment_df
    if seg.empty:
        return {"is_robust": False, "sharpe_std": float("inf"), "sharpe_min": strategy.sharpe_ratio, "sharpe_max": strategy.sharpe_ratio}

    factor_low = 1.0 - SENSITIVITY_RANGE
    factor_high = 1.0 + SENSITIVITY_RANGE
    sl_vars = np.linspace(strategy.stop_loss * factor_low, strategy.stop_loss * factor_high, variations)
    tp_vars = np.linspace(strategy.target_profit * factor_low, strategy.target_profit * factor_high, variations)

    sharpe_list = []
    mg = np.asarray(seg["MaxGain"].fillna(-np.inf), dtype=float)
    ml = np.asarray(seg["MaxLoss"].fillna(np.inf), dtype=float)
    fr = np.asarray(seg["IntraDayReturn"].fillna(0.0), dtype=float)

    for sl in sl_vars:
        for tp in tp_vars:
            ex = np.where(ml <= sl, sl, np.where(mg >= tp, tp, fr))
            if ex.size > 1 and np.std(ex, ddof=0) > 0:
                s = float(np.mean(ex) / np.std(ex, ddof=0) * np.sqrt(252))
                sharpe_list.append(s)
    if not sharpe_list:
        return {"is_robust": False, "sharpe_std": float("inf"), "sharpe_min": strategy.sharpe_ratio, "sharpe_max": strategy.sharpe_ratio}
    arr = np.array(sharpe_list)
    sharpe_std = float(np.std(arr))
    is_robust = (sharpe_std / abs(strategy.sharpe_ratio) < ROBUSTNESS_THRESHOLD) if strategy.sharpe_ratio != 0 else False
    return {"is_robust": is_robust, "sharpe_std": sharpe_std, "sharpe_min": float(arr.min()), "sharpe_max": float(arr.max())}

# -------------------------
# Minimum candidates analysis
# -------------------------
def analyze_minimum_candidates(df: pd.DataFrame, strategies: List[SegmentResult]) -> Dict:
    """
    Analyze the minimum number of candidates needed for statistically significant performance.
    Groups data by daily candidate count and finds the threshold where performance stabilizes.
    
    Returns dict with minimum candidate threshold and analysis details.
    """
    if not strategies or df.empty:
        return {
            "minimum_candidates": 0,
            "threshold_confidence": 0.0,
            "analysis_method": "insufficient_data",
            "candidate_segments": [],
            "recommended_minimum": 0
        }
    
    # Filter data to only include recommended strategy ranges
    masks = []
    for s in strategies:
        masks.append((df["PercentOpen"] >= s.range_min) & (df["PercentOpen"] <= s.range_max))
    combined_mask = np.logical_or.reduce(masks)
    strategy_df = df.loc[combined_mask].copy()
    
    if strategy_df.empty:
        return {
            "minimum_candidates": 0,
            "threshold_confidence": 0.0,
            "analysis_method": "no_strategy_data",
            "candidate_segments": [],
            "recommended_minimum": 0
        }
    
    # Group by date to get daily candidate counts and performance
    daily_data = []
    
    if "Date" in strategy_df.columns and strategy_df["Date"].notnull().any():
        # Use actual dates if available
        try:
            daily_groups = strategy_df.groupby(strategy_df["Date"].dt.date)
            for date, group in daily_groups:
                if len(group) > 0:
                    daily_data.append({
                        "date": date,
                        "candidate_count": len(group),
                        "win_rate": float((group["IntraDayReturn"] > 0).mean()),
                        "avg_return": float(group["IntraDayReturn"].mean()),
                        "positive_trades": int((group["IntraDayReturn"] > 0).sum()),
                        "total_trades": len(group),
                        "max_return": float(group["IntraDayReturn"].max()),
                        "min_return": float(group["IntraDayReturn"].min())
                    })
        except Exception:
            # Fallback to symbol grouping
            pass
    
    # If no date-based grouping worked, use symbol-based synthetic days
    if not daily_data:
        # Create synthetic "days" by grouping symbols into batches
        symbols = strategy_df["Symbol"].unique()
        batch_size = max(1, len(symbols) // 20)  # Aim for ~20 synthetic days
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            group = strategy_df[strategy_df["Symbol"].isin(batch_symbols)]
            if len(group) > 0:
                daily_data.append({
                    "date": f"batch_{i//batch_size}",
                    "candidate_count": len(group),
                    "win_rate": float((group["IntraDayReturn"] > 0).mean()),
                    "avg_return": float(group["IntraDayReturn"].mean()),
                    "positive_trades": int((group["IntraDayReturn"] > 0).sum()),
                    "total_trades": len(group),
                    "max_return": float(group["IntraDayReturn"].max()),
                    "min_return": float(group["IntraDayReturn"].min())
                })
    
    if len(daily_data) < 5:  # Need minimum data for analysis
        return {
            "minimum_candidates": 0,
            "threshold_confidence": 0.0,
            "analysis_method": "insufficient_daily_data",
            "candidate_segments": [],
            "recommended_minimum": 0
        }
    
    # Convert to DataFrame for easier analysis
    daily_df = pd.DataFrame(daily_data)
    
    # Create candidate count segments using quantiles
    candidate_counts = daily_df["candidate_count"].values
    num_segments = min(6, len(daily_df) // 2)  # Use fewer segments for smaller datasets
    
    if num_segments < 3:
        # Too few data points for segmentation - use percentile approach
        # For skewed data with mostly high values, use lower percentile
        min_threshold = max(1, int(np.percentile(candidate_counts, 25)))  # 25th percentile
        return {
            "minimum_candidates": min_threshold,
            "threshold_confidence": 0.5,  # Medium confidence for percentile method
            "analysis_method": "percentile_25_fallback",
            "candidate_segments": [],
            "recommended_minimum": min_threshold
        }
    
    # Create quantile-based segments
    quantiles = np.linspace(0.0, 1.0, num_segments + 1)
    bounds = np.quantile(candidate_counts, quantiles)
    
    # Analyze each segment
    segments = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if i == len(bounds) - 2:  # Last segment, include upper bound
            mask = (daily_df["candidate_count"] >= lo) & (daily_df["candidate_count"] <= hi)
        else:
            mask = (daily_df["candidate_count"] >= lo) & (daily_df["candidate_count"] < hi)
        
        segment_data = daily_df[mask]
        if len(segment_data) < 2:
            continue
            
        segment_info = {
            "range_min": float(lo),
            "range_max": float(hi),
            "days_count": len(segment_data),
            "avg_candidates": float(segment_data["candidate_count"].mean()),
            "median_candidates": float(segment_data["candidate_count"].median()),
            "avg_win_rate": float(segment_data["win_rate"].mean()),
            "avg_return": float(segment_data["avg_return"].mean()),
            "win_rate_std": float(segment_data["win_rate"].std()),
            "return_std": float(segment_data["avg_return"].std()),
            "min_candidates": int(segment_data["candidate_count"].min()),
            "max_candidates": int(segment_data["candidate_count"].max())
        }
        segments.append(segment_info)
    
    if len(segments) < 2:
        # Use percentile approach for insufficient segments
        min_threshold = max(1, int(np.percentile(candidate_counts, 25)))
        return {
            "minimum_candidates": min_threshold,
            "threshold_confidence": 0.3,  # Low confidence due to insufficient data
            "analysis_method": "insufficient_segments_percentile",
            "candidate_segments": segments,
            "recommended_minimum": min_threshold
        }
    
    # Find the minimum threshold where performance stabilizes
    # Look for the point where increasing candidates doesn't significantly improve performance
    min_threshold = 0
    threshold_confidence = 0.0
    analysis_method = "performance_stability"
    
    # Sort segments by average candidate count
    segments.sort(key=lambda x: x["avg_candidates"])
    
    # Find stabilization point using multiple criteria
    stabilization_scores = []
    
    for i in range(1, len(segments)):
        current_seg = segments[i]
        prev_seg = segments[i-1]
        
        # Calculate improvement metrics
        win_rate_improvement = current_seg["avg_win_rate"] - prev_seg["avg_win_rate"]
        return_improvement = current_seg["avg_return"] - prev_seg["avg_return"]
        
        # Calculate stability (lower std deviation is better)
        win_rate_stability = 1.0 / (1.0 + current_seg["win_rate_std"]) if current_seg["win_rate_std"] > 0 else 1.0
        return_stability = 1.0 / (1.0 + current_seg["return_std"]) if current_seg["return_std"] > 0 else 1.0
        
        # Combined score: high performance + stability, low marginal improvement
        performance_score = current_seg["avg_win_rate"] * 0.4 + (current_seg["avg_return"] / 10.0) * 0.3
        stability_score = (win_rate_stability + return_stability) * 0.3
        
        # Penalty for requiring too many candidates (diminishing returns)
        candidate_penalty = min(0.2, current_seg["avg_candidates"] / 100.0)
        
        # Bonus for being early in the sequence (prefer lower candidate counts)
        early_bonus = (len(segments) - i) / len(segments) * 0.1
        
        total_score = performance_score + stability_score - candidate_penalty + early_bonus
        
        stabilization_scores.append({
            "segment_index": i,
            "candidates": current_seg["avg_candidates"],
            "score": total_score,
            "win_rate": current_seg["avg_win_rate"],
            "avg_return": current_seg["avg_return"],
            "stability": (win_rate_stability + return_stability) / 2.0
        })
    
    if stabilization_scores:
        # Find the segment with the best score
        best_segment = max(stabilization_scores, key=lambda x: x["score"])
        min_threshold = int(best_segment["candidates"])
        threshold_confidence = best_segment["score"]
        
        # Additional validation: ensure this segment meets minimum performance criteria
        corresponding_segment = segments[best_segment["segment_index"]]
        if corresponding_segment["avg_win_rate"] < 0.3 or corresponding_segment["avg_return"] < 0:
            # Performance too low, fall back to higher candidate count
            analysis_method = "performance_fallback"
            if len(segments) > best_segment["segment_index"] + 1:
                min_threshold = int(segments[-1]["avg_candidates"])  # Use highest segment
            else:
                min_threshold = int(np.median(candidate_counts))
    else:
        # Fallback to median
        min_threshold = int(np.median(candidate_counts))
        analysis_method = "median_fallback"
    
    # Statistical significance test between low and high candidate days
    low_threshold = np.percentile(candidate_counts, 33)
    high_threshold = np.percentile(candidate_counts, 67)
    
    low_days = daily_df[daily_df["candidate_count"] <= low_threshold]
    high_days = daily_df[daily_df["candidate_count"] >= high_threshold]
    
    significance_p_value = 1.0
    if len(low_days) >= 3 and len(high_days) >= 3:
        try:
            _, significance_p_value = stats.ttest_ind(
                high_days["win_rate"], low_days["win_rate"], 
                equal_var=False, nan_policy="omit"
            )
        except Exception:
            pass
    
    # Final recommendation: ensure minimum is reasonable
    recommended_minimum = max(1, min(min_threshold, int(np.percentile(candidate_counts, 75))))
    
    return {
        "minimum_candidates": min_threshold,
        "threshold_confidence": float(threshold_confidence),
        "analysis_method": analysis_method,
        "candidate_segments": segments,
        "recommended_minimum": recommended_minimum,
        "statistical_significance": {
            "p_value": float(significance_p_value),
            "low_candidate_avg_win_rate": float(low_days["win_rate"].mean()) if len(low_days) > 0 else 0.0,
            "high_candidate_avg_win_rate": float(high_days["win_rate"].mean()) if len(high_days) > 0 else 0.0,
            "sample_size_low": len(low_days),
            "sample_size_high": len(high_days)
        },
        "summary_stats": {
            "total_days_analyzed": len(daily_df),
            "avg_daily_candidates": float(daily_df["candidate_count"].mean()),
            "median_daily_candidates": float(daily_df["candidate_count"].median()),
            "min_daily_candidates": int(daily_df["candidate_count"].min()),
            "max_daily_candidates": int(daily_df["candidate_count"].max()),
            "overall_win_rate": float(daily_df["win_rate"].mean()),
            "overall_avg_return": float(daily_df["avg_return"].mean())
        }
    }

# -------------------------
# Recommend daily buy count
# -------------------------
def recommend_daily_count(df: pd.DataFrame, strategies: List[SegmentResult], min_candidates: int = 0) -> Dict:
    """
    For all recommended bins, compute historical average candidate/day and candidate win rate,
    then recommend int(round(avg_candidates_per_day * win_rate)).
    If min_candidates > 0, only consider days with at least that many candidates.
    If Date not available, fall back to heuristics (unique symbols/20 days).
    """
    if not strategies:
        return {
            "avg_candidates_per_day": 0.0, 
            "win_rate": 0.0, 
            "recommended_count": 0,
            "total_days": 0,
            "qualifying_days": 0,
            "min_candidates_filter": min_candidates
        }

    masks = []
    for s in strategies:
        masks.append((df["PercentOpen"] >= s.range_min) & (df["PercentOpen"] <= s.range_max))
    combined_mask = np.logical_or.reduce(masks)
    cand = df.loc[combined_mask].copy()
    if cand.empty:
        return {
            "avg_candidates_per_day": 0.0, 
            "win_rate": 0.0, 
            "recommended_count": 0,
            "total_days": 0,
            "qualifying_days": 0,
            "min_candidates_filter": min_candidates
        }

    # Group by date to filter by minimum candidates if specified
    total_days = 0
    qualifying_days = 0
    filtered_candidates = cand.copy()
    
    if min_candidates > 0 and "Date" in cand.columns and cand["Date"].notnull().any():
        try:
            daily_groups = cand.groupby(cand["Date"].dt.date)
            qualifying_dates = []
            
            for date, group in daily_groups:
                total_days += 1
                if len(group) >= min_candidates:
                    qualifying_days += 1
                    qualifying_dates.append(date)
            
            # Filter to only include data from qualifying days
            if qualifying_dates:
                filtered_candidates = cand[cand["Date"].dt.date.isin(qualifying_dates)].copy()
            else:
                filtered_candidates = pd.DataFrame()  # No qualifying days
                
        except Exception:
            # If date filtering fails, use all data but note the issue
            total_days = 1
            qualifying_days = 1 if len(cand) >= min_candidates else 0

    # Calculate metrics from filtered data
    if filtered_candidates.empty:
        return {
            "avg_candidates_per_day": 0.0,
            "win_rate": 0.0, 
            "recommended_count": 0,
            "total_days": total_days,
            "qualifying_days": qualifying_days,
            "min_candidates_filter": min_candidates
        }

    if "Date" in filtered_candidates.columns and filtered_candidates["Date"].notnull().any():
        try:
            days = filtered_candidates["Date"].dt.date.nunique()
            avg_candidates_per_day = len(filtered_candidates) / days if days > 0 else float(len(filtered_candidates))
        except Exception:
            avg_candidates_per_day = float(len(filtered_candidates))
    else:
        unique_symbols = filtered_candidates["Symbol"].nunique() if "Symbol" in filtered_candidates.columns else len(filtered_candidates)
        heuristic_days = max(1, qualifying_days) if qualifying_days > 0 else 20
        avg_candidates_per_day = unique_symbols / heuristic_days

    win_rate = float((filtered_candidates["IntraDayReturn"] > 0).mean())
    recommended_count = int(round(avg_candidates_per_day * win_rate))
    if recommended_count < 0:
        recommended_count = 0
        
    return {
        "avg_candidates_per_day": float(avg_candidates_per_day), 
        "win_rate": win_rate, 
        "recommended_count": recommended_count,
        "total_days": total_days,
        "qualifying_days": qualifying_days,
        "min_candidates_filter": min_candidates
    }

# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(input_file: str = INPUT_FILE,
                 num_segments: int = DEFAULT_NUM_SEGMENTS,
                 output_json: str = OUTPUT_JSON) -> Dict:
    print("Loading data from:", input_file)
    df = parse_earningsreport_text(input_file)
    if df.empty:
        raise RuntimeError("No data parsed from input file. Check format / path.")

    # some quick sanity conversions
    df["PercentOpen"] = pd.to_numeric(df["PercentOpen"], errors="coerce")
    df["IntraDayReturn"] = pd.to_numeric(df["IntraDayReturn"], errors="coerce")
    df["MaxGain"] = pd.to_numeric(df.get("MaxGain", np.nan), errors="coerce")
    df["MaxLoss"] = pd.to_numeric(df.get("MaxLoss", np.nan), errors="coerce")
    df = df.dropna(subset=["PercentOpen", "IntraDayReturn"]).reset_index(drop=True)

    # identify segments
    segments = identify_segments(df, num_segments=num_segments)

    # analyze segments in parallel
    results: List[SegmentResult] = []
    def analyze_one(segment):
        sig = test_statistical_significance(df, segment)
        if not (sig["significant"] and sig["meets_goal"]):
            return None
        # get segment df
        lo, hi = segment
        seg_df = df[(df["PercentOpen"] >= lo) & (df["PercentOpen"] <= hi)].copy()
        if len(seg_df) < MIN_TRADES_PER_SEGMENT:
            return None
        opt = optimize_exit_strategy(seg_df)
        if not opt:
            return None
        # enforce minimum thresholds
        if opt["win_rate"] < MIN_WIN_RATE or opt["sharpe_ratio"] < MIN_SHARPE_RATIO or opt["total_trades"] < MIN_TRADES_PER_SEGMENT:
            return None
        # build SegmentResult
        sr = SegmentResult(
            range_min=float(lo),
            range_max=float(hi),
            stop_loss=float(opt["stop_loss"]),
            target_profit=float(opt["target_profit"]),
            win_rate=float(opt["win_rate"]),
            avg_return=float(opt["expected_return"]),
            trade_count=int(opt["total_trades"]),
            sharpe_ratio=float(opt["sharpe_ratio"]),
            effect_size=float(sig.get("effect_size", 0.0)),
            p_value=float(sig.get("p_value", 1.0)),
            segment_mean=float(sig.get("segment_mean", 0.0)),
            is_robust=None
        )
        # optional sensitivity check (not too expensive)
        sens = conduct_sensitivity(seg_df, sr)
        sr.is_robust = bool(sens.get("is_robust", False))
        return sr

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(segments)))) as ex:
        futures = list(ex.map(analyze_one, segments))

    for r in futures:
        if r is not None:
            results.append(r)

    # Sort by Sharpe desc
    results.sort(key=lambda x: x.sharpe_ratio if x else 0.0, reverse=True)

    # Analyze minimum candidates needed
    min_candidates_analysis = analyze_minimum_candidates(df, results)
    
    # Recommended daily count (filtered by minimum candidates)
    rec = recommend_daily_count(df, results, min_candidates_analysis.get("recommended_minimum", 0))

    # Build output JSON structure
    metadata = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "goal_return": GOAL_RETURN,
        "num_segments_tested": len(segments),
        "data_rows": int(len(df)),
        "recommended_daily_candidates_avg": rec["avg_candidates_per_day"],
        "recommended_daily_buy_count": rec["recommended_count"],
        "minimum_candidates_analysis": min_candidates_analysis,
        "daily_count_stats": {
            "total_days_analyzed": rec["total_days"],
            "qualifying_days": rec["qualifying_days"],
            "min_candidates_filter_applied": rec["min_candidates_filter"],
            "qualification_rate": rec["qualifying_days"] / rec["total_days"] if rec["total_days"] > 0 else 0.0
        }
    }

    bins = []
    for s in results:
        bins.append({
            "range_min": s.range_min,
            "range_max": s.range_max,
            "stop_loss": s.stop_loss,
            "target_profit": s.target_profit,
            "win_rate": s.win_rate,
            "avg_return": s.avg_return,
            "trade_count": s.trade_count,
            "sharpe_ratio": s.sharpe_ratio,
            "effect_size": s.effect_size,
            "p_value": s.p_value,
            "segment_mean": s.segment_mean,
            "is_robust": bool(s.is_robust)
        })

    out = {"metadata": metadata, "bins": bins}

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    
    print(f"Wrote {len(bins)} recommended bins to {output_json}")
    print(f"Minimum candidates analysis: {min_candidates_analysis['recommended_minimum']} candidates recommended")
    print(f"Analysis method: {min_candidates_analysis['analysis_method']}")
    print(f"Confidence score: {min_candidates_analysis['threshold_confidence']:.3f}")
    
    # Print daily count statistics
    total_days = rec["total_days"]
    qualifying_days = rec["qualifying_days"]
    if total_days > 0:
        qual_rate = qualifying_days / total_days * 100
        print(f"Daily count filtering: {qualifying_days}/{total_days} days qualify ({qual_rate:.1f}%)")
        print(f"Filtered recommendations: {rec['avg_candidates_per_day']:.1f} avg candidates/day, {rec['recommended_count']} daily buys")
    
    return out

# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EarningsStats - compute optimized bins for EarningsBot")
    parser.add_argument("--input", "-i", default=INPUT_FILE, help="Path to earningsreport.txt (appended reports)")
    parser.add_argument("--segments", "-s", type=int, default=DEFAULT_NUM_SEGMENTS, help="Number of quantile segments to test")
    parser.add_argument("--output", "-o", default=OUTPUT_JSON, help="Output JSON file for EarningsBot")
    args = parser.parse_args()

    try:
        result = run_pipeline(input_file=args.input, num_segments=args.segments, output_json=args.output)
    except Exception as e:
        print("ERROR:", str(e))
        raise