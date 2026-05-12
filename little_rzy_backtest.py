"""
Little RZY Strategy Backtest
=============================
Mechanical interpretation of Marci's "Little RZY" swing trading strategy.

Run modes:
    python little_rzy_backtest.py --source yahoo       # pull ES=F from yfinance (free, ~2yr)
    python little_rzy_backtest.py --source csv --file data.csv
    python little_rzy_backtest.py --source synthetic   # fake data for smoke-testing

Outputs:
    - trades.csv        (full trade log)
    - equity_curve.png  (equity + drawdown chart)
    - sample_trades.png (worst / median / best trade annotated)
    - stats printed to stdout
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# STRATEGY PARAMETERS  (tune these)
# =============================================================================
@dataclass
class Params:
    # Trend detection
    ema_period: int = 50              # price above EMA = uptrend, below = downtrend

    # Impulse detection
    impulse_atr_mult: float = 2.0     # impulse must move >= 2 ATR
    impulse_max_bars: int = 8
    atr_period: int = 14

    # Pullback detection
    pullback_min_bars: int = 3
    pullback_max_bars: int = 12
    pullback_min_pct: float = 0.30    # pullback must retrace >= 30% of impulse

    # Entry trigger: price touches trendline then closes back through it
    entry_touch_tolerance: float = 0.001  # 0.1% counts as "touch"

    # Risk management
    stop_buffer_atr: float = 0.5      # stop distance beyond trendline
    risk_per_trade: float = 0.01      # 1% account risk per trade
    target_scale: float = 0.75        # target = impulse_extreme ± measured * target_scale

    # Filters
    use_bollinger_filter: bool = True
    bb_period: int = 20
    bb_std: float = 2.0

    use_exhaustion_filter: bool = True
    max_rzy_per_trend: int = 2        # only first 2 RZYs per trend

    # Backtest engine
    starting_capital: float = 100_000
    commission_per_trade: float = 4.0  # ES futures roundtrip estimate


P = Params()


# =============================================================================
# DATA LOADING
# =============================================================================
def load_yahoo() -> pd.DataFrame:
    import yfinance as yf
    print("  Pulling ES=F from Yahoo Finance (1H, 730 days)...")
    df = yf.Ticker("ES=F").history(period="730d", interval="1h")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = [c.lower() for c in df.columns]
    df = df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return df


def load_databento(api_key: str, start: str = "2020-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    import databento as db
    client = db.Historical(key=api_key)
    print(f"  Fetching ES.c.0 ohlcv-1m from {start} to {end}...")
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["ES.c.0"],
        schema="ohlcv-1m",
        stype_in="continuous",
        start=start,
        end=end,
    )
    df = data.to_df()
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c] / 1e9
    df = df[["open", "high", "low", "close", "volume"]]
    return df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def load_csv(path: str) -> pd.DataFrame:
    """
    Auto-detects CSV format from Barchart, Investing.com, TradingView,
    or any standard OHLCV export (datetime index + open/high/low/close/volume).
    """
    raw = pd.read_csv(path, thousands=",")
    raw.columns = [c.strip().lower().replace(" ", "_").replace("%", "pct")
                   for c in raw.columns]

    dt_candidates = [c for c in raw.columns if c in
                     ("datetime", "date", "time", "timestamp", "bar_time")]
    dt_col = dt_candidates[0] if dt_candidates else raw.columns[0]

    if dt_col == "time" and pd.to_numeric(raw[dt_col], errors="coerce").notna().all():
        raw["datetime"] = pd.to_datetime(raw[dt_col], unit="s", utc=True)
    else:
        raw["datetime"] = pd.to_datetime(raw[dt_col])
    raw = raw.set_index("datetime").sort_index()

    if "close" not in raw.columns and "price" in raw.columns:
        raw = raw.rename(columns={"price": "close"})
    if "close" not in raw.columns and "last" in raw.columns:
        raw = raw.rename(columns={"last": "close"})

    if "volume" not in raw.columns:
        vol_col = next((c for c in raw.columns if "vol" in c), None)
        if vol_col:
            raw = raw.rename(columns={vol_col: "volume"})
    if "volume" in raw.columns:
        def _parse_vol(v):
            if isinstance(v, str):
                v = v.replace(",", "")
                if v.endswith("K"):
                    return float(v[:-1]) * 1_000
                if v.endswith("M"):
                    return float(v[:-1]) * 1_000_000
                if v in ("-", ""):
                    return 0.0
            try:
                return float(v)
            except Exception:
                return 0.0
        raw["volume"] = raw["volume"].apply(_parse_vol)
    else:
        raw["volume"] = 0.0

    for c in ["open", "high", "low", "close"]:
        raw[c] = pd.to_numeric(raw[c].astype(str).str.replace(",", ""), errors="coerce")

    df = raw[["open", "high", "low", "close", "volume"]].dropna(
        subset=["open", "high", "low", "close"]
    )

    if len(df) > 3000:
        df = df.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        print(f"  Resampled to 4H -> {len(df)} bars")

    return df


def generate_synthetic(n_bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    """ES-like 4H data with trends, pullbacks, noise. NOT real data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="4h")

    price = 4500.0
    prices = []
    regime_len = 80
    for i in range(n_bars):
        if i % regime_len == 0:
            regime = rng.choice([-1, 0, 1], p=[0.4, 0.2, 0.4])
            drift = regime * 0.15
            vol = 8.0
        pullback_factor = -0.3 * drift if (i % 15 < 4) else 1.0
        ret = drift * pullback_factor + rng.normal(0, vol)
        price += ret
        prices.append(price)

    closes = np.array(prices)
    highs = closes + np.abs(rng.normal(0, 4, n_bars))
    lows = closes - np.abs(rng.normal(0, 4, n_bars))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": rng.integers(1000, 10000, n_bars),
    }, index=dates)


# =============================================================================
# INDICATORS
# =============================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema"] = df["close"].ewm(span=P.ema_period, adjust=False).mean()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(P.atr_period).mean()

    df["bb_mid"] = df["close"].rolling(P.bb_period).mean()
    bb_std = df["close"].rolling(P.bb_period).std()
    df["bb_upper"] = df["bb_mid"] + P.bb_std * bb_std
    df["bb_lower"] = df["bb_mid"] - P.bb_std * bb_std

    return df.dropna()


# =============================================================================
# STRUCTURE DETECTION
# =============================================================================
@dataclass
class RZYStructure:
    direction: str               # "short" or "long"
    impulse_start_idx: int
    impulse_end_idx: int         # bar of the impulse extreme (low for short, high for long)
    pullback_end_idx: int
    impulse_extreme: float       # lowest low (short) or highest high (long)
    trendline_slope: float       # fit across pullback highs (short) or lows (long)
    trendline_intercept: float
    measured_distance: float     # vertical distance from extreme to trendline
    target_price: float
    rzy_number: int = 1


def trendline_at(struct: RZYStructure, idx: int) -> float:
    return struct.trendline_slope * idx + struct.trendline_intercept


def detect_short_structures(df: pd.DataFrame) -> list[RZYStructure]:
    """Downtrend: impulse down, pullback up, trendline across pullback highs."""
    structures = []
    i = P.impulse_max_bars + P.pullback_max_bars
    current_trend_id = None
    rzy_count_in_trend = 0

    while i < len(df) - 1:
        bar = df.iloc[i]

        if bar["close"] >= bar["ema"]:
            current_trend_id = None
            rzy_count_in_trend = 0
            i += 1
            continue

        atr = bar["atr"]
        impulse_threshold = P.impulse_atr_mult * atr

        window = df.iloc[max(0, i - P.impulse_max_bars - P.pullback_max_bars):i + 1]
        if len(window) < 5:
            i += 1
            continue

        low_idx_rel = window["low"].values.argmin()
        low_idx = window.index[low_idx_rel]
        low_iloc = df.index.get_loc(low_idx)
        lowest_low = window["low"].iloc[low_idx_rel]

        pre_window = df.iloc[max(0, low_iloc - P.impulse_max_bars):low_iloc + 1]
        if len(pre_window) < 3:
            i += 1
            continue
        impulse_high = pre_window["high"].max()
        impulse_size = impulse_high - lowest_low

        if impulse_size < impulse_threshold:
            i += 1
            continue

        pullback = df.iloc[low_iloc + 1:i + 1]
        if len(pullback) < P.pullback_min_bars or len(pullback) > P.pullback_max_bars:
            i += 1
            continue

        bounce = pullback["high"].max() - lowest_low
        if bounce < P.pullback_min_pct * impulse_size:
            i += 1
            continue

        if pullback["low"].min() < lowest_low:
            i += 1
            continue

        x = np.arange(low_iloc + 1, i + 1)
        y = pullback["high"].values
        if len(x) < 2:
            i += 1
            continue
        slope, intercept = np.polyfit(x, y, 1)

        measured = (slope * low_iloc + intercept) - lowest_low
        if measured <= 0:
            i += 1
            continue

        if P.use_bollinger_filter and bar["close"] < bar["bb_lower"]:
            i += 1
            continue

        trend_id_now = low_iloc // 50
        if current_trend_id != trend_id_now:
            current_trend_id = trend_id_now
            rzy_count_in_trend = 1
        else:
            rzy_count_in_trend += 1

        if P.use_exhaustion_filter and rzy_count_in_trend > P.max_rzy_per_trend:
            i += 1
            continue

        structures.append(RZYStructure(
            direction="short",
            impulse_start_idx=low_iloc - len(pre_window) + 1,
            impulse_end_idx=low_iloc,
            pullback_end_idx=i,
            impulse_extreme=lowest_low,
            trendline_slope=slope,
            trendline_intercept=intercept,
            measured_distance=measured,
            target_price=lowest_low - measured * P.target_scale,
            rzy_number=rzy_count_in_trend,
        ))
        i += P.pullback_min_bars

    return structures


def detect_long_structures(df: pd.DataFrame) -> list[RZYStructure]:
    """Uptrend: impulse up, pullback down, trendline across pullback lows."""
    structures = []
    i = P.impulse_max_bars + P.pullback_max_bars
    current_trend_id = None
    rzy_count_in_trend = 0

    while i < len(df) - 1:
        bar = df.iloc[i]

        if bar["close"] <= bar["ema"]:
            current_trend_id = None
            rzy_count_in_trend = 0
            i += 1
            continue

        atr = bar["atr"]
        impulse_threshold = P.impulse_atr_mult * atr

        window = df.iloc[max(0, i - P.impulse_max_bars - P.pullback_max_bars):i + 1]
        if len(window) < 5:
            i += 1
            continue

        # Impulse extreme is the highest high
        high_idx_rel = window["high"].values.argmax()
        high_idx = window.index[high_idx_rel]
        high_iloc = df.index.get_loc(high_idx)
        highest_high = window["high"].iloc[high_idx_rel]

        pre_window = df.iloc[max(0, high_iloc - P.impulse_max_bars):high_iloc + 1]
        if len(pre_window) < 3:
            i += 1
            continue
        impulse_low = pre_window["low"].min()
        impulse_size = highest_high - impulse_low

        if impulse_size < impulse_threshold:
            i += 1
            continue

        pullback = df.iloc[high_iloc + 1:i + 1]
        if len(pullback) < P.pullback_min_bars or len(pullback) > P.pullback_max_bars:
            i += 1
            continue

        # Pullback down must retrace >= 30% of impulse
        bounce = highest_high - pullback["low"].min()
        if bounce < P.pullback_min_pct * impulse_size:
            i += 1
            continue

        # Pullback must not break above the impulse high
        if pullback["high"].max() > highest_high:
            i += 1
            continue

        # Trendline across pullback lows
        x = np.arange(high_iloc + 1, i + 1)
        y = pullback["low"].values
        if len(x) < 2:
            i += 1
            continue
        slope, intercept = np.polyfit(x, y, 1)

        measured = highest_high - (slope * high_iloc + intercept)
        if measured <= 0:
            i += 1
            continue

        # Bollinger filter: don't buy when already stretched to upper band
        if P.use_bollinger_filter and bar["close"] > bar["bb_upper"]:
            i += 1
            continue

        trend_id_now = high_iloc // 50
        if current_trend_id != trend_id_now:
            current_trend_id = trend_id_now
            rzy_count_in_trend = 1
        else:
            rzy_count_in_trend += 1

        if P.use_exhaustion_filter and rzy_count_in_trend > P.max_rzy_per_trend:
            i += 1
            continue

        structures.append(RZYStructure(
            direction="long",
            impulse_start_idx=high_iloc - len(pre_window) + 1,
            impulse_end_idx=high_iloc,
            pullback_end_idx=i,
            impulse_extreme=highest_high,
            trendline_slope=slope,
            trendline_intercept=intercept,
            measured_distance=measured,
            target_price=highest_high + measured * P.target_scale,
            rzy_number=rzy_count_in_trend,
        ))
        i += P.pullback_min_bars

    return structures


# =============================================================================
# TRADE SIMULATION
# =============================================================================
@dataclass
class Trade:
    structure: RZYStructure
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    r_multiple: float = 0.0
    pnl: float = 0.0


def simulate(df: pd.DataFrame, structures: list[RZYStructure]) -> list[Trade]:
    trades = []

    for s in structures:
        is_short = s.direction == "short"
        entry_idx = None
        entry_price = None

        search_end = min(s.pullback_end_idx + 15, len(df) - 1)
        for j in range(s.pullback_end_idx + 1, search_end):
            tl_val = trendline_at(s, j)
            bar = df.iloc[j]

            if is_short:
                # High touches trendline, then closes back below it
                touched = bar["high"] >= tl_val * (1 - P.entry_touch_tolerance)
                confirmed = bar["close"] < tl_val
                invalidated = bar["close"] > tl_val * 1.002
            else:
                # Low touches trendline, then closes back above it
                touched = bar["low"] <= tl_val * (1 + P.entry_touch_tolerance)
                confirmed = bar["close"] > tl_val
                invalidated = bar["close"] < tl_val * 0.998

            if touched and confirmed:
                entry_idx = j
                entry_price = bar["close"]
                break
            if invalidated:
                break

        if entry_idx is None:
            continue

        atr_at_entry = df.iloc[entry_idx]["atr"]
        tl_at_entry = trendline_at(s, entry_idx)

        if is_short:
            stop = tl_at_entry + P.stop_buffer_atr * atr_at_entry
            risk_per_unit = stop - entry_price
        else:
            stop = tl_at_entry - P.stop_buffer_atr * atr_at_entry
            risk_per_unit = entry_price - stop

        if risk_per_unit <= 0:
            continue

        trade = Trade(
            structure=s,
            entry_idx=entry_idx,
            entry_price=entry_price,
            stop_price=stop,
            target_price=s.target_price,
        )

        for k in range(entry_idx + 1, len(df)):
            bar = df.iloc[k]

            if is_short:
                stop_hit = bar["high"] >= stop
                target_hit = bar["low"] <= s.target_price
            else:
                stop_hit = bar["low"] <= stop
                target_hit = bar["high"] >= s.target_price

            if stop_hit:
                trade.exit_idx = k
                trade.exit_price = stop
                trade.exit_reason = "stop"
                trade.r_multiple = -1.0
                break
            if target_hit:
                trade.exit_idx = k
                trade.exit_price = s.target_price
                trade.exit_reason = "target"
                if is_short:
                    trade.r_multiple = (entry_price - s.target_price) / risk_per_unit
                else:
                    trade.r_multiple = (s.target_price - entry_price) / risk_per_unit
                break
            if k - entry_idx > 50:
                trade.exit_idx = k
                trade.exit_price = bar["close"]
                trade.exit_reason = "timeout"
                if is_short:
                    trade.r_multiple = (entry_price - bar["close"]) / risk_per_unit
                else:
                    trade.r_multiple = (bar["close"] - entry_price) / risk_per_unit
                break

        if trade.exit_idx is None:
            trade.exit_idx = len(df) - 1
            trade.exit_price = df.iloc[-1]["close"]
            trade.exit_reason = "end_of_data"
            if is_short:
                trade.r_multiple = (entry_price - trade.exit_price) / risk_per_unit
            else:
                trade.r_multiple = (trade.exit_price - entry_price) / risk_per_unit

        ES_MULTIPLIER = 50
        risk_dollars = P.starting_capital * P.risk_per_trade
        contracts = max(1, int(risk_dollars / (risk_per_unit * ES_MULTIPLIER)))
        if is_short:
            trade.pnl = (entry_price - trade.exit_price) * ES_MULTIPLIER * contracts - P.commission_per_trade
        else:
            trade.pnl = (trade.exit_price - entry_price) * ES_MULTIPLIER * contracts - P.commission_per_trade

        trades.append(trade)

    # Sort by entry time so equity curve is chronological
    trades.sort(key=lambda t: t.entry_idx)
    return trades


# =============================================================================
# STATS & REPORTING
# =============================================================================
def report(trades: list[Trade], df: pd.DataFrame) -> dict:
    if not trades:
        print("No trades generated.")
        return {}

    n = len(trades)
    wins = [t for t in trades if t.r_multiple > 0]
    losses = [t for t in trades if t.r_multiple <= 0]
    win_rate = len(wins) / n
    avg_win_r = np.mean([t.r_multiple for t in wins]) if wins else 0
    avg_loss_r = np.mean([t.r_multiple for t in losses]) if losses else 0
    expectancy_r = win_rate * avg_win_r + (1 - win_rate) * avg_loss_r
    total_pnl = sum(t.pnl for t in trades)

    pnls = np.array([t.pnl for t in trades])
    equity = P.starting_capital + np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd = dd.min()

    longs  = [t for t in trades if t.structure.direction == "long"]
    shorts = [t for t in trades if t.structure.direction == "short"]

    stats = {
        "trades": n,
        "  longs": len(longs),
        "  shorts": len(shorts),
        "win_rate": win_rate,
        "avg_win_R": avg_win_r,
        "avg_loss_R": avg_loss_r,
        "expectancy_R": expectancy_r,
        "total_pnl": total_pnl,
        "final_equity": equity[-1],
        "max_drawdown_pct": max_dd * 100,
        "return_pct": (equity[-1] / P.starting_capital - 1) * 100,
    }

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:25s} {v:>12.2f}")
        else:
            print(f"  {k:25s} {v:>12}")
    print("=" * 50)

    from collections import Counter
    reasons = Counter(t.exit_reason for t in trades)
    print("\nExit reasons:")
    for r, c in reasons.most_common():
        print(f"  {r:15s} {c:>4}  ({c/n*100:.1f}%)")

    return stats


def plot_equity(trades: list[Trade], path: str):
    pnls = np.array([t.pnl for t in trades])
    equity = P.starting_capital + np.cumsum(pnls)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity, linewidth=1.5, color="#1f77b4")
    ax1.axhline(P.starting_capital, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Equity Curve — Little RZY Strategy (ES 4H, Long + Short)", fontsize=13)
    ax1.set_ylabel("Account Equity ($)")
    ax1.grid(alpha=0.3)

    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max * 100
    ax2.fill_between(range(len(dd)), dd, 0, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trade #")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()


def plot_sample_trades(df: pd.DataFrame, trades: list[Trade], path: str, n_samples: int = 3):
    if len(trades) < n_samples:
        n_samples = len(trades)
    if n_samples == 0:
        return

    sorted_trades = sorted(trades, key=lambda t: t.r_multiple)
    if len(trades) >= 3:
        samples = [sorted_trades[0], sorted_trades[len(trades) // 2], sorted_trades[-1]]
    else:
        samples = sorted_trades

    fig, axes = plt.subplots(len(samples), 1, figsize=(13, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for ax, t in zip(axes, samples):
        s = t.structure
        is_short = s.direction == "short"
        start = max(0, s.impulse_start_idx - 10)
        end = min(len(df) - 1, t.exit_idx + 5)
        window = df.iloc[start:end + 1]

        for idx in range(len(window)):
            row = window.iloc[idx]
            x = start + idx
            color = "green" if row["close"] >= row["open"] else "red"
            ax.plot([x, x], [row["low"], row["high"]], color=color, linewidth=0.7, alpha=0.6)
            ax.add_patch(Rectangle((x - 0.3, min(row["open"], row["close"])),
                                    0.6, abs(row["close"] - row["open"]),
                                    facecolor=color, alpha=0.6, edgecolor=color))

        x_tl = np.array([s.impulse_end_idx, t.exit_idx])
        y_tl = s.trendline_slope * x_tl + s.trendline_intercept
        ax.plot(x_tl, y_tl, "b--", linewidth=1.5, label="Pullback trendline")

        extreme_label = f"{'Low' if is_short else 'High'} {s.impulse_extreme:.1f}"
        ax.axhline(s.impulse_extreme, color="purple", linestyle=":", alpha=0.6, label=extreme_label)
        ax.axhline(s.target_price, color="green", linestyle=":", alpha=0.6, label=f"Target {s.target_price:.1f}")
        ax.axhline(t.stop_price, color="red", linestyle=":", alpha=0.6, label=f"Stop {t.stop_price:.1f}")

        entry_marker = "v" if is_short else "^"
        ax.scatter([t.entry_idx], [t.entry_price], color="black", marker=entry_marker,
                   s=80, zorder=5, label="Entry")
        ax.scatter([t.exit_idx], [t.exit_price], color="orange", marker="x",
                   s=100, zorder=5, label=f"Exit ({t.exit_reason})")

        direction_label = "SHORT" if is_short else "LONG"
        ax.set_title(f"[{direction_label}] R={t.r_multiple:+.2f}  PnL=${t.pnl:+.0f}  ({t.exit_reason})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()


def trades_to_df(trades: list[Trade], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            "entry_time":  df.index[t.entry_idx],
            "exit_time":   df.index[t.exit_idx] if t.exit_idx else None,
            "direction":   t.structure.direction,
            "rzy_number":  t.structure.rzy_number,
            "entry":       round(t.entry_price, 2),
            "stop":        round(t.stop_price, 2),
            "target":      round(t.target_price, 2),
            "exit":        round(t.exit_price, 2) if t.exit_price else None,
            "exit_reason": t.exit_reason,
            "r_multiple":  round(t.r_multiple, 2),
            "pnl_usd":     round(t.pnl, 2),
        })
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["yahoo", "csv", "databento", "synthetic"],
                        default="synthetic")
    parser.add_argument("--file",  default=None,         help="CSV path (for --source csv)")
    parser.add_argument("--key",   default=None,         help="Databento API key")
    parser.add_argument("--start", default="2020-01-01", help="Start date (databento)")
    parser.add_argument("--end",   default="2025-01-01", help="End date (databento)")
    parser.add_argument("--out",   default=".",          help="Output directory")
    args = parser.parse_args()

    print(f"Loading data ({args.source})...")
    if args.source == "yahoo":
        df = load_yahoo()
    elif args.source == "csv":
        if not args.file:
            raise SystemExit("--file is required for --source csv")
        df = load_csv(args.file)
    elif args.source == "databento":
        if not args.key:
            raise SystemExit("--key is required for --source databento")
        df = load_databento(args.key, args.start, args.end)
    else:
        df = generate_synthetic()
        print("  *** SYNTHETIC DATA — for code validation only, NOT a real backtest ***")

    print(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    df = add_indicators(df)
    print(f"  {len(df)} bars after indicators")

    print("Detecting Little RZY structures...")
    short_structs = detect_short_structures(df)
    long_structs  = detect_long_structures(df)
    structures = short_structs + long_structs
    print(f"  Found {len(short_structs)} short + {len(long_structs)} long = {len(structures)} total")

    print("Simulating trades...")
    trades = simulate(df, structures)
    n_long  = sum(1 for t in trades if t.structure.direction == "long")
    n_short = sum(1 for t in trades if t.structure.direction == "short")
    print(f"  {len(trades)} trades executed ({n_long} long, {n_short} short)")

    stats = report(trades, df)

    if trades:
        td = trades_to_df(trades, df)
        td.to_csv(f"{args.out}/trades.csv", index=False)
        print(f"\nTrade log     -> {args.out}/trades.csv")

        plot_equity(trades, f"{args.out}/equity_curve.png")
        print(f"Equity curve  -> {args.out}/equity_curve.png")

        plot_sample_trades(df, trades, f"{args.out}/sample_trades.png")
        print(f"Sample trades -> {args.out}/sample_trades.png")


if __name__ == "__main__":
    main()
