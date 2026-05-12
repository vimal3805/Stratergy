"""
Little RZY Strategy Backtest
=============================
Mechanical interpretation of Marci's "Little RZY" swing trading strategy.

Run modes:
    python little_rzy_backtest.py --source yahoo    # pull ES=F from yfinance
    python little_rzy_backtest.py --source csv --file data.csv
    python little_rzy_backtest.py --source synthetic # generate fake data for testing

Outputs:
    - trades.csv (trade log)
    - equity_curve.png
    - sample_trades.png (3 annotated examples)
    - stats printed to stdout
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# STRATEGY PARAMETERS  (tune these)
# =============================================================================
@dataclass
class Params:
    # Trend detection
    ema_period: int = 50              # price below EMA = downtrend candidate

    # Impulse detection
    impulse_atr_mult: float = 2.0     # impulse must drop >= 2 ATR
    impulse_max_bars: int = 8         # within this many bars
    atr_period: int = 14

    # Pullback detection
    pullback_min_bars: int = 3        # at least N bars of bounce
    pullback_max_bars: int = 12       # but not too long
    pullback_min_pct: float = 0.30    # bounce >= 30% of impulse

    # Trendline fit
    # We use linear regression across the pullback highs (downtrend case)

    # Entry trigger: STRICT mode = price touches trendline, then closes back below it
    entry_touch_tolerance: float = 0.001  # 0.1% of price counts as "touch"

    # Risk management
    stop_buffer_atr: float = 0.5      # stop = trendline high + 0.5 ATR
    risk_per_trade: float = 0.01      # 1% account risk

    # Filters
    use_bollinger_filter: bool = True
    bb_period: int = 20
    bb_std: float = 2.0
    # Bollinger filter: for shorts, prefer entries when price is near/above middle band
    # (not already crushed at lower band)

    use_exhaustion_filter: bool = True
    max_rzy_per_trend: int = 2        # only trade first 2 RZYs in a trend

    # Backtest engine
    starting_capital: float = 100_000
    commission_per_trade: float = 4.0  # ES futures roundtrip estimate


P = Params()


# =============================================================================
# DATA LOADING
# =============================================================================
def load_yahoo() -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker("ES=F").history(period="730d", interval="1h")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = [c.lower() for c in df.columns]
    # Resample 1H to 4H
    df = df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
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
        df[c] = df[c] / 1e9          # Databento fixed-point prices
    df = df[["open", "high", "low", "close", "volume"]]
    df4h = df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return df4h


def load_csv(path: str) -> pd.DataFrame:
    """
    Auto-detects CSV format from Barchart, Investing.com, TradingView,
    or any file written by fetch_data.py (standard datetime index + OHLCV).
    """
    raw = pd.read_csv(path, thousands=",")
    raw.columns = [c.strip().lower().replace(" ", "_").replace("%", "pct")
                   for c in raw.columns]

    # --- find the datetime column ---
    dt_candidates = [c for c in raw.columns if c in
                     ("datetime", "date", "time", "timestamp", "bar_time")]
    if not dt_candidates:
        # Fall back: use first column
        dt_candidates = [raw.columns[0]]
    dt_col = dt_candidates[0]

    # TradingView uses Unix seconds for "time"
    if dt_col == "time" and pd.to_numeric(raw[dt_col], errors="coerce").notna().all():
        raw["datetime"] = pd.to_datetime(raw[dt_col], unit="s", utc=True)
    else:
        raw["datetime"] = pd.to_datetime(raw[dt_col])
    raw = raw.set_index("datetime").sort_index()

    # --- normalize close column (Investing.com calls it "price") ---
    if "close" not in raw.columns and "price" in raw.columns:
        raw = raw.rename(columns={"price": "close"})
    if "close" not in raw.columns and "last" in raw.columns:
        raw = raw.rename(columns={"last": "close"})

    # --- volume: handle K/M suffixes (Investing.com) ---
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
        raw[c] = pd.to_numeric(
            raw[c].astype(str).str.replace(",", ""), errors="coerce"
        )

    df = raw[["open", "high", "low", "close", "volume"]].dropna(
        subset=["open", "high", "low", "close"]
    )

    # If data looks like it's sub-4H (hourly or finer), resample up
    if len(df) > 3000:
        df = df.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        print(f"  Resampled to 4H → {len(df)} bars")

    return df


def generate_synthetic(n_bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generates ES-like 4H price data with realistic trends, pullbacks, and noise.
    NOT REAL DATA. For code validation only.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="4h")

    # Build a price series with regime changes (trend up, trend down, sideways)
    price = 4500.0
    prices = []
    regime_len = 80
    for i in range(n_bars):
        if i % regime_len == 0:
            # New regime: -1=down, 0=sideways, 1=up
            regime = rng.choice([-1, 0, 1], p=[0.4, 0.2, 0.4])
            drift = regime * 0.15
            vol = 8.0
        # Add mean-reverting pullbacks every ~15 bars within trends
        pullback_factor = -0.3 * drift if (i % 15 < 4) else 1.0
        ret = drift * pullback_factor + rng.normal(0, vol)
        price += ret
        prices.append(price)

    closes = np.array(prices)
    # Build OHLC around closes
    highs = closes + np.abs(rng.normal(0, 4, n_bars))
    lows = closes - np.abs(rng.normal(0, 4, n_bars))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": rng.integers(1000, 10000, n_bars),
    }, index=dates)
    return df


# =============================================================================
# INDICATORS
# =============================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema"] = df["close"].ewm(span=P.ema_period, adjust=False).mean()

    # ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(P.atr_period).mean()

    # Bollinger
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
    """A detected Little RZY pattern (downtrend short setup)."""
    impulse_start_idx: int
    impulse_end_idx: int        # bar of lowest low
    pullback_end_idx: int       # bar where pullback peaked
    lowest_low: float
    trendline_slope: float      # of pullback highs
    trendline_intercept: float
    measured_distance: float    # vertical low->trendline
    target_price: float
    rzy_number: int = 1         # which RZY in this trend sequence


def trendline_at(struct: RZYStructure, idx: int) -> float:
    """Value of pullback-high trendline at a given bar index."""
    return struct.trendline_slope * idx + struct.trendline_intercept


def detect_short_structures(df: pd.DataFrame) -> list[RZYStructure]:
    """
    Walk forward bar by bar, finding short setups.
    A setup = downtrend impulse + pullback that allows us to draw a trendline
    across pullback highs.
    """
    structures = []
    i = P.impulse_max_bars + P.pullback_max_bars

    # Track which RZY number we're on within an ongoing downtrend
    current_trend_id = None
    rzy_count_in_trend = 0

    while i < len(df) - 1:
        bar = df.iloc[i]

        # Trend filter: must be below EMA
        if bar["close"] >= bar["ema"]:
            current_trend_id = None
            rzy_count_in_trend = 0
            i += 1
            continue

        # Look back for impulse: drop of >= impulse_atr_mult ATR within impulse_max_bars
        atr = bar["atr"]
        impulse_threshold = P.impulse_atr_mult * atr

        # Find the impulse: scan window
        window = df.iloc[max(0, i - P.impulse_max_bars - P.pullback_max_bars):i + 1]
        if len(window) < 5:
            i += 1
            continue

        # Lowest low in recent window
        low_idx_rel = window["low"].values.argmin()
        low_idx = window.index[low_idx_rel]
        low_iloc = df.index.get_loc(low_idx)
        lowest_low = window["low"].iloc[low_idx_rel]

        # Pre-impulse high: highest high in 8 bars before the low
        pre_window = df.iloc[max(0, low_iloc - P.impulse_max_bars):low_iloc + 1]
        if len(pre_window) < 3:
            i += 1
            continue
        impulse_high = pre_window["high"].max()
        impulse_size = impulse_high - lowest_low

        if impulse_size < impulse_threshold:
            i += 1
            continue

        # Pullback: bars AFTER low_iloc up to current bar i
        pullback = df.iloc[low_iloc + 1:i + 1]
        if len(pullback) < P.pullback_min_bars or len(pullback) > P.pullback_max_bars:
            i += 1
            continue

        # Bounce size check
        pullback_high = pullback["high"].max()
        bounce = pullback_high - lowest_low
        if bounce < P.pullback_min_pct * impulse_size:
            i += 1
            continue

        # Pullback must not have broken below the impulse low
        if pullback["low"].min() < lowest_low:
            i += 1
            continue

        # Fit trendline across pullback highs (linear regression)
        x = np.arange(low_iloc + 1, i + 1)
        y = pullback["high"].values
        if len(x) < 2:
            i += 1
            continue
        slope, intercept = np.polyfit(x, y, 1)

        # Distance from low to trendline AT THE LOW BAR
        trendline_at_low = slope * low_iloc + intercept
        measured = trendline_at_low - lowest_low
        if measured <= 0:
            i += 1
            continue

        target = lowest_low - measured

        # Bollinger filter: for shorts, we want price not already crushed at lower band
        if P.use_bollinger_filter:
            if bar["close"] < bar["bb_lower"]:
                i += 1
                continue  # too extended already

        # Exhaustion filter: count RZYs in this trend
        trend_id_now = low_iloc // 50  # crude: group nearby setups into same trend
        if current_trend_id != trend_id_now:
            current_trend_id = trend_id_now
            rzy_count_in_trend = 1
        else:
            rzy_count_in_trend += 1

        if P.use_exhaustion_filter and rzy_count_in_trend > P.max_rzy_per_trend:
            i += 1
            continue

        structures.append(RZYStructure(
            impulse_start_idx=low_iloc - len(pre_window) + 1,
            impulse_end_idx=low_iloc,
            pullback_end_idx=i,
            lowest_low=lowest_low,
            trendline_slope=slope,
            trendline_intercept=intercept,
            measured_distance=measured,
            target_price=target,
            rzy_number=rzy_count_in_trend,
        ))
        # Skip ahead so we don't redetect the same structure
        i += P.pullback_min_bars
        continue

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
        # Entry trigger: STRICT — after pullback_end, look for price to touch
        # the trendline (extended forward) then close back below it
        entry_idx = None
        entry_price = None

        search_end = min(s.pullback_end_idx + 15, len(df) - 1)
        for j in range(s.pullback_end_idx + 1, search_end):
            tl_val = trendline_at(s, j)
            bar = df.iloc[j]
            # Touch = high reaches within tolerance of trendline
            touched = bar["high"] >= tl_val * (1 - P.entry_touch_tolerance)
            closed_below = bar["close"] < tl_val
            if touched and closed_below:
                entry_idx = j
                entry_price = bar["close"]
                break
            # Invalidation: close above trendline by clear margin
            if bar["close"] > tl_val * 1.002:
                break

        if entry_idx is None:
            continue

        # Stop = trendline at entry bar + ATR buffer
        atr_at_entry = df.iloc[entry_idx]["atr"]
        tl_at_entry = trendline_at(s, entry_idx)
        stop = tl_at_entry + P.stop_buffer_atr * atr_at_entry

        if stop <= entry_price:
            continue  # bad geometry

        trade = Trade(
            structure=s,
            entry_idx=entry_idx,
            entry_price=entry_price,
            stop_price=stop,
            target_price=s.target_price,
        )

        # Walk forward bar by bar to find exit
        risk_per_unit = stop - entry_price  # positive number
        for k in range(entry_idx + 1, len(df)):
            bar = df.iloc[k]
            # Stop hit?
            if bar["high"] >= stop:
                trade.exit_idx = k
                trade.exit_price = stop
                trade.exit_reason = "stop"
                trade.r_multiple = (entry_price - stop) / risk_per_unit  # = -1
                break
            # Target hit?
            if bar["low"] <= s.target_price:
                trade.exit_idx = k
                trade.exit_price = s.target_price
                trade.exit_reason = "target"
                trade.r_multiple = (entry_price - s.target_price) / risk_per_unit
                break
            # Time stop: 50 bars
            if k - entry_idx > 50:
                trade.exit_idx = k
                trade.exit_price = bar["close"]
                trade.exit_reason = "timeout"
                trade.r_multiple = (entry_price - bar["close"]) / risk_per_unit
                break

        if trade.exit_idx is None:
            # still open at end of data — close at last bar
            trade.exit_idx = len(df) - 1
            trade.exit_price = df.iloc[-1]["close"]
            trade.exit_reason = "end_of_data"
            trade.r_multiple = (entry_price - trade.exit_price) / risk_per_unit

        # PnL in $ (ES = $50/point)
        ES_MULTIPLIER = 50
        # Position size from fixed-fractional risk
        risk_dollars = P.starting_capital * P.risk_per_trade
        contracts = max(1, int(risk_dollars / (risk_per_unit * ES_MULTIPLIER)))
        trade.pnl = (entry_price - trade.exit_price) * ES_MULTIPLIER * contracts - P.commission_per_trade

        trades.append(trade)

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

    # Equity curve & drawdown
    pnls = np.array([t.pnl for t in trades])
    equity = P.starting_capital + np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd = dd.min()

    stats = {
        "trades": n,
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

    # Reasons breakdown
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
    ax1.set_title("Equity Curve — Little RZY Strategy (ES 4H)", fontsize=13)
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

    # Pick: best winner, worst loser, and a middle one
    sorted_trades = sorted(trades, key=lambda t: t.r_multiple)
    samples = []
    if len(trades) >= 3:
        samples = [sorted_trades[0], sorted_trades[len(trades) // 2], sorted_trades[-1]]
    else:
        samples = sorted_trades

    fig, axes = plt.subplots(len(samples), 1, figsize=(13, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for ax, t in zip(axes, samples):
        s = t.structure
        start = max(0, s.impulse_start_idx - 10)
        end = min(len(df) - 1, t.exit_idx + 5)
        window = df.iloc[start:end + 1]

        # Candles (simplified bars)
        for idx in range(len(window)):
            row = window.iloc[idx]
            x = start + idx
            color = "green" if row["close"] >= row["open"] else "red"
            ax.plot([x, x], [row["low"], row["high"]], color=color, linewidth=0.7, alpha=0.6)
            ax.add_patch(Rectangle((x - 0.3, min(row["open"], row["close"])),
                                    0.6, abs(row["close"] - row["open"]),
                                    facecolor=color, alpha=0.6, edgecolor=color))

        # Trendline
        x_tl = np.array([s.impulse_end_idx, t.exit_idx])
        y_tl = s.trendline_slope * x_tl + s.trendline_intercept
        ax.plot(x_tl, y_tl, "b--", linewidth=1.5, label="Pullback trendline")

        # Mark low and target
        ax.axhline(s.lowest_low, color="purple", linestyle=":", alpha=0.6, label=f"Low {s.lowest_low:.1f}")
        ax.axhline(s.target_price, color="green", linestyle=":", alpha=0.6, label=f"Target {s.target_price:.1f}")
        ax.axhline(t.stop_price, color="red", linestyle=":", alpha=0.6, label=f"Stop {t.stop_price:.1f}")

        # Entry/exit markers
        ax.scatter([t.entry_idx], [t.entry_price], color="black", marker="v", s=80, zorder=5, label="Entry")
        ax.scatter([t.exit_idx], [t.exit_price], color="orange", marker="x", s=100, zorder=5, label=f"Exit ({t.exit_reason})")

        ax.set_title(f"Trade R={t.r_multiple:+.2f}  PnL=${t.pnl:+.0f}  ({t.exit_reason})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()


def trades_to_df(trades: list[Trade], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            "entry_time": df.index[t.entry_idx],
            "exit_time": df.index[t.exit_idx] if t.exit_idx else None,
            "rzy_number": t.structure.rzy_number,
            "entry": round(t.entry_price, 2),
            "stop": round(t.stop_price, 2),
            "target": round(t.target_price, 2),
            "exit": round(t.exit_price, 2) if t.exit_price else None,
            "exit_reason": t.exit_reason,
            "r_multiple": round(t.r_multiple, 2),
            "pnl_usd": round(t.pnl, 2),
        })
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["yahoo", "csv", "databento", "synthetic"], default="synthetic")
    parser.add_argument("--file", default=None)
    parser.add_argument("--key", default=None, help="Databento API key")
    parser.add_argument("--start", default="2020-01-01", help="Start date (databento)")
    parser.add_argument("--end", default="2025-01-01", help="End date (databento)")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    print(f"Loading data ({args.source})...")
    if args.source == "yahoo":
        df = load_yahoo()
    elif args.source == "csv":
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
    structures = detect_short_structures(df)
    print(f"  Found {len(structures)} structures")

    print("Simulating trades...")
    trades = simulate(df, structures)
    print(f"  {len(trades)} trades executed (entry trigger fired)")

    stats = report(trades, df)

    if trades:
        td = trades_to_df(trades, df)
        td.to_csv(f"{args.out}/trades.csv", index=False)
        print(f"\nTrade log -> {args.out}/trades.csv")

        plot_equity(trades, f"{args.out}/equity_curve.png")
        print(f"Equity curve -> {args.out}/equity_curve.png")

        plot_sample_trades(df, trades, f"{args.out}/sample_trades.png")
        print(f"Sample trades -> {args.out}/sample_trades.png")


if __name__ == "__main__":
    main()
