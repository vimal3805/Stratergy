"""
fetch_data.py  —  Download real ES/MES futures 4H data for the Little RZY backtest.

Run this on your LOCAL machine (not in the restricted sandbox).
After it finishes, copy the output CSV back and run:

    python little_rzy_backtest.py --source csv --file es_4h.csv

=============================================================================
OPTION 1 — Yahoo Finance  (free, no account needed, ~2 years of 1H data)
=============================================================================
Requires:  pip install yfinance pandas

Usage:
    python fetch_data.py --source yahoo --out es_4h.csv

=============================================================================
OPTION 2 — Barchart.com  (free account, up to 5 years, clean OHLCV)
=============================================================================
1. Go to https://www.barchart.com/futures/quotes/ESM25/historical-download
2. Set: Interval=Hourly, Date Range=last 3-5 years, Format=CSV
3. Download and save as  barchart_raw.csv
4. Run:
    python fetch_data.py --source barchart --file barchart_raw.csv --out es_4h.csv

=============================================================================
OPTION 3 — Investing.com  (free, just export the chart data)
=============================================================================
1. Go to https://www.investing.com/indices/us-spx-500-futures-historical-data
   (or search for "S&P 500 Futures" then click Historical Data)
2. Select daily granularity → download CSV
   For 1H: use the TradingView export below instead.
3. Run:
    python fetch_data.py --source investing --file investing_raw.csv --out es_4h.csv

=============================================================================
OPTION 4 — TradingView  (best quality, requires Pro for 1H+ history export)
=============================================================================
1. Open ES1! or MES1! chart on TradingView at 1H
2. In Pine Script or via the Export button (chart → … → Export chart data)
3. Save as  tv_raw.csv
4. Run:
    python fetch_data.py --source tradingview --file tv_raw.csv --out es_4h.csv

=============================================================================
OPTION 5 — Databento  (paid, institutional quality, recommended for serious use)
=============================================================================
    pip install databento
    python fetch_data.py --source databento --key YOUR_KEY --out es_4h.csv

=============================================================================
"""

import argparse
import sys
import pandas as pd


def from_yahoo(out: str):
    import yfinance as yf
    print("Downloading ES=F from Yahoo Finance (1H, 730 days)...")
    df = yf.Ticker("ES=F").history(period="730d", interval="1h")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = [c.lower() for c in df.columns]
    df4h = df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    df4h.index.name = "datetime"
    df4h.to_csv(out)
    print(f"Saved {len(df4h)} 4H bars to {out}")


def from_barchart(file: str, out: str):
    """
    Barchart hourly CSV columns:
      Symbol, Time, Open, High, Low, Last, Change, %Change, Volume, Open Interest
    OR newer format:
      Time,Open,High,Low,Last,Change,%Chg,Volume,"Open Int"
    """
    df = pd.read_csv(file)
    df.columns = [c.strip().lower().replace(" ", "_").replace("%", "pct") for c in df.columns]

    # Normalize column names across Barchart format variants
    col_map = {}
    for c in df.columns:
        if c in ("time", "date", "datetime"):
            col_map[c] = "datetime"
        elif c in ("open",):
            col_map[c] = "open"
        elif c in ("high",):
            col_map[c] = "high"
        elif c in ("low",):
            col_map[c] = "low"
        elif c in ("last", "close", "price"):
            col_map[c] = "close"
        elif "vol" in c:
            col_map[c] = "volume"
    df = df.rename(columns=col_map)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].apply(
        lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
    ).dropna()

    # Resample to 4H if data is hourly (rows > 3000 = likely sub-4H)
    if len(df) > 3000:
        print(f"Input has {len(df)} bars — resampling to 4H...")
        df = df.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

    df.index.name = "datetime"
    df.to_csv(out)
    print(f"Saved {len(df)} 4H bars to {out}")


def from_investing(file: str, out: str):
    """
    Investing.com CSV columns:
      Date,Price,Open,High,Low,Vol.,Change %
    Price = close.  Vol. may use K/M suffixes.
    """
    df = pd.read_csv(file, thousands=",")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"date": "datetime", "price": "close", "vol.": "volume"})

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    def parse_vol(v):
        if isinstance(v, str):
            v = v.replace(",", "")
            if v.endswith("K"):
                return float(v[:-1]) * 1_000
            if v.endswith("M"):
                return float(v[:-1]) * 1_000_000
            if v == "-":
                return 0
        try:
            return float(v)
        except Exception:
            return 0

    df["volume"] = df["volume"].apply(parse_vol)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index.name = "datetime"
    df.to_csv(out)
    print(f"Saved {len(df)} bars to {out}")


def from_tradingview(file: str, out: str):
    """
    TradingView export CSV columns:
      time,open,high,low,close,Volume
    time is a Unix timestamp (seconds).
    """
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    if "time" in df.columns and df["time"].dtype != "object":
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    else:
        df["datetime"] = pd.to_datetime(df.get("time", df.get("date", df.get("datetime"))))

    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if len(df) > 3000:
        print(f"Input has {len(df)} bars — resampling to 4H...")
        df = df.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

    df.index.name = "datetime"
    df.to_csv(out)
    print(f"Saved {len(df)} 4H bars to {out}")


def from_databento(api_key: str, out: str):
    """
    Pulls ~2 years of ES continuous front-month 1H OHLCV from Databento,
    resamples to 4H.  Requires:  pip install databento
    """
    import databento as db
    import datetime as dt

    client = db.Historical(api_key)
    end = dt.date.today()
    start = end - dt.timedelta(days=730)

    print(f"Fetching ES 1H OHLCV from Databento ({start} → {end})...")
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["ES.c.0"],          # continuous front-month
        schema="ohlcv-1h",
        start=start.isoformat(),
        end=end.isoformat(),
    )
    df = data.to_df()
    df = df.rename(columns={
        "ts_event": "datetime",
        "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
    })
    # Databento prices are in fixed-point (divide by 1e9)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c] / 1e9

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    df4h = df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    df4h.index.name = "datetime"
    df4h.to_csv(out)
    print(f"Saved {len(df4h)} 4H bars to {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Download real ES futures data for Little RZY backtest"
    )
    parser.add_argument("--source", required=True,
                        choices=["yahoo", "barchart", "investing", "tradingview", "databento"])
    parser.add_argument("--file", help="Input raw CSV (for barchart/investing/tradingview)")
    parser.add_argument("--out", default="es_4h.csv", help="Output CSV path")
    parser.add_argument("--key", help="API key (for databento)")
    args = parser.parse_args()

    if args.source == "yahoo":
        from_yahoo(args.out)
    elif args.source == "barchart":
        if not args.file:
            sys.exit("--file required for barchart source")
        from_barchart(args.file, args.out)
    elif args.source == "investing":
        if not args.file:
            sys.exit("--file required for investing source")
        from_investing(args.file, args.out)
    elif args.source == "tradingview":
        if not args.file:
            sys.exit("--file required for tradingview source")
        from_tradingview(args.file, args.out)
    elif args.source == "databento":
        if not args.key:
            sys.exit("--key required for databento source")
        from_databento(args.key, args.out)


if __name__ == "__main__":
    main()
