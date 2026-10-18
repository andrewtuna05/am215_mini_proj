"""Generate historical Monte Carlo forecasts using repository data and results folders.

Example:
  uv run python scripts/run_forecasts.py --all --rolling
The first forecast follows the first window of training returns. Forecasts
are spaced by trading rows; no calendar evaluation start date is imposed.
"""
import argparse
from pathlib import Path
import zlib

import numpy as np
import pandas as pd

from am215_mini_proj.models import BernoulliRandomWalk, estimate_rolling_parameters


# Installed at <repository>/scripts/run_forecasts.py.
ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def load_prices(path):
    prices = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Date must contain valid dates.")
    if prices.index.has_duplicates:
        raise ValueError("Date must not contain duplicate dates.")
    return prices.sort_index()


def forecast_stock(series, ticker, args):
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{ticker}: prices must be finite, positive, and non-missing.")
    required = args.window + args.horizon + 1
    if len(series) < required:
        raise ValueError(f"{ticker}: need at least {required} prices, got {len(series)}.")
    # Exactly window + 1 prices provide window training returns.
    selected = series if args.rolling else series.iloc[:required]
    windows = estimate_rolling_parameters(
        selected, window_size=args.window, step_size=args.step,
        forecast_horizon=args.horizon, target_gain=args.threshold,
    ).dropna(subset=["actual_return"]).copy()
    ticker_id = zlib.crc32(ticker.encode("utf-8"))
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, ticker_id]))
    probabilities = []
    for row in windows.itertuples(index=False):
        model = BernoulliRandomWalk(row.p, row.delta_up, row.delta_down)
        probabilities.append(float(model.predict(
            s0=row.s0, threshold=args.threshold, n_days=args.horizon,
            n_paths=args.paths, rng=rng,
        )))
    windows.insert(0, "ticker", ticker)
    windows = windows.rename(columns={"p": "p_up"})
    windows["predicted_probability"] = probabilities
    windows["realized_gain_target"] = windows["realized_gain_target"].astype(int)
    windows["squared_error"] = (
        windows["predicted_probability"] - windows["realized_gain_target"]
    ) ** 2
    for key, value in {
        "window_returns": args.window, "forecast_horizon": args.horizon,
        "target_gain": args.threshold, "n_paths": args.paths,
        "seed": args.seed, "step_size": args.step,
    }.items():
        windows[key] = value
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ticker", help="One ticker; defaults to KO.")
    selection.add_argument("--all", action="store_true", help="Use every stock price column.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/stock_prices.csv")
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--window", type=positive_int, default=252)
    parser.add_argument("--horizon", type=positive_int, default=21)
    parser.add_argument("--step", type=positive_int, default=21)
    parser.add_argument("--paths", type=positive_int, default=10_000)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative.")
    if not np.isfinite(args.threshold) or args.threshold <= -1:
        parser.error("--threshold must be finite and greater than -1.")
    try:
        prices = load_prices(args.data)
        tickers = prices.columns.tolist() if args.all else [args.ticker or "KO"]
        if not tickers:
            raise ValueError("No stock columns found.")
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise ValueError(f"Ticker(s) absent from price CSV: {', '.join(missing)}")
        tables = []
        for ticker in tickers:
            print(f"Running {ticker}...", flush=True)
            tables.append(forecast_stock(prices[ticker], ticker, args))
        results = pd.concat(tables, ignore_index=True)
        mode = "backtest" if args.rolling else "forecast"
        output = args.output or ROOT / "results" / f"{mode}_{'all' if args.all else tickers[0]}.csv"
        if output.resolve() == args.data.resolve():
            raise ValueError("Output must differ from input data.")
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False, date_format="%Y-%m-%d")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    summary = results.groupby("ticker", sort=False).agg(
        forecasts=("predicted_probability", "size"),
        mean_probability=("predicted_probability", "mean"),
        observed_frequency=("realized_gain_target", "mean"),
        mse=("squared_error", "mean"),
    )
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {len(results)} rows to {output}")
    if not args.rolling:
        print("Single mode uses the first eligible historical forecast per stock.")


if __name__ == "__main__":
    main()
