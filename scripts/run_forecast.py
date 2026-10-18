"""Run one historical forecast or a rolling backtest using the AM215 package."""

import argparse
from pathlib import Path
import zlib

import numpy as np
import pandas as pd

from am215_mini_proj.models import (
    BernoulliRandomWalk,
    estimate_rolling_parameters,
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def forecast_stock(series, ticker, args):
    """Return evaluated forecasts for one stock, using only past prices to fit."""
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{ticker}: prices must be finite, positive, and non-missing.")

    required = args.window + args.horizon + 1
    if len(series) < required:
        raise ValueError(f"{ticker}: need at least {required} prices, got {len(series)}.")

    # The first forecast uses window + 1 prices, hence exactly window returns.
    # In single mode, truncate after the first observable outcome.
    selected = series if args.rolling else series.iloc[:required]
    windows = estimate_rolling_parameters(
        selected,
        window_size=args.window,
        step_size=args.step,
        forecast_horizon=args.horizon,
        target_gain=args.threshold,
    )
    windows = windows.dropna(subset=["actual_return"]).copy()

    # A stable stream per ticker keeps its results unchanged in an --all run.
    ticker_id = zlib.crc32(ticker.encode("utf-8"))
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, ticker_id]))
    probabilities = []
    for row in windows.itertuples(index=False):
        model = BernoulliRandomWalk(row.p, row.delta_up, row.delta_down)
        probabilities.append(float(model.predict(
            s0=row.s0,
            threshold=args.threshold,
            n_days=args.horizon,
            n_paths=args.paths,
            rng=rng,
        )))

    windows.insert(0, "ticker", ticker)
    windows = windows.rename(columns={"p": "p_up"})
    windows["predicted_probability"] = probabilities
    windows["realized_gain_target"] = windows["realized_gain_target"].astype(int)
    windows["squared_error"] = (
        windows["predicted_probability"] - windows["realized_gain_target"]
    ) ** 2
    windows["window_returns"] = args.window
    windows["forecast_horizon"] = args.horizon
    windows["target_gain"] = args.threshold
    windows["n_paths"] = args.paths
    windows["seed"] = args.seed
    windows["step_size"] = args.step
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ticker", help="One ticker; defaults to KO.")
    selection.add_argument("--all", action="store_true", help="Use every price column.")
    parser.add_argument("--data", type=Path, default=Path("data/stock_prices.csv"))
    parser.add_argument("--rolling", action="store_true", help="Evaluate all available windows.")
    parser.add_argument("--window", type=positive_int, default=252, help="Number of training returns.")
    parser.add_argument("--horizon", type=positive_int, default=21)
    parser.add_argument("--step", type=positive_int, default=21, help="Trading rows between forecasts.")
    parser.add_argument("--paths", type=positive_int, default=10_000)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Output CSV path; parent folders are created.")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative.")
    if not np.isfinite(args.threshold) or args.threshold <= -1:
        parser.error("--threshold must be finite and greater than -1.")

    try:
        prices = pd.read_csv(args.data, index_col="Date", parse_dates=["Date"])
        if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
            raise ValueError("Date must contain valid dates.")
        if prices.index.has_duplicates:
            raise ValueError("Date must not contain duplicate dates.")
        prices = prices.sort_index()
        tickers = prices.columns.tolist() if args.all else [args.ticker or "KO"]
        if not tickers:
            raise ValueError("No stock columns found.")
        missing = [ticker for ticker in tickers if ticker not in prices.columns]
        if missing:
            raise ValueError(f"Ticker(s) absent from price CSV: {', '.join(missing)}")

        tables = []
        for ticker in tickers:
            print(f"Running {ticker}...", flush=True)
            tables.append(forecast_stock(prices[ticker], ticker, args))
        results = pd.concat(tables, ignore_index=True)

        mode = "backtest" if args.rolling else "forecast"
        label = "all" if args.all else tickers[0]
        output = args.output or Path("results") / f"{mode}_{label}.csv"
        if output.resolve() == args.data.resolve():
            raise ValueError("Output must differ from the input data path.")
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False, date_format="%Y-%m-%d")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")

    summary = results.groupby("ticker", sort=False).agg(
        forecasts=("predicted_probability", "size"),
        mean_probability=("predicted_probability", "mean"),
        observed_frequency=("realized_gain_target", "mean"),
        mse=("squared_error", "mean"),
    )
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved {len(results)} rows to {output}")
    if not args.rolling:
        print("Single mode uses the earliest complete historical window per stock.")
    print("MSE is the Brier score; calibration requires grouping forecasts into probability bins.")


if __name__ == "__main__":
    main()
