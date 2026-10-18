import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Configuration
SELECTION_START = "2016-08-01"
SELECTION_END = "2021-08-01"  # Exclusive
PRICE_END = "2026-10-01"      # Exclusive
NUM_STOCKS = 30
BATCH_SIZE = 50

# Root-level data directory
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_historical_constituents():
    """Get S&P 500 constituents from July 2021."""
    url = (
        "https://raw.githubusercontent.com/"
        "pierrebrunelle/sp500-historical-constituents/"
        "main/data/spy202107"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    constituents = pd.read_csv(
        io.StringIO(response.text),
        sep="\t",
        header=None,
        names=["Ticker", "Company", "Weight"],
    )

    # Convert ticker notation for Yahoo Finance
    constituents["Ticker"] = (
        constituents["Ticker"]
        .str.replace("/", "-", regex=False)
        .str.replace(".", "-", regex=False)
    )

    return constituents


def download_prices(tickers, start, end):
    """Download adjusted closing prices in batches."""
    batches = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]

        print(
            f"Downloading batch "
            f"{i // BATCH_SIZE + 1}/"
            f"{(len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE}"
        )

        try:
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if data.empty:
                continue

            close = data["Close"]

            if isinstance(close, pd.Series):
                close = close.to_frame(name=batch[0])

            batches.append(close)

        except Exception as exc:
            print(f"Batch failed: {exc}")

    if not batches:
        raise RuntimeError("No stock prices downloaded.")

    prices = pd.concat(batches, axis=1)
    return prices.sort_index()


def main():
    # 1. Historical S&P 500 constituents
    constituents = get_historical_constituents()

    tickers = constituents["Ticker"].tolist()
    ticker_to_name = dict(
        zip(constituents["Ticker"], constituents["Company"])
    )

    print(f"Loaded {len(tickers)} historical S&P 500 tickers.")

    # 2. Download selection and evaluation data
    prices = download_prices(
        tickers,
        start=SELECTION_START,
        end=PRICE_END,
    )

    # 3. Restrict selection to pre-backtest data
    selection_prices = prices.loc[
        SELECTION_START:"2021-07-31"
    ]

    # Require complete selection-window histories
    selection_prices = selection_prices.dropna(axis=1)

    # 4. Calculate annualized historical volatility
    returns = selection_prices.pct_change(fill_method=None)

    volatility = returns.std() * np.sqrt(252)
    volatility = volatility.dropna().sort_values()

    if len(volatility) < NUM_STOCKS:
        raise RuntimeError(
            f"Only {len(volatility)} eligible stocks found; "
            f"need {NUM_STOCKS}."
        )

    # 5. Select the lowest-volatility stocks
    lowest_30 = volatility.head(NUM_STOCKS)

    summary_df = pd.DataFrame({
        "Rank": range(1, NUM_STOCKS + 1),
        "Ticker": lowest_30.index,
        "Company": [
            ticker_to_name.get(t, "N/A")
            for t in lowest_30.index
        ],
        "Annualized Vol": lowest_30.values.round(4),
        "Vol (%)": (
            (lowest_30.values * 100)
            .round(2)
            .astype(str) + "%"
        ),
    })

    # 6. Save selected stock information
    summary_path = DATA_DIR / "lowest_30_stocks.csv"
    summary_df.to_csv(summary_path, index=False)

    # 7. Save prices from August 2021 onward
    selected_prices = prices.loc[
        "2021-08-01":"2026-09-30",
        lowest_30.index,
    ]

    price_path = DATA_DIR / "stock_prices.csv"
    selected_prices.to_csv(
        price_path,
        index_label="Date",
    )

    # 8. Report missing evaluation data
    missing = selected_prices.isna().sum()
    if missing.any():
        print("\nMissing price observations by stock:")
        print(missing[missing > 0].to_string())

    print("\n30 Lowest-Volatility S&P 500 Stocks:")
    print(summary_df.to_string(index=False))

    print(f"\nSaved stock list: {summary_path}")
    print(f"Saved stock prices: {price_path}")
    print("Forecasting begins after the first 252 daily returns (default training window).")


if __name__ == "__main__":
    main()
