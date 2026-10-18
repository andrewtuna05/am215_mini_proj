## AM215 / AM115

# Mini-Project: Random-Walk Stock Forecasts

Evaluate how well a Bernoulli random-walk model predicts a stock gaining at least 3% over 21 trading days.

## Setup

Requires Python 3.13+ and `uv`. Run all commands from the repository root.

```bash
uv sync
```

## Download data

```bash
uv run python -m am215_mini_proj.get_low_vol_stocks
```

Selects 30 low-volatility stocks using historical S&P 500 membership and pre-August 2021 returns. Saves `data/lowest_30_stocks.csv` and `data/stock_prices.csv`. Requires internet access; skip if these files are already available.

## Generate forecasts

```bash
uv run python scripts/run_forecast.py --all --rolling
```

Uses 252 training returns, a 21-trading-day horizon, 21-day forecast spacing, and 10,000 Monte Carlo paths per forecast. The first forecast follows the first 252 returns. Saves `results/backtest_all.csv`.

For one stock and its first historical forecast:

```bash
uv run python scripts/run_forecasts.py --ticker KO
```

## Evaluate calibration

```bash
uv run python -m am215_mini_proj.calibration --verify
```

Reads `results/backtest_all.csv` and the original price data. Saves outputs in `results/calibration/`:

- `reliability.png` and `reliability.csv`: calibration by probability bin.
- `brier_comparison.png` and `metrics.json`: model versus historical-baseline MSE and confidence intervals.
- `per_stock.csv` and `forecasts_with_baseline.csv`: detailed results.
- `verification_mc_vs_exact.csv`: Monte Carlo verification against exact model probabilities.
