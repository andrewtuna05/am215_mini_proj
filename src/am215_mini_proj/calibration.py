"""Evaluate the CSV from run_forecasts.py without regenerating predictions.

Run from the repository root:
  uv run python -m am215_mini_proj.calibration --verify

Baseline: fraction of fully observed horizon-length returns meeting the target,
using every possible start within that forecast's window + 1 training prices.
For window=252 and horizon=21, this uses 232 historical outcomes. Their overlap
is allowed; no outcome ending after the forecast date is included.

Uncertainty: percentile moving-block bootstrap of forecast dates, keeping all
stocks at each date together. Default blocks contain 6 forecast dates (about
126 trading days when step=21). Intervals describe evaluation uncertainty
conditional on the saved forecasts, not uncertainty in fitted parameters.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Installed at <repository>/src/am215_mini_proj/calibration.py.
ROOT = Path(__file__).resolve().parents[2]


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def load_and_add_baseline(forecast_path, price_path):
    df = pd.read_csv(forecast_path, parse_dates=["forecast_date"])
    required = {"ticker", "forecast_date", "predicted_probability",
                "realized_gain_target", "window_returns", "forecast_horizon", "target_gain"}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError(f"Need nonempty forecast CSV with columns {sorted(required)}.")
    if df.forecast_date.isna().any() or df.ticker.isna().any():
        raise ValueError("Forecast dates and tickers must not be missing.")
    if df.duplicated(["ticker", "forecast_date"]).any():
        raise ValueError("Duplicate ticker/forecast_date rows.")
    for col in ("predicted_probability", "realized_gain_target", "target_gain",
                "window_returns", "forecast_horizon"):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"{col} must be finite.")
    if not df.predicted_probability.between(0, 1).all():
        raise ValueError("Predicted probabilities must lie in [0, 1].")
    if not df.realized_gain_target.isin([0, 1]).all():
        raise ValueError("Realized outcomes must be 0 or 1.")
    for col in ("window_returns", "forecast_horizon"):
        if not ((df[col] > 0) & (df[col] == np.floor(df[col]))).all():
            raise ValueError(f"{col} must contain positive integers.")
        df[col] = df[col].astype(int)
    if (df.target_gain <= -1).any():
        raise ValueError("target_gain must exceed -1.")
    for col in ("window_returns", "forecast_horizon", "target_gain", "step_size"):
        if col in df and df[col].nunique() != 1:
            raise ValueError(f"Evaluate one {col} setting at a time.")
    prices = pd.read_csv(price_path, index_col="Date", parse_dates=["Date"]).sort_index()
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans or prices.index.has_duplicates:
        raise ValueError("Price dates must be valid and unique.")
    cache = {}
    for ticker in df.ticker.unique():
        if ticker not in prices:
            raise ValueError(f"Ticker absent from prices: {ticker}")
        values = prices[ticker].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{ticker}: prices must be finite, positive, and non-missing.")
        cache[ticker] = values
    baselines = []
    for row in df.itertuples(index=False):
        t = prices.index.get_indexer([row.forecast_date])[0]
        w, h = row.window_returns, row.forecast_horizon
        if t < w or t + h >= len(prices) or w < h:
            raise ValueError(f"{row.ticker}, {row.forecast_date}: insufficient history/horizon for baseline.")
        px = cache[row.ticker]
        history = px[t - w:t + 1]
        past_returns = (history[h:] - history[:-h]) / history[:-h]
        baselines.append(float(np.mean(past_returns >= row.target_gain)))
        # Same arithmetic as estimate_rolling_parameters().
        actual = (px[t + h] - px[t]) / px[t]
        if int(actual >= row.target_gain) != row.realized_gain_target:
            raise ValueError("Saved outcome disagrees with prices; use the original price CSV.")
        if hasattr(row, "s0") and not np.isclose(row.s0, px[t], rtol=1e-10, atol=1e-10):
            raise ValueError("Saved initial price disagrees with the supplied price CSV.")
    df["baseline"] = baselines
    return df.rename(columns={"forecast_date": "date", "predicted_probability": "p_hat",
                              "realized_gain_target": "y"}).sort_values(["date", "ticker"]).reset_index(drop=True)


def block_indices(T, B, block, rng):
    if T < 2:
        raise ValueError("Calibration intervals need at least two distinct forecast dates.")
    if block >= T:
        raise ValueError("--block must be smaller than the number of forecast dates.")
    starts = rng.integers(0, T - block + 1, size=(B, math.ceil(T / block)))
    return (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :T]


def evaluate(df, bins, B, block, rng):
    df = df.copy()
    # Equal-width bins are fixed before seeing outcomes and handle tied predictions.
    edges = np.linspace(0, 1, bins + 1)
    df["bin"] = pd.cut(df.p_hat, edges, labels=False, include_lowest=True).astype(int)
    dates = pd.Index(sorted(df.date.unique()))
    ti = dates.get_indexer(df.date)
    bi = df.bin.to_numpy()
    n = np.zeros((len(dates), bins))
    ps = np.zeros_like(n)
    ys = np.zeros_like(n)
    for dest, values in ((n, np.ones(len(df))), (ps, df.p_hat), (ys, df.y)):
        np.add.at(dest, (ti, bi), values)
    idx = block_indices(len(dates), B, block, rng)
    bn, bp, by = n[idx].sum(1), ps[idx].sum(1), ys[idx].sum(1)
    count = n.sum(0)
    rows = []
    for k in range(bins):
        if count[k] == 0:
            continue
        valid = bn[:, k] > 0
        freq = by[valid, k] / bn[valid, k]
        gaps = (bp[valid, k] - by[valid, k]) / bn[valid, k]
        pred = ps[:, k].sum() / count[k]
        obs = ys[:, k].sum() / count[k]
        lo, hi = np.percentile(freq, [2.5, 97.5])
        glo, ghi = np.percentile(gaps, [2.5, 97.5])
        rows.append(dict(bin=k, lower_edge=edges[k], upper_edge=edges[k + 1],
                         n=int(count[k]), mean_pred=pred, obs_freq=obs,
                         obs_lo=lo, obs_hi=hi, gap=pred - obs, gap_lo=glo, gap_hi=ghi,
                         within_10pts=bool(abs(pred - obs) <= 0.10),
                         gap_interval_within_10pts=bool(glo >= -0.10 and ghi <= 0.10),
                         bootstrap_replicates=int(valid.sum())))
    tab = pd.DataFrame(rows)
    df["se_model"] = (df.p_hat - df.y) ** 2
    df["se_baseline"] = (df.baseline - df.y) ** 2
    g = df.groupby("date", sort=True).agg(
        model=("se_model", "sum"), baseline=("se_baseline", "sum"),
        gap=("p_hat", "sum"), y=("y", "sum"), n=("y", "size"),
    ).to_numpy()
    total, boot = g.sum(0), g[idx].sum(1)
    point = {"brier_model": total[0] / total[4], "brier_baseline": total[1] / total[4],
             "brier_difference": (total[0] - total[1]) / total[4],
             "overall_gap": (total[2] - total[3]) / total[4]}
    boot_metrics = {"brier_model": boot[:, 0] / boot[:, 4],
                    "brier_baseline": boot[:, 1] / boot[:, 4],
                    "brier_difference": (boot[:, 0] - boot[:, 1]) / boot[:, 4],
                    "overall_gap": (boot[:, 2] - boot[:, 3]) / boot[:, 4]}
    summary = {"n_forecasts": len(df), "n_dates": len(dates), "n_stocks": df.ticker.nunique(),
               "bootstrap_replicates": B, "block_forecast_dates": block,
               "bins_requested": bins, "bins_occupied": len(tab),
               "first_forecast": str(dates.min().date()), "last_forecast": str(dates.max().date()),
               "metrics": {}}
    for name, value in point.items():
        lo, hi = np.percentile(boot_metrics[name], [2.5, 97.5])
        summary["metrics"][name] = {"estimate": float(value), "lo": float(lo), "hi": float(hi)}
    per_stock = df.groupby("ticker").agg(
        n=("y", "size"), mean_pred=("p_hat", "mean"), obs_freq=("y", "mean"),
        brier_model=("se_model", "mean"), brier_baseline=("se_baseline", "mean"),
    )
    per_stock["gap"] = per_stock.mean_pred - per_stock.obs_freq
    return df, tab, summary, per_stock


def exact_prob(p, du, dd, horizon=21, threshold=0.03):
    # No SciPy dependency: only horizon + 1 possible terminal values.
    return sum(math.comb(horizon, k) * p**k * (1 - p)**(horizon - k)
               for k in range(horizon + 1)
               if (1 + du)**k * (1 + dd)**(horizon - k) >= 1 + threshold)


def verify_model(out, seed):
    from am215_mini_proj.models import BernoulliRandomWalk
    rng = np.random.default_rng(seed)
    rows = []
    for p, du, dd in ((1.0, .01, -.01), (0.0, .01, -.01),
                      (.55, .008, -.007), (.4, .012, -.006), (.65, .005, -.009)):
        exact = exact_prob(p, du, dd)
        mc = float(BernoulliRandomWalk(p, du, dd).predict(100, n_paths=200_000, rng=rng))
        se = math.sqrt(exact * (1 - exact) / 200_000)
        passed = abs(mc - exact) <= max(5 * se, 1e-12)
        rows.append(dict(p=p, delta_up=du, delta_down=dd, exact=exact,
                         monte_carlo=mc, mc_se=se, within_5se=passed))
    table = pd.DataFrame(rows)
    table.to_csv(out / "verification_mc_vs_exact.csv", index=False)
    if not table.within_5se.all():
        raise ValueError("Monte Carlo verification failed; inspect verification_mc_vs_exact.csv.")
    print("Verification: deterministic limits and exact binomial comparison passed.")


def plot_results(df, tab, summary, out):
    """Plot saved evaluation estimates; sparse bins are descriptive only."""
    from matplotlib.ticker import PercentFormatter

    sparse_cutoff = 20  # Display flag, not a statistical significance threshold.
    edges = np.linspace(0, 1, summary["bins_requested"] + 1)
    blue, orange = "#2463A6", "#B76517"
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.7))
    ax = axes[0]
    x = np.linspace(0, 1, 200)
    ax.fill_between(x, np.maximum(0, x - .1), np.minimum(1, x + .1),
                    alpha=.12, color="green", label="±10 percentage points")
    ax.plot(x, x, "k--", lw=1, label="Perfect calibration")
    for sparse in (False, True):
        rows = tab.loc[(tab.n < sparse_cutoff) == sparse]
        if rows.empty:
            continue
        color = orange if sparse else blue
        # Suppress sparse-bin intervals: percentile bootstrap intervals can
        # collapse at 0 or 1 and suggest unwarranted precision.
        if not sparse:
            ax.vlines(rows.mean_pred, rows.obs_lo, rows.obs_hi, color=color, lw=1.5)
        ax.scatter(rows.mean_pred, rows.obs_freq, s=48,
                   facecolors="white" if sparse else color, edgecolors=color,
                   linewidths=1.5, zorder=3,
                   label="Sparse bin (n < 20); interval omitted" if sparse else
                         "Bin estimate; 95% pointwise interval")
    for row in tab.itertuples():
        ax.annotate(f"n={row.n:,}", (row.mean_pred, row.obs_freq),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(-.045, 1.03), xlabel="Mean predicted probability",
           ylabel="Observed target-event frequency",
           title="Calibration by probability bin")
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=.18)

    counts = np.zeros(len(edges) - 1, dtype=int)
    counts[tab.bin.to_numpy(dtype=int)] = tab.n.to_numpy(dtype=int)
    bars = axes[1].bar(edges[:-1], counts, width=np.diff(edges), align="edge",
                       color=[orange if 0 < n < sparse_cutoff else blue for n in counts],
                       edgecolor="white", alpha=.85)
    axes[1].bar_label(bars, labels=[f"{n:,}" for n in counts], padding=4, fontsize=9)
    axes[1].set(xlim=(0, 1), ylim=(0, max(1, counts.max()) * 1.18),
                xlabel="Predicted probability (same bins as calibration)",
                ylabel="Number of forecasts", title="Forecast counts by probability bin")
    axes[1].set_xticks(edges)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].grid(axis="y", alpha=.18)
    axes[1].set_axisbelow(True)
    fig.text(.07, .035,
             "Intervals: 95% pointwise block-bootstrap intervals. Hollow orange dots flag fewer than 20 forecasts;\n"
             "this display cutoff does not guarantee precision in other bins. Empty bins have no calibration dot.",
             fontsize=9, va="bottom")
    fig.tight_layout(rect=(0, .13, 1, 1))
    fig.savefig(out / "reliability.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.3))
    maximum = max(summary["metrics"][key]["hi"] for key in
                  ("brier_model", "brier_baseline"))
    for i, key in enumerate(("brier_model", "brier_baseline")):
        m = summary["metrics"][key]
        ax.bar(i, m["estimate"], width=.6, color=[blue, "#888888"][i])
        ax.vlines(i, m["lo"], m["hi"], color="black")
        ax.hlines([m["lo"], m["hi"]], i - .06, i + .06, color="black")
        ax.text(i, max(m["hi"], m["estimate"]) + maximum * .035,
                f'{m["estimate"]:.4f}', ha="center", fontsize=11)
    ax.set_xticks([0, 1], ["Bernoulli model", "Historical frequency"])
    ax.set(ylim=(0, maximum * 1.2), ylabel="MSE / Brier score (lower is better)",
           title="Model versus historical baseline")
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    d = summary["metrics"]["brier_difference"]
    conclusion = ("Interval includes zero: no clear difference." if d["lo"] <= 0 <= d["hi"] else
                  "Interval below zero: favors the model." if d["hi"] < 0 else
                  "Interval above zero: favors the baseline.")
    fig.text(.13, .035,
             f'Model − baseline = {d["estimate"]:+.4f}\n'
             f'95% paired block-bootstrap interval: [{d["lo"]:+.4f}, {d["hi"]:+.4f}]\n'
             f'{conclusion}\nBlack lines show 95% intervals for each score.', fontsize=9, va="bottom")
    fig.tight_layout(rect=(0, .23, 1, 1))
    fig.savefig(out / "brier_comparison.png", dpi=180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, default=ROOT / "results/backtest_all.csv")
    parser.add_argument("--data", type=Path, default=ROOT / "data/stock_prices.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results/calibration")
    parser.add_argument("--bins", type=positive_int, default=8)
    parser.add_argument("--bootstrap", type=positive_int, default=2000)
    parser.add_argument("--block", type=positive_int, default=6)
    parser.add_argument("--seed", type=int, default=215)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative.")
    try:
        # Protect input paths from accidental replacement by an output file.
        output_names = ("forecasts_with_baseline.csv", "reliability.csv", "metrics.json",
                        "per_stock.csv", "reliability.png", "brier_comparison.png",
                        "verification_mc_vs_exact.csv")
        inputs = {args.forecasts.resolve(), args.data.resolve()}
        if any((args.out / name).resolve() in inputs for name in output_names):
            raise ValueError("Output files must differ from input files.")
        df = load_and_add_baseline(args.forecasts, args.data)
        df, tab, summary, stocks = evaluate(
            df, args.bins, args.bootstrap, args.block, np.random.default_rng(args.seed),
        )
        summary["bootstrap_seed"] = args.seed
        summary["binning"] = "equal-width over [0,1]; empty bins omitted"
        args.out.mkdir(parents=True, exist_ok=True)
        if args.verify:
            verify_model(args.out, args.seed)
        df.to_csv(args.out / "forecasts_with_baseline.csv", index=False, date_format="%Y-%m-%d")
        tab.to_csv(args.out / "reliability.csv", index=False)
        stocks.to_csv(args.out / "per_stock.csv")
        (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        plot_results(df, tab, summary, args.out)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(json.dumps(summary, indent=2))
    print(f"Saved calibration outputs to {args.out}")
    print("within_10pts describes estimated bin gaps; intervals are pointwise, not simultaneous.")
    print("Per-stock summaries are descriptive and do not have confidence intervals.")


if __name__ == "__main__":
    main()
