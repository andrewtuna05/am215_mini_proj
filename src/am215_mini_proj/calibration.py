
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom

try:
    from am215_mini_proj.models import BernoulliRandomWalk
except ImportError:
    from models import BernoulliRandomWalk

WINDOW, HORIZON, THRESH = 252, 21, 0.03
STEP = 5
N_PATHS = 10_000
EVAL_START = "2021-09-01"     # must equal BACKTEST_START in get_low_vol_stocks.py
BENCHMARK = "SPY"             # dividend-adjusted S&P 500 proxy, consistent with auto_adjust stock prices
BINS = 8                      # quantile bins for reliability diagram
BOOT_B, BLOCK = 2000, 6       # 6 forecast dates x 5 days = 30 trading days > horizon
SEED = 215
OUT = None  # set in main()

def fit_params(r):
    """p, delta_up, delta_down from a window of simple returns (zero counts as 'down')."""
    r = np.asarray(r)
    up = r > 0
    p = up.mean()
    d_up = r[up].mean() if up.any() else 0.0
    d_dn = r[~up].mean() if (~up).any() else 0.0
    return p, d_up, d_dn


def exact_prob(p, d_up, d_dn, n=HORIZON, thr=THRESH):
    """Exact P(V >= 1+thr): V depends only on k = #up days ~ Binomial(n, p)."""
    k = np.arange(n + 1)
    V = (1 + d_up) ** k * (1 + d_dn) ** (n - k)
    return binom.pmf(k, n, p)[V >= 1 + thr].sum()


def forecast_series(prices, name, rng, use_mc=False, eval_start=EVAL_START):
    """Rolling forecasts for one price series -> DataFrame of forecasts and outcomes."""
    prices = prices.dropna()
    ret = prices.pct_change().values
    past21 = prices.pct_change(HORIZON).values       # return over (i-21, i], known at i
    px, dates = prices.values, prices.index
    rows = []
    t0 = max(WINDOW, int(np.searchsorted(dates, pd.Timestamp(eval_start))))
    for t in range(t0, len(px) - HORIZON, STEP):
        p, du, dd = fit_params(ret[t - WINDOW + 1: t + 1])
        if use_mc:
            ph = BernoulliRandomWalk(p, du, dd).predict(px[t], THRESH, HORIZON, N_PATHS, rng)
        else:
            ph = exact_prob(p, du, dd)
        base = np.mean(past21[t - WINDOW + HORIZON: t + 1] >= THRESH)  # windows realised by t
        y = float(px[t + HORIZON] / px[t] - 1 >= THRESH)
        rows.append((dates[t], name, ph, base, y, p, du, dd))
    return pd.DataFrame(rows, columns=["date", "ticker", "p_hat", "baseline", "y",
                                       "p", "delta_up", "delta_down"])

def verify(rng, n_trials=8):
    """Monte Carlo from models.py vs exact binomial tail; limiting cases."""
    rows = []
    for _ in range(n_trials):
        p = rng.uniform(0.35, 0.65)
        du, dd = rng.uniform(0.002, 0.012), -rng.uniform(0.002, 0.012)
        ex = exact_prob(p, du, dd)
        mc = BernoulliRandomWalk(p, du, dd).predict(100.0, THRESH, HORIZON, 200_000, rng)
        se = np.sqrt(max(ex * (1 - ex), 1e-12) / 200_000)
        rows.append((p, du, dd, ex, mc, abs(mc - ex) / se))
    v = pd.DataFrame(rows, columns=["p", "d_up", "d_down", "exact", "mc", "abs_err_in_SE"])
    # limiting cases
    lim = {
        "p=1, 21 up-days of +1% -> 1": BernoulliRandomWalk(1.0, 0.01, -0.01).predict(100, rng=rng),
        "p=0 -> 0": BernoulliRandomWalk(0.0, 0.01, -0.01).predict(100, rng=rng),
    }
    # convergence ~ 1/sqrt(N)
    p, du, dd = 0.55, 0.008, -0.007
    ex = exact_prob(p, du, dd)
    conv = []
    for N in [100, 1_000, 10_000, 100_000]:
        errs = [abs(BernoulliRandomWalk(p, du, dd).predict(100, n_paths=N, rng=rng) - ex) for _ in range(200)]
        conv.append((N, np.sqrt(np.mean(np.square(errs)))))
    return v, lim, pd.DataFrame(conv, columns=["N", "rmse"]), ex

def add_bins(df, k=BINS):
    df = df.copy()
    df["bin"] = pd.qcut(df["p_hat"], k, labels=False, duplicates="drop")
    return df


def block_idx(T, B, block, rng):
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, max(T - block + 1, 1), size=(B, nb))
    return (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :T].clip(max=T - 1)


def ci(x, lo=2.5, hi=97.5):
    return np.percentile(x, [lo, hi])


def scores(df, rng, B=BOOT_B, block=BLOCK):
    """Brier (MSE) of model and baseline, skill, calibration-in-the-large + block-bootstrap CIs."""
    d = df.assign(se_m=(df.p_hat - df.y) ** 2, se_b=(df.baseline - df.y) ** 2)
    g = d.groupby("date").agg(se_m=("se_m", "sum"), se_b=("se_b", "sum"),
                              ph=("p_hat", "sum"), yy=("y", "sum"), n=("y", "size")).sort_index()
    arr = g.values
    s = arr.sum(0)
    idx = block_idx(len(arr), B, block, rng)
    bs = arr[idx].sum(1)
    N = bs[:, 4]
    out = {}
    for key, pt, bt in [("brier_model", s[0] / s[4], bs[:, 0] / N),
                        ("brier_baseline", s[1] / s[4], bs[:, 1] / N),
                        ("brier_diff (model-baseline)", (s[0] - s[1]) / s[4], (bs[:, 0] - bs[:, 1]) / N),
                        ("skill (1-Bm/Bb)", 1 - s[0] / s[1], 1 - bs[:, 0] / bs[:, 1]),
                        ("mean(p_hat)-mean(y)", (s[2] - s[3]) / s[4], (bs[:, 2] - bs[:, 3]) / N)]:
        lo, hi = ci(bt)
        out[key] = (pt, lo, hi)
    out["n_forecasts"] = int(s[4]); out["n_dates"] = len(arr)
    return out


def reliability(df, rng, B=BOOT_B, block=BLOCK):
    """Per-bin predicted mean, observed frequency, block-bootstrap CI, Brier decomposition."""
    d = add_bins(df)
    K = int(d["bin"].max()) + 1
    dates = np.sort(d["date"].unique())
    di = pd.Series(np.arange(len(dates)), index=dates)
    ti, kb = di[d["date"]].values, d["bin"].astype(int).values
    n = np.zeros((len(dates), K)); ys = np.zeros_like(n); ps = np.zeros_like(n)
    np.add.at(n, (ti, kb), 1); np.add.at(ys, (ti, kb), d["y"].values); np.add.at(ps, (ti, kb), d["p_hat"].values)
    idx = block_idx(len(dates), B, block, rng)
    bn, by = n[idx].sum(1), ys[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_freq = by / bn
    N = n.sum(0)
    tab = pd.DataFrame({"mean_pred": ps.sum(0) / N, "obs_freq": ys.sum(0) / N, "n": N.astype(int)})
    tab["lo"] = np.nanpercentile(boot_freq, 2.5, axis=0)
    tab["hi"] = np.nanpercentile(boot_freq, 97.5, axis=0)
    tab["gap"] = tab["mean_pred"] - tab["obs_freq"]
    tab["within_10pts"] = tab["gap"].abs() <= 0.10
    ybar = d["y"].mean(); w = tab["n"] / tab["n"].sum()
    dec = {"reliability": float((w * tab["gap"] ** 2).sum()),
           "resolution": float((w * (tab["obs_freq"] - ybar) ** 2).sum()),
           "uncertainty": float(ybar * (1 - ybar))}
    return tab, dec


def per_stock(df):
    g = df.groupby("ticker")
    t = pd.DataFrame({"mean_pred": g["p_hat"].mean(), "obs_freq": g["y"].mean(),
                      "brier_model": g.apply(lambda x: np.mean((x.p_hat - x.y) ** 2)),
                      "brier_base": g.apply(lambda x: np.mean((x.baseline - x.y) ** 2))})
    t["gap"] = t["mean_pred"] - t["obs_freq"]
    t["within_10pts"] = t["gap"].abs() <= 0.10
    t["skill"] = 1 - t["brier_model"] / t["brier_base"]
    return t.sort_values("gap")

def repo_root():
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for d in [start, *start.parents]:
            if (d / "pyproject.toml").exists():
                return d
    return Path.cwd()


def load_from_repo(data_dir):
    """Read the files written by get_low_vol_stocks.py, plus the benchmark."""
    data_dir = Path(data_dir)
    sel = pd.read_csv(data_dir / "lowest_30_stocks.csv")
    px = pd.read_csv(data_dir / "stock_prices.csv", index_col="Date", parse_dates=True).sort_index()
    px = px[[t for t in sel["Ticker"] if t in px.columns]]
    cache = data_dir / "sp500_benchmark.csv"
    if cache.exists():
        bench = pd.read_csv(cache, index_col="Date", parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf
        end = (px.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        bench = yf.download(BENCHMARK, start=px.index.min().strftime("%Y-%m-%d"), end=end,
                            auto_adjust=True, progress=False)["Close"].squeeze()
        bench.rename("Close").to_csv(cache, index_label="Date")
    return px, bench.sort_index(), sel.set_index("Ticker")["Annualized Vol"]


def load_synthetic(rng, n=30):
    idx = pd.bdate_range("2020-08-01", "2026-09-30")
    mkt = rng.normal(0.0004, 0.010, len(idx))
    sig = rng.uniform(0.006, 0.014, n)
    beta = rng.uniform(0.3, 1.1, n)
    r = mkt[:, None] * beta + rng.normal(0.0003, 1, (len(idx), n)) * sig
    px = pd.DataFrame(100 * np.cumprod(1 + r, 0), index=idx, columns=[f"S{i:02d}" for i in range(n)])
    return px, pd.Series(400 * np.cumprod(1 + mkt), index=idx), pd.Series(sig * np.sqrt(252), index=px.columns)


def check_history(px):
    """Each forecast needs WINDOW prior returns; warn if the saved prices start too late."""
    before = int((px.index < pd.Timestamp(EVAL_START)).sum())
    if before < WINDOW + 1:
        first = px.index[min(WINDOW, len(px) - 1)].date()
        print(f"WARNING: only {before} trading days of prices before {EVAL_START}; "
              f"the first forecast is {first}, not {EVAL_START}. In get_low_vol_stocks.py, save prices "
              f"from at least 2020-08-01 (change the 2021-08-01 slice in step 7).")

def savefig(name):
    plt.tight_layout(); plt.savefig(os.path.join(OUT, name), dpi=200); plt.close()


def fig_reliability(tab, df, title, name):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    m = max(tab.mean_pred.max(), tab.obs_freq.max(), tab.hi.max()) * 1.1
    xs = np.linspace(0, m, 50)
    ax[0].fill_between(xs, xs - 0.1, xs + 0.1, color="tab:green", alpha=0.12, label="±10 percentage points")
    ax[0].plot(xs, xs, "k--", lw=1, label="perfect calibration")
    ax[0].errorbar(tab.mean_pred, tab.obs_freq, yerr=[tab.obs_freq - tab.lo, tab.hi - tab.obs_freq],
                   fmt="o-", color="tab:blue", capsize=3, label="model (95% block-bootstrap CI)")
    ax[0].set_xlim(0, m); ax[0].set_ylim(0, m)
    ax[0].set_xlabel("Predicted P(≥3% gain in 21 trading days)"); ax[0].set_ylabel("Observed frequency")
    ax[0].set_title(title); ax[0].legend(fontsize=8, loc="upper left"); ax[0].set_aspect("equal")
    ax[1].hist(df.p_hat, bins=30, color="tab:blue", alpha=0.8)
    ax[1].set_xlabel("Predicted probability"); ax[1].set_ylabel("Number of forecasts")
    ax[1].set_title("Distribution of forecasts")
    savefig(name)


def fig_brier(sc_basket, sc_spx, name):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    labels = ["Low-vol basket", "S&P 500"]
    w = 0.35
    for j, (key, lab, c) in enumerate([("brier_model", "Bernoulli model", "tab:blue"),
                                       ("brier_baseline", "Historical-frequency baseline", "tab:gray")]):
        vals = [sc_basket[key][0], sc_spx[key][0]]
        err = np.array([[v - s[key][1], s[key][2] - v] for v, s in zip(vals, [sc_basket, sc_spx])]).T
        ax.bar(np.arange(2) + (j - 0.5) * w, vals, w, yerr=err, capsize=4, color=c, label=lab)
    ax.set_xticks(range(2)); ax.set_xticklabels(labels); ax.set_ylabel("Brier score / MSE (lower is better)")
    ax.set_title("Model vs no-model baseline (95% block-bootstrap CI)"); ax.legend(fontsize=8)
    savefig(name)


def fig_per_stock(ps, name):
    fig, ax = plt.subplots(figsize=(7, 8))
    colors = np.where(ps.gap.abs() <= 0.10, "tab:green", "tab:red")
    ax.barh(ps.index, ps.gap, color=colors)
    for x in (-0.1, 0.1):
        ax.axvline(x, color="k", ls="--", lw=1)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Mean predicted − observed frequency of ≥3% gain"); ax.set_title("Per-stock calibration gap")
    savefig(name)


def fig_time(df, spx_df, name):
    def monthly(d):
        g = d.assign(m=d.date.dt.to_period("M")).groupby("m")
        return g["p_hat"].mean(), g["y"].mean()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    p, y = monthly(df)
    x = p.index.to_timestamp()
    ax.plot(x, p, color="tab:blue", label="Basket: mean predicted")
    ax.plot(x, y, color="tab:orange", label="Basket: observed frequency")
    p2, y2 = monthly(spx_df)
    ax.plot(x[:len(p2)], y2.values[:len(x)], color="gray", ls=":", label="S&P 500: observed frequency")
    ax.set_ylabel("P(≥3% gain, 21d)"); ax.set_title("Predicted vs realised over time (regime sensitivity)")
    ax.legend(fontsize=8); savefig(name)


def fig_spx_overlay(tab_b, tab_s, name):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    m = max(tab_b.mean_pred.max(), tab_b.obs_freq.max(), tab_s.mean_pred.max(), tab_s.obs_freq.max()) * 1.1
    ax.plot([0, m], [0, m], "k--", lw=1)
    for tab, lab, c in [(tab_b, "Low-vol basket", "tab:blue"), (tab_s, "S&P 500", "tab:red")]:
        ax.errorbar(tab.mean_pred, tab.obs_freq, yerr=[tab.obs_freq - tab.lo, tab.hi - tab.obs_freq],
                    fmt="o-", color=c, capsize=3, label=lab)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability: basket vs S&P 500"); ax.legend(); ax.set_aspect("equal"); savefig(name)


def fig_verify(conv, name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(conv.N, conv.rmse, "o-", label="Monte Carlo RMSE vs exact")
    c = conv.rmse.iloc[0] * np.sqrt(conv.N.iloc[0])
    ax.loglog(conv.N, c / np.sqrt(conv.N), "k--", label="$1/\\sqrt{N}$")
    ax.set_xlabel("Paths N"); ax.set_ylabel("RMSE"); ax.set_title("Verification: convergence"); ax.legend()
    savefig(name)

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--mc", action="store_true", help="use Monte Carlo from models.py for every forecast (slow)")
    ap.add_argument("--data-dir", default=None, help="folder with lowest_30_stocks.csv and stock_prices.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = repo_root()
    OUT = a.out or str(root / "results"); os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 0. verification
    v, lim, conv, ex = verify(rng)
    v.to_csv(f"{OUT}/verification_mc_vs_exact.csv", index=False)
    conv.to_csv(f"{OUT}/verification_convergence.csv", index=False)
    fig_verify(conv, "fig6_verification.png")
    print("\nVERIFICATION\n", v.round(4).to_string(index=False))
    print("limiting cases:", lim, "| max |MC-exact| =", v.abs_err_in_SE.max().round(2), "SE")

    # 1. data (selection already done upstream, no look-ahead)
    px, spx, sel_vol = load_synthetic(rng) if a.synthetic else load_from_repo(a.data_dir or root / "data")
    check_history(px)
    print(f"\nLoaded {px.shape[1]} stocks, {px.index.min().date()} to {px.index.max().date()}")
    sel_vol.rename("annualized_vol_selection_window").to_csv(f"{OUT}/selected_stocks.csv")

    # 2. rolling forecasts
    frames = [forecast_series(px[t], t, rng, use_mc=a.mc) for t in px.columns]
    df = pd.concat(frames, ignore_index=True)
    spx_df = forecast_series(spx, "S&P500", rng, use_mc=a.mc)
    df.to_csv(f"{OUT}/forecasts_basket.csv", index=False)
    spx_df.to_csv(f"{OUT}/forecasts_spx.csv", index=False)

    # 3. scores, calibration
    sc, sc_s = scores(df, rng), scores(spx_df, rng)
    tab, dec = reliability(df, rng)
    tab_s, dec_s = reliability(spx_df, rng)
    ps = per_stock(df)
    tab.to_csv(f"{OUT}/reliability_basket.csv"); tab_s.to_csv(f"{OUT}/reliability_spx.csv")
    ps.to_csv(f"{OUT}/per_stock.csv")

    # 4. S&P comparison: beta, vol ratio, event frequencies
    mkt_all = spx.dropna().loc[EVAL_START:].pct_change().dropna()
    rows = {}
    for t in px.columns:
        r = px[t].dropna().loc[EVAL_START:].pct_change().dropna()
        m = mkt_all.reindex(r.index).dropna(); r = r.reindex(m.index)
        rows[t] = (np.cov(r, m)[0, 1] / m.var(), r.std() / m.std())
    comp = pd.DataFrame(rows, index=["beta_vs_SPX", "vol_ratio_vs_SPX"]).T
    comp.to_csv(f"{OUT}/beta_and_vol_ratio.csv")

    # 5. figures
    fig_reliability(tab, df, "Low-vol basket: reliability", "fig1_reliability_basket.png")
    fig_brier(sc, sc_s, "fig2_brier_model_vs_baseline.png")
    fig_per_stock(ps, "fig3_per_stock_gap.png")
    fig_time(df, spx_df, "fig4_time_series.png")
    fig_spx_overlay(tab, tab_s, "fig5_reliability_vs_spx.png")

    # 6. summary for report
    def show(name, s):
        print(f"\n{name}  (n={s['n_forecasts']} forecasts, {s['n_dates']} dates)")
        for k in ["brier_model", "brier_baseline", "brier_diff (model-baseline)", "skill (1-Bm/Bb)", "mean(p_hat)-mean(y)"]:
            pt, lo, hi = s[k]; print(f"  {k:30s} {pt: .4f}  [{lo: .4f}, {hi: .4f}]")
    show("LOW-VOL BASKET", sc); show("S&P 500", sc_s)
    print("\nBasket reliability table\n", tab.round(3).to_string())
    print("Brier decomposition (binned):", {k: round(x, 4) for k, x in dec.items()})
    print(f"\nBins within ±10 pts: {tab.within_10pts.mean():.0%} | "
          f"stocks within ±10 pts: {ps.within_10pts.mean():.0%} | "
          f"overall observed freq {df.y.mean():.3f} vs mean predicted {df.p_hat.mean():.3f}")
    print(f"S&P observed freq {spx_df.y.mean():.3f} vs mean predicted {spx_df.p_hat.mean():.3f}")
    print(f"Median beta of basket vs S&P: {comp.beta_vs_SPX.median():.2f}; "
          f"median vol ratio: {comp.vol_ratio_vs_SPX.median():.2f}")


if __name__ == "__main__":
    main()