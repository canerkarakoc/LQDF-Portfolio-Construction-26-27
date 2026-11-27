"""
MVO with Bayes-Stein estimates, a short-selling constraint and a maximum weight cap
==================================================================================
Also plots the Markowitz efficient frontier (mvo_results/efficient_frontier.png).

Strategies compared out of sample (walk-forward, no hindsight):
    1. Equal weight (1/N)
    2. Judgement-based portfolio (fixed weights chosen by the team)
    3. GMV                              - minimum variance, ignores expected returns
    4. Max-Sharpe, sample means         - classic MVO
    5. Max-Sharpe, Bayes-Stein means    - Jorion (1986) shrinkage of the means

Constraints applied to every MVO portfolio:
    - Fully invested:        weights sum to 100%
    - No short selling:      every weight >= 0              (SHORT_SELLING = False)
    - Maximum weight cap:    every weight <= MAX_WEIGHT     (limits concentration)

At each month the models see ONLY the previous LOOKBACK months of returns, choose
weights, and are scored on the following month's actual return.

Usage:
    python mvo_bayes_stein_short_selling.py              # demo on synthetic data
    python mvo_bayes_stein_short_selling.py prices.csv   # Date column + one price
                                                         # column per asset in ASSETS
Requires numpy, pandas, scipy, matplotlib. Results are saved to ./mvo_results/
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
SHOW_PLOTS = False              # True = also open each chart in a window (or inline in Jupyter)
if not SHOW_PLOTS and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")       # save charts to files only, no windows
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =====================================================================
# 1. SETTINGS - agree these with the team BEFORE looking at any results
# =====================================================================
ASSETS = [                      # column names in the prices CSV (one currency, e.g. GBP)
    "UK Equity", "US Equity", "Europe ex-UK Equity", "Emerging Markets Equity",
    "UK Gilts", "Global Corporate Bonds", "UK Index-Linked Gilts", "Gold",
]
SHORT_SELLING = False           # False = no short selling (all weights >= 0)
MAX_WEIGHT = 0.35               # maximum weight in any single asset (set to 1.0 to switch off)
LOOKBACK = 60                   # estimation window T, in months
RF_ANNUAL = 0.0                 # annual risk-free rate, e.g. average UK T-bill yield
TC_BPS = 10                     # trading cost per unit of turnover, in basis points
PERIODS_PER_YEAR = 12

# Judgement-based portfolio: PLACEHOLDER - replace with the team's own weights,
# chosen using only information available at the start of the backtest.
MANUAL_WEIGHTS = {
    "UK Equity": 0.15, "US Equity": 0.20, "Europe ex-UK Equity": 0.10,
    "Emerging Markets Equity": 0.05, "UK Gilts": 0.20,
    "Global Corporate Bonds": 0.15, "UK Index-Linked Gilts": 0.10, "Gold": 0.05,
}
OUT_DIR = "mvo_results"


# =====================================================================
# 2. ESTIMATION
# =====================================================================
def sample_moments(window):
    """Per-period sample means and sample covariance matrix (divides by T - 1)."""
    return window.mean(axis=0), np.cov(window, rowvar=False)


def bayes_stein(mu, S, T):
    """Jorion (1986) Bayes-Stein estimator. Inputs are PER-PERIOD; T = number of periods.
        mu_BS = (1 - delta) * mu + delta * mu_0
        mu_0  = return of the (unconstrained) minimum-variance portfolio
        delta = (n + 2) / [(n + 2) + T (mu - mu_0 1)' S^-1 (mu - mu_0 1)]
    """
    n = len(mu)
    s_inv_1 = np.linalg.solve(S, np.ones(n))
    mu_0 = (s_inv_1 / s_inv_1.sum()) @ mu
    d = mu - mu_0
    delta = (n + 2) / ((n + 2) + T * (d @ np.linalg.solve(S, d)))
    delta = float(np.clip(delta, 0.0, 1.0))
    return (1 - delta) * mu + delta * mu_0, delta


# =====================================================================
# 3. CONSTRAINTS AND OPTIMISERS
# =====================================================================
def weight_bounds():
    """Short-selling constraint (lower bound) and maximum weight cap (upper bound)."""
    lower = 0.0 if not SHORT_SELLING else -MAX_WEIGHT
    return lower, MAX_WEIGHT


FULLY_INVESTED = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}


def solve_gmv(S, lower, upper):
    """Minimise w' S w  subject to  sum(w) = 1  and  lower <= w_i <= upper."""
    n = len(S)
    res = minimize(lambda w: w @ S @ w, np.full(n, 1 / n), jac=lambda w: 2 * S @ w,
                   method="SLSQP", bounds=[(lower, upper)] * n, constraints=[FULLY_INVESTED],
                   options={"ftol": 1e-14, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"GMV optimisation failed: {res.message}")
    return _clean(res.x)


def solve_max_sharpe(mu, S, rf, lower, upper):
    """Maximise (w' mu - rf) / sqrt(w' S w)  subject to  sum(w) = 1  and the bounds.
    Tries several starting points and keeps the best. If no allowed portfolio has a
    positive expected excess return, returns the GMV portfolio instead.
    Returns (weights, fell_back_to_gmv)."""
    n = len(mu)
    ex = mu - rf
    if ex.max() <= 0:
        return solve_gmv(S, lower, upper), True
    neg_sharpe = lambda w: -(w @ ex) / np.sqrt(w @ S @ w)
    starts = [np.full(n, 1 / n), solve_gmv(S, lower, upper)]
    best = None
    for w0 in starts:
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=[(lower, upper)] * n,
                       constraints=[FULLY_INVESTED], options={"ftol": 1e-14, "maxiter": 1000})
        if res.success and -res.fun > 0 and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return solve_gmv(S, lower, upper), True
    return _clean(best.x), False


def _clean(w):
    w = np.where(np.abs(w) < 1e-10, 0.0, w)
    return w / w.sum()


def solve_target_return(mu, S, target, lower, upper):
    """Minimum-variance portfolio for a given target return (one point on the frontier)."""
    n = len(mu)
    target_con = {"type": "eq", "fun": lambda w: w @ mu - target, "jac": lambda w: mu}
    res = minimize(lambda w: w @ S @ w, np.full(n, 1 / n), jac=lambda w: 2 * S @ w,
                   method="SLSQP", bounds=[(lower, upper)] * n, constraints=[FULLY_INVESTED, target_con],
                   options={"ftol": 1e-14, "maxiter": 1000})
    return _clean(res.x) if res.success else None


def efficient_frontier(mu, S, lower, upper, points=60):
    """Constrained efficient frontier: from the GMV portfolio up to the highest
    return achievable within the weight bounds. Returns arrays (volatility, return)."""
    n = len(mu)
    w_gmv = solve_gmv(S, lower, upper)
    top = minimize(lambda w: -(w @ mu), np.full(n, 1 / n), method="SLSQP",
                   bounds=[(lower, upper)] * n, constraints=[FULLY_INVESTED])
    vols, rets = [], []
    for target in np.linspace(w_gmv @ mu, _clean(top.x) @ mu, points):
        w = solve_target_return(mu, S, target, lower, upper)
        if w is not None:
            vols.append(np.sqrt(w @ S @ w)); rets.append(w @ mu)
    return np.array(vols), np.array(rets)


def unconstrained_frontier(mu, S, targets):
    """Closed-form Markowitz frontier with short selling allowed and no weight cap:
        sigma_p^2 = (A m^2 - 2 B m + C) / D."""
    inv = np.linalg.inv(S)
    one = np.ones(len(mu))
    A, B, C = one @ inv @ one, one @ inv @ mu, mu @ inv @ mu
    return np.sqrt((A * targets ** 2 - 2 * B * targets + C) / (A * C - B ** 2))


# =====================================================================
# 4. STRATEGIES: each maps (per-period mu, S, T) -> (weights, info)
# =====================================================================
def make_strategies(names):
    ppy, rf = PERIODS_PER_YEAR, RF_ANNUAL
    lower, upper = weight_bounds()
    n = len(names)
    manual = np.array([MANUAL_WEIGHTS[a] for a in names])
    assert abs(manual.sum() - 1) < 1e-9, "MANUAL_WEIGHTS must sum to 1"
    assert n * upper >= 1, "MAX_WEIGHT too small for the weights to sum to 100%"

    def max_sharpe_sample(mu, S, T):
        w, fb = solve_max_sharpe(mu * ppy, S * ppy, rf, lower, upper)
        return w, {"fallback": fb}

    def max_sharpe_bayes_stein(mu, S, T):
        mu_bs, delta = bayes_stein(mu, S, T)           # per-period inputs, T in periods
        w, fb = solve_max_sharpe(mu_bs * ppy, S * ppy, rf, lower, upper)
        return w, {"fallback": fb, "delta": delta}

    return {
        "Equal weight": lambda mu, S, T: (np.full(n, 1 / n), {}),
        "Judgement": lambda mu, S, T: (manual.copy(), {}),
        "GMV": lambda mu, S, T: (solve_gmv(S * ppy, lower, upper), {}),
        "Max Sharpe (sample)": max_sharpe_sample,
        "Max Sharpe (Bayes-Stein)": max_sharpe_bayes_stein,
    }


# =====================================================================
# 5. WALK-FORWARD BACKTEST
# =====================================================================
def backtest(returns, strategies):
    R = returns.values
    dates = returns.index[LOOKBACK:]
    tc = TC_BPS / 1e4
    out = {k: {"ret": [], "w": [], "turnover": [], "delta": [], "fallbacks": 0} for k in strategies}
    held = {k: None for k in strategies}                  # weights after drift
    for t in range(LOOKBACK, len(R)):
        mu, S = sample_moments(R[t - LOOKBACK:t])          # ONLY past data
        r_next = R[t]
        for name, strat in strategies.items():
            w, info = strat(mu, S, LOOKBACK)
            old = w if held[name] is None else held[name]  # no cost for the first purchase
            turnover = np.abs(w - old).sum()
            gross = w @ r_next
            out[name]["ret"].append(gross - tc * turnover)
            out[name]["w"].append(w)
            out[name]["turnover"].append(turnover)
            out[name]["delta"].append(info.get("delta", np.nan))
            out[name]["fallbacks"] += int(info.get("fallback", False))
            held[name] = w * (1 + r_next) / (1 + gross)     # weights drift until next rebalance
    for name in out:
        out[name]["ret"] = pd.Series(out[name]["ret"], index=dates)
        out[name]["w"] = pd.DataFrame(out[name]["w"], index=dates, columns=returns.columns)
    return out


# =====================================================================
# 6. PERFORMANCE, DIVERSIFICATION AND DOWNSIDE-RISK METRICS
# =====================================================================
def metrics(res):
    ppy = PERIODS_PER_YEAR
    lower, upper = weight_bounds()
    r = res["ret"]
    ex = r - RF_ANNUAL / ppy
    wealth = pd.concat([pd.Series([1.0]), (1 + r).cumprod().reset_index(drop=True)])
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    W = res["w"].values
    at_bound = (np.abs(W - lower) < 1e-4) | (np.abs(W - upper) < 1e-4)
    return {
        "CAGR": (1 + r).prod() ** (ppy / len(r)) - 1,
        "Volatility (ann.)": r.std(ddof=1) * np.sqrt(ppy),
        "Sharpe": ex.mean() / r.std(ddof=1) * np.sqrt(ppy),
        "Sortino": ex.mean() / downside * np.sqrt(ppy) if downside > 0 else np.nan,
        "Max drawdown": (wealth / wealth.cummax() - 1).min(),
        "CVaR 5% (monthly)": r[r <= r.quantile(0.05)].mean(),
        "Effective N (avg)": np.mean(1 / (W ** 2).sum(axis=1)),
        "Max weight (avg)": W.max(axis=1).mean(),
        "Turnover (avg)": np.mean(res["turnover"][1:]),
        "% weights at a bound": at_bound.mean(),
        "Avg delta (Bayes-Stein)": np.nanmean(res["delta"]) if np.any(~np.isnan(res["delta"])) else np.nan,
        "Max-Sharpe fallbacks": res["fallbacks"],
    }


# =====================================================================
# 7. DATA
# =====================================================================
def load_monthly_returns(path):
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    missing = [a for a in ASSETS if a not in px.columns]
    if missing:
        raise ValueError(f"Columns missing from {path}: {missing}")
    px = px[ASSETS]
    try:
        px = px.resample("ME").last()
    except ValueError:                                   # older pandas versions
        px = px.resample("M").last()
    return px.pct_change().dropna(how="any")


def synthetic_returns(months=240, seed=1):
    """Stationary demo data with rough asset-class characteristics (annual figures)."""
    mu = np.array([0.07, 0.08, 0.07, 0.085, 0.03, 0.038, 0.032, 0.045])
    vol = np.array([0.15, 0.16, 0.17, 0.22, 0.07, 0.07, 0.09, 0.15])
    kind = ["E", "E", "E", "E", "B", "B", "B", "G"]
    corr = np.eye(8)
    for i in range(8):
        for j in range(8):
            if i != j:
                pair = {kind[i], kind[j]}
                corr[i, j] = (0.75 if pair == {"E"} else 0.6 if pair == {"B"}
                              else 0.15 if pair == {"E", "B"} else 0.05 if pair == {"E", "G"} else 0.2)
    data = np.random.default_rng(seed).multivariate_normal(mu / 12, np.outer(vol, vol) * corr / 12, months)
    idx = pd.period_range("2006-01", periods=months, freq="M").to_timestamp(how="end").normalize()
    return pd.DataFrame(data, index=idx, columns=ASSETS)


# =====================================================================
# 8. VALIDATION TESTS
# =====================================================================
def self_test():
    print("Validation tests")
    vol = np.array([0.05, 0.04, 0.06, 0.03, 0.045])
    corr = np.full((5, 5), 0.4); np.fill_diagonal(corr, 1)
    S = np.outer(vol, vol) * corr
    mu = np.array([0.010, 0.006, 0.008, 0.004, 0.007])
    one = np.ones(5)

    w_cf = np.linalg.solve(S, one); w_cf /= w_cf.sum()                  # closed-form GMV
    print(f"  GMV (shorting allowed) vs closed form, max diff:        {np.abs(solve_gmv(S, -10, 10) - w_cf).max():.1e}")
    w_cf = np.linalg.solve(S, mu); w_cf /= w_cf.sum()                   # closed-form tangency
    print(f"  Max-Sharpe (shorting allowed) vs closed form, max diff: {np.abs(solve_max_sharpe(mu, S, 0, -10, 10)[0] - w_cf).max():.1e}")
    w = solve_max_sharpe(mu, S, 0, 0.0, 0.35)[0]
    print(f"  No-short + 35% cap respected: min w = {w.min():.3f}, max w = {w.max():.3f}, sum = {w.sum():.3f}")
    print(f"  Bayes-Stein delta on spreadsheet example (expect 0.8655): {bayes_stein(mu, S, 60)[1]:.4f}\n")


# =====================================================================
# 9. RUN
# =====================================================================
def _finish(fig):
    """Display the chart if SHOW_PLOTS is on (or in Jupyter), then close it."""
    if SHOW_PLOTS or "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)


def plot_results(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[2, 1])
    for name, res in results.items():
        wealth = (1 + res["ret"]).cumprod()
        a1.plot(wealth.index, wealth, label=name)
        a2.plot(wealth.index, wealth / wealth.cummax() - 1, label=name)
    a1.set_ylabel("Growth of 1 (net of costs)"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    a1.set_title("Out-of-sample walk-forward performance")
    a2.set_ylabel("Drawdown"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/performance.png", dpi=150); _finish(fig)

    opt = ["GMV", "Max Sharpe (sample)", "Max Sharpe (Bayes-Stein)"]
    fig, axes = plt.subplots(len(opt), 1, figsize=(10, 9), sharex=True)
    for ax, name in zip(axes, opt):
        W = results[name]["w"]
        if SHORT_SELLING:
            for col in W.columns:
                ax.plot(W.index, W[col], label=col)
            ax.axhline(0, color="k", lw=0.8)
        else:
            ax.stackplot(W.index, W.T.values, labels=W.columns); ax.set_ylim(0, 1)
        ax.set_title(name, fontsize=10); ax.set_ylabel("Weight")
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/weights.png", dpi=150); _finish(fig)


def plot_efficient_frontier(returns):
    """In-sample efficient frontier from the most recent LOOKBACK months of data."""
    ppy, rf = PERIODS_PER_YEAR, RF_ANNUAL
    lower, upper = weight_bounds()
    window = returns.values[-LOOKBACK:]
    mu_p, S_p = sample_moments(window)
    mu, S = mu_p * ppy, S_p * ppy                                     # annualised
    mu_bs = bayes_stein(mu_p, S_p, LOOKBACK)[0] * ppy
    vol = lambda w: np.sqrt(w @ S @ w)

    v_c, r_c = efficient_frontier(mu, S, lower, upper)                # constrained, sample means
    v_bs, r_bs = efficient_frontier(mu_bs, S, lower, upper)           # constrained, Bayes-Stein means
    targets = np.linspace(r_c.min() - 0.01, max(r_c.max(), mu.max()) + 0.01, 200)
    v_u = unconstrained_frontier(mu, S, targets)

    n = len(mu)
    ports = {
        "Equal weight": np.full(n, 1 / n),
        "Judgement": np.array([MANUAL_WEIGHTS[a] for a in returns.columns]),
        "GMV": solve_gmv(S, lower, upper),
        "Max Sharpe (sample)": solve_max_sharpe(mu, S, rf, lower, upper)[0],
    }
    w_bs = solve_max_sharpe(mu_bs, S, rf, lower, upper)[0]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.plot(v_u, targets, "--", color="grey", lw=1.2, label="Unconstrained frontier (shorting allowed, no cap)")
    ax.plot(v_c, r_c, color="#1F4E79", lw=2.5,
            label=f"Constrained frontier, sample means ({lower:.0%} to {upper:.0%} per asset)")
    ax.plot(v_bs, r_bs, ":", color="#C55A11", lw=2.2, label="Constrained frontier, Bayes-Stein means")
    ax.scatter(np.sqrt(np.diag(S)), mu, color="black", s=18, zorder=3)
    for name, x, y in zip(returns.columns, np.sqrt(np.diag(S)), mu):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)
    markers = ["s", "D", "o", "*"]
    for (name, w), m in zip(ports.items(), markers):
        ax.scatter(vol(w), w @ mu, marker=m, s=140 if m == "*" else 70, zorder=4, label=name)
    ax.scatter(vol(w_bs), w_bs @ mu_bs, marker="P", s=110, zorder=4, color="#C55A11",
               label="Max Sharpe (Bayes-Stein), at its shrunk return")
    ax.set_xlim(0, np.sqrt(np.diag(S)).max() * 1.15)
    ax.set_xlabel("Annualised volatility (standard deviation)")
    ax.set_ylabel("Annualised expected return (estimated)")
    ax.set_title(f"Markowitz efficient frontier (in-sample, estimated from the last {LOOKBACK} months)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout(); os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(f"{OUT_DIR}/efficient_frontier.png", dpi=150); _finish(fig)


def main():
    self_test()
    if len(sys.argv) > 1:
        returns = load_monthly_returns(sys.argv[1])
        print(f"Loaded {len(returns)} monthly returns, {returns.index[0]:%b %Y} - {returns.index[-1]:%b %Y}")
    else:
        returns = synthetic_returns()
        print("No CSV given: running on SYNTHETIC data (demo only)")
    if len(returns) <= LOOKBACK + 12:
        raise ValueError("Not enough data: need well over LOOKBACK months of returns")
    lower, upper = weight_bounds()
    print(f"Constraints: weights between {lower:.0%} and {upper:.0%}, fully invested | "
          f"T = {LOOKBACK}, n = {len(ASSETS)}, out-of-sample months = {len(returns) - LOOKBACK}\n")

    results = backtest(returns, make_strategies(ASSETS))
    table = pd.DataFrame({k: metrics(v) for k, v in results.items()})
    pct = ["CAGR", "Volatility (ann.)", "Max drawdown", "CVaR 5% (monthly)",
           "Max weight (avg)", "Turnover (avg)", "% weights at a bound"]
    shown = table.astype(object).copy()
    for row in table.index:
        shown.loc[row] = [f"{x:.1%}" if row in pct else (f"{x:.2f}" if pd.notna(x) else "-")
                          for x in table.loc[row]]
    pd.set_option("display.width", 200)
    print(shown.to_string())

    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(f"{OUT_DIR}/metrics.csv")
    for name, res in results.items():
        res["w"].to_csv(f"{OUT_DIR}/weights_{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv")
    plot_results(results)
    plot_efficient_frontier(returns)
    print(f"\nSaved metrics, weight histories and charts (incl. efficient_frontier.png) to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
