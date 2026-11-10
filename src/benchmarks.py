
from pathlib import Path
import numpy as np
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "results" / "benchmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["VTI", "EFA", "AGG", "GLD", "VNQ"]
START_DATE = "2015-01-01"
END_DATE = "2026-10-30"


def load_returns():
    """Load and align the portfolio and SPY return datasets."""
    assets = pd.read_csv(
        DATA_DIR / "asset_returns_daily.csv",
        index_col="Date",
        parse_dates=True,
    )
    benchmark = pd.read_csv(
        DATA_DIR / "benchmark_returns_daily.csv",
        index_col="Date",
        parse_dates=True,
    )

    assets = assets[ASSETS].sort_index()
    spy = benchmark[["SPY"]].sort_index()

    # Require complete common observations, without filling gaps.
    combined = assets.join(spy, how="inner").dropna()

    if combined.empty:
        raise ValueError("No aligned return observations found.")

    if not np.isfinite(combined.to_numpy()).all():
        raise ValueError("Non-finite returns detected.")

    if (combined <= -1).any().any():
        raise ValueError("Invalid return of -100% or below.")

    return combined.loc[START_DATE:END_DATE]


def equal_weight_portfolio(returns):
    """
    Monthly rebalanced 1/N benchmark.

    Weights are reset before the first observed trading-day
    return of each month and drift between rebalances.
    """
    n_assets = returns.shape[1]
    target = np.full(n_assets, 1.0 / n_assets)

    weights = target.copy()
    portfolio_returns = []
    weights_history = []

    current_month = None

    for date, row in returns.iterrows():
        month = date.to_period("M")

        if month != current_month:
            weights = target.copy()
            current_month = month

        # Beginning-of-day holdings: no same-day return used
        weights_history.append(weights.copy())

        daily = row.to_numpy(dtype=float)
        portfolio_return = float(weights @ daily)
        portfolio_returns.append(portfolio_return)

        # Let weights drift after daily asset returns
        weights = weights * (1.0 + daily)
        weights = weights / (1.0 + portfolio_return)

    portfolio = pd.Series(
        portfolio_returns,
        index=returns.index,
        name="Equal_Weight",
    )
    weights_df = pd.DataFrame(
        weights_history,
        index=returns.index,
        columns=returns.columns,
    )

    return portfolio, weights_df


def calculate_metrics(returns):
    """Basic annualised performance metrics."""
    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1

    n_days = len(returns)

    annual_return = cumulative.iloc[-1] ** (252 / n_days) - 1
    volatility = returns.std(ddof=1) * np.sqrt(252)

    return {
        "Annualised Return": annual_return,
        "Annualised Volatility": volatility,
        "Maximum Drawdown": drawdown.min(),
        "Final Growth of $1": cumulative.iloc[-1],
    }


def main():
    data = load_returns()

    equal_returns, weights = equal_weight_portfolio(
        data[ASSETS]
    )

    # Broad market benchmark: SPY buy-and-hold
    spy_returns = data["SPY"].rename("SPY")

    comparison = pd.concat(
        [equal_returns, spy_returns],
        axis=1,
    )

    wealth = (1 + comparison).cumprod()

    summary = pd.DataFrame({
        col: calculate_metrics(comparison[col])
        for col in comparison.columns
    }).T

    # Save research outputs
    comparison.to_csv(
        OUTPUT_DIR / "benchmark_returns.csv"
    )
    wealth.to_csv(
        OUTPUT_DIR / "benchmark_growth.csv"
    )
    weights.to_csv(
        OUTPUT_DIR / "equal_weight_allocations.csv"
    )
    summary.to_csv(
        OUTPUT_DIR / "benchmark_metrics.csv"
    )

    print("Benchmark backtest completed.")
    print("\nTesting period:")
    print(data.index.min(), "to", data.index.max())
    print("\nObservations:", len(data))

    print("\nPerformance summary:")
    print(summary.round(4))

    print("\nFiles saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
