
"""
LQDF Portfolio Construction
Phase 2 - Unified Model Comparison

Compare:
    1. MVO (Global Minimum Variance)
    2. HRP (Hierarchical Risk Parity)
    3. Equal Weight (1/N)
    4. SPY (S&P 500 benchmark)

All strategies use the same out-of-sample dates.

Sharpe ratios currently assume a zero risk-free rate.
Transaction costs are not yet deducted.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest import (
    ROOT,
    ASSETS,
    load_returns,
    get_rebalance_windows,
    run_backtest,
    equal_weight_allocator,
)

from mvo import calculate_weights as mvo_allocator

from hrp import (
    calculate_clustering,
    calculate_hrp_weights,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

OUTPUT_DIR = ROOT / "results" / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.0


# --------------------------------------------------
# 2. HRP adapter
# --------------------------------------------------

def hrp_allocator(returns_window):
    """Use Josh's HRP functions with the shared backtester."""

    correlation, linkage_matrix, ordered_assets = (
        calculate_clustering(returns_window)
    )

    weights = calculate_hrp_weights(
        returns_window,
        ordered_assets
    )

    return weights.reindex(returns_window.columns)


# --------------------------------------------------
# 3. Performance metrics
# --------------------------------------------------

def calculate_metrics(returns):
    """Calculate standardised performance metrics."""

    returns = returns.astype(float)

    if returns.empty or not np.isfinite(returns).all():
        raise ValueError("Invalid performance series.")

    growth = (1.0 + returns).cumprod()

    final_growth = float(growth.iloc[-1])

    annual_return = (
        final_growth ** (TRADING_DAYS / len(returns))
        - 1.0
    )

    annual_volatility = (
        returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    # Preliminary Sharpe with annual risk-free rate = 0
    sharpe = (
        (returns.mean() * TRADING_DAYS - RISK_FREE_RATE)
        / annual_volatility
        if annual_volatility > 0
        else np.nan
    )

    # Include initial wealth of 1 in drawdown calculation
    wealth = np.r_[1.0, growth.to_numpy()]
    running_peak = np.maximum.accumulate(wealth)

    max_drawdown = float(
        (wealth / running_peak - 1.0).min()
    )

    return {
        "Annualised Return": annual_return,
        "Annualised Volatility": annual_volatility,
        "Sharpe (RF=0)": sharpe,
        "Maximum Drawdown": max_drawdown,
        "Final Growth of $1": final_growth,
    }


# --------------------------------------------------
# 4. Calculate one-way portfolio turnover
# --------------------------------------------------

def calculate_turnover(weights, asset_returns):
    """
    Calculate one-way turnover at each monthly rebalance.

    Compare new target weights against the holdings
    immediately before rebalancing, after the previous
    day's market movements.

    The initial portfolio purchase is excluded.
    """

    rebalance_dates = (
        weights
        .groupby(weights.index.to_period("M"))
        .head(1)
        .index
    )

    turnover = {}

    for i, date in enumerate(rebalance_dates):

        if i == 0:
            continue

        location = weights.index.get_loc(date)
        previous_date = weights.index[location - 1]

        previous_weights = weights.loc[
            previous_date
        ].to_numpy(dtype=float)

        previous_returns = asset_returns.loc[
            previous_date, ASSETS
        ].to_numpy(dtype=float)

        # Portfolio holdings after previous day's returns
        ending_values = (
            previous_weights * (1.0 + previous_returns)
        )

        pre_rebalance_weights = (
            ending_values / ending_values.sum()
        )

        new_weights = weights.loc[
            date, ASSETS
        ].to_numpy(dtype=float)

        # One-way turnover
        turnover[date] = (
            0.5
            * np.abs(
                new_weights - pre_rebalance_weights
            ).sum()
        )

    return pd.Series(
        turnover,
        name="One_Way_Turnover",
        dtype=float
    )


# --------------------------------------------------
# 5. Concentration statistics
# --------------------------------------------------

def calculate_concentration(weights):
    """
    Herfindahl-Hirschman Index (HHI):
        sum of squared portfolio weights.

    With five equal weights, HHI = 0.20.
    With one asset at 100%, HHI = 1.00.
    """

    hhi = weights.pow(2).sum(axis=1)

    max_weight = weights.max(axis=1)

    return {
        "Average HHI": float(hhi.mean()),
        "Average Largest Weight": float(
            max_weight.mean()
        ),
    }


# --------------------------------------------------
# 6. Main comparison
# --------------------------------------------------

def main():

    print("LQDF - Unified Portfolio Comparison")
    print("----------------------------------")

    returns = load_returns()
    windows = get_rebalance_windows(returns)

    print("\nMonthly estimation windows:", len(windows))

    models = {
        "MVO": mvo_allocator,
        "HRP": hrp_allocator,
        "Equal_Weight": equal_weight_allocator,
    }

    portfolios = {}
    allocations = {}
    metrics = {}
    turnover_history = {}

    # ----------------------------------------------
    # Run the three allocation strategies
    # ----------------------------------------------

    for name, allocator in models.items():

        print(f"\nRunning {name}...")

        portfolio, weights = run_backtest(
            returns,
            allocator
        )

        portfolio.name = name

        portfolios[name] = portfolio
        allocations[name] = weights

        model_metrics = calculate_metrics(portfolio)

        # Concentration
        model_metrics.update(
            calculate_concentration(weights)
        )

        # Turnover
        test_asset_returns = returns.loc[weights.index]

        turnover = calculate_turnover(
            weights,
            test_asset_returns
        )

        turnover_history[name] = turnover

        # Annualise turnover across the test period
        years = len(portfolio) / TRADING_DAYS

        model_metrics["Annualised Turnover"] = (
            turnover.sum() / years
        )

        metrics[name] = model_metrics

        # Save the model's daily allocations
        weights.to_csv(
            OUTPUT_DIR / f"{name.lower()}_weights.csv"
        )

        print(f"{name} completed successfully.")

    # ----------------------------------------------
    # Load SPY benchmark
    # ----------------------------------------------

    benchmark_file = (
        ROOT / "results" / "benchmarks"
        / "benchmark_returns.csv"
    )

    benchmark_data = pd.read_csv(
        benchmark_file,
        index_col="Date",
        parse_dates=True
    )

    spy = benchmark_data["SPY"].reindex(
        portfolios["Equal_Weight"].index
    )

    if spy.isna().any():
        raise ValueError(
            "SPY benchmark contains missing observations."
        )

    portfolios["SPY"] = spy.rename("SPY")

    metrics["SPY"] = calculate_metrics(spy)

    # Allocation concentration and rebalancing turnover
    # are not directly comparable for the single-index SPY.
    metrics["SPY"].update({
        "Average HHI": np.nan,
        "Average Largest Weight": np.nan,
        "Annualised Turnover": np.nan,
    })

    # ----------------------------------------------
    # Verify identical evaluation dates
    # ----------------------------------------------

    comparison = pd.DataFrame(portfolios)

    if comparison.isna().any().any():
        raise ValueError(
            "Models have different evaluation dates."
        )

    assert len(comparison) == 2975

    print("\nPASS: All four strategies share 2975 dates.")

    # Independently verify equal-weight benchmark
    original_equal_weight = benchmark_data[
        "Equal_Weight"
    ].reindex(comparison.index)

    assert np.allclose(
        comparison["Equal_Weight"].to_numpy(),
        original_equal_weight.to_numpy(),
        rtol=0,
        atol=1e-12
    )

    print("PASS: Equal-weight benchmark reproduced.")

    # ----------------------------------------------
    # Build performance tables
    # ----------------------------------------------

    metrics_df = pd.DataFrame.from_dict(
        metrics,
        orient="index"
    )

    growth_df = (1.0 + comparison).cumprod()

    turnover_df = pd.DataFrame(turnover_history)

    # ----------------------------------------------
    # Save research outputs
    # ----------------------------------------------

    comparison.to_csv(
        OUTPUT_DIR / "daily_returns.csv"
    )

    growth_df.to_csv(
        OUTPUT_DIR / "cumulative_growth.csv"
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "performance_metrics.csv"
    )

    turnover_df.to_csv(
        OUTPUT_DIR / "monthly_turnover.csv"
    )

    # ----------------------------------------------
    # Print summary
    # ----------------------------------------------

    print("\nEvaluation period:")
    print(
        comparison.index.min().date(),
        "to",
        comparison.index.max().date()
    )

    display = metrics_df.copy()

    percentage_columns = [
        "Annualised Return",
        "Annualised Volatility",
        "Maximum Drawdown",
        "Average Largest Weight",
        "Annualised Turnover",
    ]

    display[percentage_columns] *= 100

    print("\nPerformance metrics:")
    print(
        display[
            [
                "Annualised Return",
                "Annualised Volatility",
                "Sharpe (RF=0)",
                "Maximum Drawdown",
                "Final Growth of $1",
            ]
        ].round(3).to_string()
    )

    print("\nPortfolio concentration and turnover:")
    print(
        display[
            [
                "Average HHI",
                "Average Largest Weight",
                "Annualised Turnover",
            ]
        ].round(3).to_string()
    )

    print("\nResults saved to:")
    print(OUTPUT_DIR)

    print("\nPASS: Unified comparison completed.")


if __name__ == "__main__":
    main()
