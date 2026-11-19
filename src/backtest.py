
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "asset_returns_daily.csv"

ASSETS = ["VTI", "EFA", "AGG", "GLD", "VNQ"]

LOOKBACK = 252
START_DATE = "2015-01-01"
END_DATE = "2026-10-30"


# --------------------------------------------------
# 2. Load and validate historical returns
# --------------------------------------------------

def load_returns():
    """
    Load historical ETF returns, including pre-2015
    observations required for the estimation windows.
    """

    returns = pd.read_csv(
        DATA_FILE,
        index_col="Date",
        parse_dates=True
    )

    returns = returns[ASSETS].sort_index()

    if returns.index.has_duplicates:
        raise ValueError("Duplicate dates detected.")

    if returns.isna().any().any():
        raise ValueError("Missing returns detected.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Invalid numerical returns detected.")

    if (returns <= -1).any().any():
        raise ValueError("Invalid daily returns detected.")

    return returns.astype(float)


# --------------------------------------------------
# 3. Generate rolling estimation windows
# --------------------------------------------------

def get_rebalance_windows(returns):
    """
    Generate monthly rebalance dates and the previous
    252 trading days of returns for each date.

    The rebalance day's return is excluded to prevent
    look-ahead bias.
    """

    test_returns = returns.loc[START_DATE:END_DATE]

    rebalance_dates = (
        test_returns
        .groupby(test_returns.index.to_period("M"))
        .head(1)
        .index
    )

    windows = {}

    for date in rebalance_dates:

        position = returns.index.get_loc(date)

        if position < LOOKBACK:
            raise ValueError(
                f"Insufficient historical data before {date}"
            )

        window = returns.iloc[
            position - LOOKBACK:position
        ].copy()

        assert len(window) == LOOKBACK
        assert window.index.max() < date

        windows[date] = window

    return windows


# --------------------------------------------------
# 4. Shared walk-forward backtesting engine
# --------------------------------------------------

def run_backtest(returns, allocation_function):
    """
    Run a monthly-rebalanced portfolio backtest.

    At each rebalance:
        1. Provide the model with the previous
           252 trading days of returns.
        2. Calculate target portfolio weights.
        3. Apply weights before the day's return.
        4. Allow weights to drift between rebalances.

    The allocation function can later be replaced
    with MVO or HRP.
    """

    windows = get_rebalance_windows(returns)

    test_returns = returns.loc[START_DATE:END_DATE]

    current_weights = None

    portfolio_returns = []
    weights_history = []

    for date, daily_returns in test_returns.iterrows():

        # Rebalance on the first trading day of each month
        if date in windows:

            target_weights = allocation_function(
                windows[date]
            )

            target_weights = pd.Series(
                target_weights,
                dtype=float
            ).reindex(ASSETS)

            # Validate model output
            if not np.isfinite(
                target_weights.to_numpy()
            ).all():
                raise ValueError(
                    f"Invalid portfolio weights on {date}"
                )

            if (target_weights < -1e-10).any():
                raise ValueError(
                    f"Negative portfolio weights on {date}"
                )

            if not np.isclose(
                target_weights.sum(), 1.0
            ):
                raise ValueError(
                    f"Weights do not sum to 1 on {date}"
                )

            current_weights = target_weights.copy()

        if current_weights is None:
            raise ValueError(
                "No initial portfolio allocation."
            )

        # Save weights before daily asset returns
        weights_history.append(
            current_weights.copy()
        )

        # Calculate daily weighted portfolio return
        daily_return = float(
            current_weights @ daily_returns
        )

        portfolio_returns.append(daily_return)

        # Update weights following market movements
        # (without rebalancing until next month)
        current_weights = (
            current_weights * (1 + daily_returns)
        ) / (1 + daily_return)

    portfolio = pd.Series(
        portfolio_returns,
        index=test_returns.index,
        name="Portfolio"
    )

    weights = pd.DataFrame(
        weights_history,
        index=test_returns.index,
        columns=ASSETS
    )

    return portfolio, weights


# --------------------------------------------------
# 5. Equal-weight allocation model
# --------------------------------------------------

def equal_weight_allocator(returns_window):
    """
    Allocate 20% to each ETF.

    Used to verify the shared backtesting engine
    against our existing 1/N benchmark.
    """

    n_assets = returns_window.shape[1]

    weights = pd.Series(
        np.full(n_assets, 1.0 / n_assets),
        index=returns_window.columns
    )

    return weights


# --------------------------------------------------
# 6. Main execution and validation
# --------------------------------------------------

def main():

    returns = load_returns()

    windows = get_rebalance_windows(returns)

    print("Walk-forward estimation windows created.")
    print("Monthly rebalances:", len(windows))

    # Run equal-weight model through the shared engine
    portfolio, weights = run_backtest(
        returns,
        equal_weight_allocator
    )

    print("\nWalk-forward backtest completed.")

    print("\nTesting period:")
    print(
        portfolio.index.min().date(),
        "to",
        portfolio.index.max().date()
    )

    print("\nDaily observations:", len(portfolio))

    print("\nFirst five portfolio returns:")
    print(portfolio.head().round(6))

    print("\nFirst five portfolio allocations:")
    print(weights.head().round(6))

    # Calculate cumulative portfolio growth
    growth = (1 + portfolio).cumprod()

    print("\nFinal growth of $1:")
    print(round(growth.iloc[-1], 4))

    # --------------------------------------------------
    # Compare with previously validated 1/N benchmark
    # --------------------------------------------------

    benchmark_file = (
        ROOT
        / "results"
        / "benchmarks"
        / "benchmark_returns.csv"
    )

    benchmark = pd.read_csv(
        benchmark_file,
        index_col="Date",
        parse_dates=True
    )["Equal_Weight"]

    benchmark = benchmark.reindex(portfolio.index)

    if benchmark.isna().any():
        raise ValueError(
            "Missing benchmark returns for comparison."
        )

    differences = np.abs(
        portfolio.to_numpy()
        - benchmark.to_numpy()
    )

    max_difference = differences.max()

    print("\nMaximum difference from existing benchmark:")
    print(max_difference)

    assert np.allclose(
        portfolio.to_numpy(),
        benchmark.to_numpy(),
        rtol=0,
        atol=1e-12
    ), "Backtest does not match the existing benchmark."

    print(
        "\nPASS: Backtester matches existing 1/N benchmark."
    )

    # Confirm portfolio weights remain fully invested
    assert np.allclose(
        weights.sum(axis=1),
        1.0
    )

    print("PASS: Portfolio weights sum to 100%.")

    print("\nPhase 2 backtesting engine: initial checks passed.")


if __name__ == "__main__":
    main()
