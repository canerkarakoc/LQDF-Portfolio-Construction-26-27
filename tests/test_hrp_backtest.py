
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Make src modules importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import (
    load_returns,
    get_rebalance_windows,
    run_backtest
)

from hrp import (
    calculate_clustering,
    calculate_hrp_weights
)


def hrp_allocator(returns_window):
    """
    Calculate HRP allocations using only the
    historical returns supplied by the backtester.
    """

    correlation, linkage_matrix, ordered_assets = (
        calculate_clustering(returns_window)
    )

    weights = calculate_hrp_weights(
        returns_window,
        ordered_assets
    )

    return weights.reindex(returns_window.columns)


def main():

    returns = load_returns()
    windows = get_rebalance_windows(returns)

    # Test the first historical estimation window
    first_date = next(iter(windows))
    first_window = windows[first_date]

    first_weights = hrp_allocator(first_window)

    print("First HRP rebalance:", first_date.date())

    print("\nInitial HRP allocations:")
    print((first_weights * 100).round(2))

    assert np.isfinite(first_weights.to_numpy()).all()
    assert (first_weights >= 0).all()
    assert np.isclose(first_weights.sum(), 1.0)

    print("\nPASS: Initial HRP weights are valid.")

    # Run HRP through the shared walk-forward engine
    portfolio, weights = run_backtest(
        returns,
        hrp_allocator
    )

    assert len(portfolio) == 2975
    assert np.isfinite(portfolio.to_numpy()).all()
    assert np.allclose(weights.sum(axis=1), 1.0)

    print("\nHRP walk-forward backtest completed.")
    print("Daily observations:", len(portfolio))
    print("Monthly estimation windows:", len(windows))

    growth = (1 + portfolio).cumprod()

    print("\nFinal growth of $1:")
    print(round(growth.iloc[-1], 4))

    # Annualised performance
    annual_return = growth.iloc[-1] ** (
        252 / len(portfolio)
    ) - 1

    annual_volatility = (
        portfolio.std(ddof=1) * np.sqrt(252)
    )

    print("\nAnnualised return:",
          f"{annual_return:.2%}")

    print("Annualised volatility:",
          f"{annual_volatility:.2%}")

    print("\nPASS: HRP integrates with shared backtester.")


if __name__ == "__main__":
    main()
