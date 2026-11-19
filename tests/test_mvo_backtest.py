
from pathlib import Path
import sys

import numpy as np

# Make src modules importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import (
    load_returns,
    get_rebalance_windows,
    run_backtest
)

from mvo import calculate_weights


def main():

    returns = load_returns()
    windows = get_rebalance_windows(returns)

    # ---------------------------------------------
    # 1. Test first 252-day estimation window
    # ---------------------------------------------

    first_date = next(iter(windows))
    first_window = windows[first_date]

    first_weights = calculate_weights(first_window)

    print("First MVO rebalance:", first_date.date())

    print("\nInitial MVO allocations:")
    print((first_weights * 100).round(2))

    assert np.isfinite(first_weights.to_numpy()).all()
    assert (first_weights >= -1e-8).all()
    assert np.isclose(first_weights.sum(), 1.0)

    print("\nPASS: Initial MVO weights are valid.")

    # ---------------------------------------------
    # 2. Check reproducibility
    # ---------------------------------------------

    repeated_weights = calculate_weights(first_window)

    assert np.allclose(
        first_weights.to_numpy(),
        repeated_weights.to_numpy()
    )

    print("PASS: MVO results are reproducible.")

    # ---------------------------------------------
    # 3. Run shared walk-forward backtest
    # ---------------------------------------------

    portfolio, weights = run_backtest(
        returns,
        calculate_weights
    )

    assert len(portfolio) == 2975
    assert np.isfinite(portfolio.to_numpy()).all()
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= -1e-8).all().all()

    print("\nMVO walk-forward backtest completed.")
    print("Daily observations:", len(portfolio))
    print("Monthly estimation windows:", len(windows))

    # ---------------------------------------------
    # 4. Calculate initial performance statistics
    # ---------------------------------------------

    growth = (1 + portfolio).cumprod()

    annual_return = (
        growth.iloc[-1] ** (252 / len(portfolio)) - 1
    )

    annual_volatility = (
        portfolio.std(ddof=1) * np.sqrt(252)
    )

    peak = np.maximum.accumulate(
        np.r_[1.0, growth.to_numpy()]
    )

    max_drawdown = (
        np.r_[1.0, growth.to_numpy()] / peak - 1.0
    ).min()

    print("\nFinal growth of $1:")
    print(round(growth.iloc[-1], 4))

    print("\nAnnualised return:",
          f"{annual_return:.2%}")

    print("Annualised volatility:",
          f"{annual_volatility:.2%}")

    print("Maximum drawdown:",
          f"{max_drawdown:.2%}")

    # ---------------------------------------------
    # 5. Inspect portfolio concentration
    # ---------------------------------------------

    print("\nLargest initial asset allocation:")
    print(first_weights.idxmax(),
          f"{first_weights.max():.2%}")

    print("\nPASS: MVO integrates with shared backtester.")


if __name__ == "__main__":
    main()
