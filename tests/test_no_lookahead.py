
"""
LQDF Portfolio Construction
Phase 2 Validation - Look-Ahead Bias

Verify that MVO and HRP receive only historical
data available before each rebalance date.

Also check that changing current/future returns
does not affect the previously calculated weights.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import (
    load_returns,
    get_rebalance_windows,
    run_backtest,
    LOOKBACK,
)

from mvo import calculate_weights as mvo_allocator

from hrp import (
    calculate_clustering,
    calculate_hrp_weights,
)


def hrp_allocator(window):
    """Adapt Josh's HRP functions to the common interface."""

    _, _, ordered_assets = calculate_clustering(window)

    weights = calculate_hrp_weights(
        window, ordered_assets
    )

    return weights.reindex(window.columns)


def test_windows(returns):
    """Verify every estimation window uses past data only."""

    windows = get_rebalance_windows(returns)

    assert len(windows) == 142

    for date, window in windows.items():
        assert len(window) == LOOKBACK
        assert window.index.max() < date
        assert window.index.is_monotonic_increasing
        assert window.index.is_unique

        position = returns.index.get_loc(date)

        expected = returns.iloc[
            position - LOOKBACK:position
        ]

        pd.testing.assert_frame_equal(window, expected)

    print("PASS: All 142 windows use historical data only.")

    return windows


def test_model(name, allocator, returns, windows):
    """Verify timing across the full model backtest."""

    rebalance_dates = list(windows)
    call_dates = []
    call_windows = []

    def tracked_allocator(window):

        # The backtester should pass one of
        # the exact historical windows.
        call_position = len(call_dates)

        assert call_position < len(rebalance_dates)

        date = rebalance_dates[call_position]

        assert len(window) == LOOKBACK
        assert window.index.max() < date

        call_dates.append(date)
        call_windows.append(window.copy())

        return allocator(window)

    portfolio, weights = run_backtest(
        returns, tracked_allocator
    )

    assert call_dates == rebalance_dates
    assert len(call_windows) == 142
    assert len(portfolio) == 2975
    assert portfolio.index.equals(weights.index)

    print(
        f"PASS: {name} uses historical data "
        "at all 142 rebalances."
    )

    # Test one rebalance in the middle of the period
    test_date = rebalance_dates[len(rebalance_dates) // 2]

    original_window = windows[test_date]

    original_weights = allocator(original_window)

    # Change the rebalance day's return and all
    # subsequent returns. The historical window
    # must remain completely unchanged.
    modified_returns = returns.copy()

    modified_returns.loc[
        test_date:, "VTI"
    ] = modified_returns.loc[
        test_date:, "VTI"
    ] + 0.05

    modified_windows = get_rebalance_windows(
        modified_returns
    )

    modified_window = modified_windows[test_date]

    pd.testing.assert_frame_equal(
        original_window,
        modified_window
    )

    modified_weights = allocator(modified_window)

    assert np.allclose(
        np.asarray(original_weights, dtype=float),
        np.asarray(modified_weights, dtype=float),
        rtol=0,
        atol=1e-10
    )

    print(
        f"PASS: {name} weights on "
        f"{test_date.date()} are unaffected "
        "by current/future return changes."
    )


def main():

    print("LQDF - Look-Ahead Bias Validation")
    print("--------------------------------")

    returns = load_returns()

    windows = test_windows(returns)

    models = {
        "MVO": mvo_allocator,
        "HRP": hrp_allocator,
    }

    for name, allocator in models.items():

        print(f"\nTesting {name}...")

        test_model(
            name,
            allocator,
            returns,
            windows
        )

    print(
        "\nPASS: All look-ahead timing checks passed."
    )


if __name__ == "__main__":
    main()
