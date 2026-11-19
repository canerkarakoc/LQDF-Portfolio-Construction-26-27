
"""
LQDF Portfolio Construction
Phase 2 - HRP Stability & Reproducibility

Validates:
1. HRP weight constraints across all monthly windows.
2. Reproducibility of allocations and clustering.
3. Monthly allocation changes.
4. Portfolio concentration.
5. Unexpected or extreme allocations.

Large changes are reported for research review,
not automatically treated as implementation errors.
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
    ASSETS,
)

from hrp import (
    calculate_clustering,
    calculate_hrp_weights,
)


OUTPUT_DIR = ROOT / "results" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def hrp_allocator(window):
    """Calculate HRP allocations and clustering outputs."""

    correlation, linkage_matrix, ordered_assets = (
        calculate_clustering(window)
    )

    weights = calculate_hrp_weights(
        window,
        ordered_assets
    ).reindex(window.columns)

    return weights, linkage_matrix, ordered_assets


def main():

    print("LQDF - HRP Stability Validation")
    print("------------------------------")

    returns = load_returns()
    windows = get_rebalance_windows(returns)

    allocations = {}
    cluster_orders = {}
    largest_weights = {}

    for date, window in windows.items():

        # First calculation
        weights, linkage_matrix, order = (
            hrp_allocator(window)
        )

        # Repeat using a fresh copy of the input
        repeated_weights, repeated_linkage, repeated_order = (
            hrp_allocator(window.copy(deep=True))
        )

        # ------------------------------------
        # Test 1: Weight validity
        # ------------------------------------

        w = weights.to_numpy(dtype=float)

        assert np.isfinite(w).all()
        assert np.all(w >= -1e-10)
        assert np.all(w <= 1 + 1e-10)
        assert np.isclose(w.sum(), 1.0, atol=1e-10)

        assert list(weights.index) == ASSETS

        # ------------------------------------
        # Test 2: Reproducibility
        # ------------------------------------

        pd.testing.assert_series_equal(
            weights,
            repeated_weights
        )

        assert np.allclose(
            linkage_matrix,
            repeated_linkage,
            rtol=0,
            atol=1e-12
        )

        assert order == repeated_order

        # ------------------------------------
        # Save observations
        # ------------------------------------

        allocations[date] = weights
        cluster_orders[date] = ", ".join(order)
        largest_weights[date] = float(weights.max())

    # ----------------------------------------
    # Convert results to monthly table
    # ----------------------------------------

    weights_df = pd.DataFrame.from_dict(
        allocations,
        orient="index"
    )[ASSETS].sort_index()

    weights_df.index.name = "Date"

    # ----------------------------------------
    # Test 3: Monthly target-weight changes
    # ----------------------------------------

    # These are target-to-target differences,
    # NOT trading turnover.
    monthly_changes = (
        weights_df.diff().abs().sum(axis=1) / 2
    ).iloc[1:]

    # ----------------------------------------
    # Test 4: Portfolio concentration
    # ----------------------------------------

    hhi = weights_df.pow(2).sum(axis=1)
    max_weights = weights_df.max(axis=1)

    dominant_asset = weights_df.idxmax(axis=1)

    # ----------------------------------------
    # Test 5: Clustering changes
    # ----------------------------------------

    # Leaf-order changes are only a diagnostic:
    # different leaf orders can represent the
    # same hierarchical clustering tree.
    orders = pd.Series(cluster_orders).sort_index()

    order_changed = (
        orders != orders.shift(1)
    ).iloc[1:]

    # ----------------------------------------
    # Print results
    # ----------------------------------------

    print("\nEstimation windows tested:", len(windows))

    assert len(windows) == 142

    print("\nPASS: All HRP weight constraints satisfied.")
    print("PASS: All 142 HRP runs are reproducible.")

    print("\nAllocation stability:")
    print(
        "Average monthly target-weight change:",
        f"{monthly_changes.mean():.2%}"
    )
    print(
        "Largest monthly target-weight change:",
        f"{monthly_changes.max():.2%}"
    )
    print(
        "Date of largest change:",
        monthly_changes.idxmax().date()
    )

    print("\nPortfolio concentration:")
    print(
        "Average maximum asset allocation:",
        f"{max_weights.mean():.2%}"
    )
    print(
        "Highest individual asset allocation:",
        f"{max_weights.max():.2%}"
    )
    print(
        "Average HHI:",
        f"{hhi.mean():.4f}"
    )

    print("\nMost frequently dominant asset:")
    print(dominant_asset.value_counts().to_string())

    print("\nClustering:")
    print(
        "Months with a change in leaf ordering:",
        int(order_changed.sum())
    )

    print("\nLargest five allocation changes:")
    print(
        monthly_changes.nlargest(5).to_string()
    )

    # ----------------------------------------
    # Save research outputs
    # ----------------------------------------

    weights_df.to_csv(
        OUTPUT_DIR / "hrp_monthly_target_weights.csv"
    )

    summary = pd.DataFrame({
        "Largest_Weight": max_weights,
        "HHI": hhi,
        "Dominant_Asset": dominant_asset,
        "Cluster_Leaf_Order": orders,
    })

    summary.index.name = "Date"

    summary.to_csv(
        OUTPUT_DIR / "hrp_stability_summary.csv"
    )

    monthly_changes.rename(
        "Target_Weight_Change"
    ).to_csv(
        OUTPUT_DIR / "hrp_monthly_weight_changes.csv"
    )

    print("\nResults saved to:")
    print(OUTPUT_DIR)

    print(
        "\nPASS: HRP reproducibility checks completed."
    )
    print(
        "NOTE: Allocation stability requires interpretation."
    )


if __name__ == "__main__":
    main()
