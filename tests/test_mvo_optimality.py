
"""
LQDF Portfolio Construction
Phase 2 - MVO Optimality Validation

Tests Global Minimum Variance solutions across
all 142 monthly estimation windows.

For convex portfolio variance minimisation,
first-order optimality conditions provide
a certificate of global optimality.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import (
    load_returns,
    get_rebalance_windows,
)

from mvo import calculate_weights


# Numerical tolerances
WEIGHT_TOL = 1e-8
PSD_TOL = 1e-10
OPTIMALITY_TOL = 1e-6


def main():

    print("LQDF - MVO Optimality Validation")
    print("--------------------------------")

    returns = load_returns()
    windows = get_rebalance_windows(returns)

    tested = 0
    max_gap = 0.0
    max_weight_error = 0.0

    for date, window in windows.items():

        # Calculate minimum-variance weights
        weights = calculate_weights(window)
        w = weights.to_numpy(dtype=float)

        # Annualised covariance matrix
        covariance = (
            window.cov().to_numpy(dtype=float) * 252
        )

        # ----------------------------------
        # Test 1: Weight constraints
        # ----------------------------------

        assert np.isfinite(w).all()

        assert np.all(w >= -WEIGHT_TOL), (
            f"Negative weights on {date.date()}"
        )

        assert np.all(w <= 1 + WEIGHT_TOL), (
            f"Weights above 100% on {date.date()}"
        )

        weight_error = abs(w.sum() - 1.0)

        assert weight_error < WEIGHT_TOL, (
            f"Weight sum invalid on {date.date()}"
        )

        max_weight_error = max(
            max_weight_error,
            weight_error
        )

        # ----------------------------------
        # Test 2: Covariance properties
        # ----------------------------------

        assert np.allclose(
            covariance,
            covariance.T,
            atol=1e-12
        )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        assert eigenvalues.min() >= -PSD_TOL, (
            f"Covariance not PSD on {date.date()}"
        )

        # ----------------------------------
        # Test 3: First-order optimality
        # ----------------------------------

        # Objective: f(w) = w.T @ covariance @ w
        gradient = 2.0 * covariance @ w

        # For weights on the probability simplex,
        # the minimum gradient component gives
        # the best possible linearised direction.
        #
        # Convexity means the gap below bounds
        # how far the solution can be from the
        # globally optimal objective value.

        optimality_gap = float(
            w @ gradient - gradient.min()
        )

        optimality_gap = max(
            0.0,
            optimality_gap
        )

        assert optimality_gap < OPTIMALITY_TOL, (
            f"MVO optimality check failed on "
            f"{date.date()}: "
            f"gap = {optimality_gap:.10f}"
        )

        max_gap = max(
            max_gap,
            optimality_gap
        )

        tested += 1

    # --------------------------------------
    # Results
    # --------------------------------------

    print("\nEstimation windows tested:", tested)

    assert tested == 142

    print("\nMaximum weight-sum error:")
    print(f"{max_weight_error:.12e}")

    print("\nMaximum optimality gap:")
    print(f"{max_gap:.12e}")

    print(
        "\nPASS: All MVO portfolios satisfy "
        "long-only and full-investment constraints."
    )

    print(
        "PASS: All covariance matrices are "
        "positive semidefinite."
    )

    print(
        "PASS: All MVO solutions satisfy "
        "the specified numerical optimality tolerance."
    )

    print(
        "\nPASS: MVO optimality validation completed."
    )


if __name__ == "__main__":
    main()
