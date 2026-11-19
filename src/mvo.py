
"""
LQDF Portfolio Construction
Mean-Variance Optimisation (MVO)

Global Minimum Variance implementation adapted from
Flynn's MVO research prototype.

Designed for the shared walk-forward backtester.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_weights(returns_window):
    """
    Calculate long-only Global Minimum Variance weights.

    Input:
        DataFrame of 252 historical daily asset returns.

    Output:
        Series of optimal weights indexed by ETF ticker.

    Constraints:
        - All weights >= 0
        - All weights <= 1
        - Weights sum to 1
        - No short selling or leverage
    """

    if not isinstance(returns_window, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if len(returns_window) < 2:
        raise ValueError("Insufficient historical returns.")

    if returns_window.columns.has_duplicates:
        raise ValueError("Duplicate asset names detected.")

    if returns_window.isna().any().any():
        raise ValueError("Missing returns detected.")

    if not np.isfinite(
        returns_window.to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite returns detected.")

    assets = list(returns_window.columns)
    n_assets = len(assets)

    # Sample covariance matrix of daily returns.
    # Annualising improves numerical scaling for SLSQP,
    # without changing the minimum-variance weights.
    covariance = (
        returns_window.cov().to_numpy(dtype=float) * 252
    )

    if not np.isfinite(covariance).all():
        raise ValueError("Invalid covariance matrix.")

    # Start optimisation from equal weights
    initial_weights = np.full(
        n_assets, 1.0 / n_assets
    )

    # Portfolio variance
    def objective(weights):
        return float(
            weights @ covariance @ weights
        )

    # Analytical gradient
    def gradient(weights):
        return 2.0 * covariance @ weights

    # Fully invested portfolio
    constraints = [{
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
        "jac": lambda w: np.ones_like(w)
    }]

    # Long-only, with no additional weight cap
    bounds = [(0.0, 1.0)] * n_assets

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 1000
        }
    )

    if not result.success:
        raise RuntimeError(
            f"MVO optimisation failed: {result.message}"
        )

    weights = pd.Series(
        result.x,
        index=assets,
        name="MVO_Weight"
    )

    # Validate optimisation result
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("MVO produced invalid weights.")

    if (weights < -1e-8).any():
        raise ValueError("MVO produced negative weights.")

    if not np.isclose(
        weights.sum(), 1.0, atol=1e-8
    ):
        raise ValueError("MVO weights do not sum to 1.")

    if not np.isfinite(objective(result.x)):
        raise ValueError("Invalid optimised portfolio variance.")

    return weights
