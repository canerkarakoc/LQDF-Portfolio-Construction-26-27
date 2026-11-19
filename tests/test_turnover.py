
"""
LQDF Portfolio Construction
Phase 2 - Independent Turnover Validation

Reconstruct pre-rebalance weights from the previous
target allocation and compounded asset returns.

Compare independently calculated turnover against
the saved results from compare_models.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results" / "comparison"

DATA_FILE = (
    ROOT / "data" / "processed"
    / "asset_returns_daily.csv"
)

ASSETS = ["VTI", "EFA", "AGG", "GLD", "VNQ"]

MODELS = ["MVO", "HRP", "Equal_Weight"]


# --------------------------------------------------
# 2. Load historical data and saved turnover
# --------------------------------------------------

def load_data():

    returns = pd.read_csv(
        DATA_FILE,
        index_col="Date",
        parse_dates=True
    )[ASSETS].sort_index()

    # Use the first CSV column as the date index,
    # regardless of whether it is named "Date".
    saved_turnover = pd.read_csv(
        RESULTS / "monthly_turnover.csv",
        index_col=0,
        parse_dates=True
    ).sort_index()

    if returns.index.has_duplicates:
        raise ValueError("Duplicate return dates.")

    if saved_turnover.index.has_duplicates:
        raise ValueError("Duplicate turnover dates.")

    return returns, saved_turnover


# --------------------------------------------------
# 3. Independently calculate portfolio turnover
# --------------------------------------------------

def independent_turnover(weights, returns):
    """
    Reconstruct pre-rebalance portfolio holdings.

    Start from the previous month's target weights,
    compound asset returns up to the day before the
    next rebalance, and compare with the new target.

    Initial portfolio purchases are excluded.
    """

    rebalance_dates = (
        weights.groupby(
            weights.index.to_period("M")
        ).head(1).index
    )

    turnover = {}

    for i in range(1, len(rebalance_dates)):

        previous_date = rebalance_dates[i - 1]
        current_date = rebalance_dates[i]

        # Target weights at previous rebalance
        starting_weights = weights.loc[
            previous_date, ASSETS
        ].to_numpy(dtype=float)

        # Include the previous rebalance day's return
        # but exclude the current rebalance day's return.
        period_returns = returns.loc[
            (returns.index >= previous_date)
            & (returns.index < current_date),
            ASSETS
        ]

        # Compound individual asset returns
        asset_growth = (
            1.0 + period_returns
        ).prod(axis=0).to_numpy(dtype=float)

        # Value of holdings before the new rebalance
        holding_values = (
            starting_weights * asset_growth
        )

        pre_rebalance_weights = (
            holding_values / holding_values.sum()
        )

        # New target allocation
        new_target = weights.loc[
            current_date, ASSETS
        ].to_numpy(dtype=float)

        # One-way turnover
        turnover[current_date] = (
            0.5
            * np.abs(
                new_target - pre_rebalance_weights
            ).sum()
        )

    return pd.Series(
        turnover,
        dtype=float,
        name="Independent_Turnover"
    )


# --------------------------------------------------
# 4. Run validation for all three models
# --------------------------------------------------

def main():

    print("LQDF - Turnover Validation")
    print("-------------------------")

    returns, saved_turnover = load_data()

    for model in MODELS:

        print(f"\nTesting {model}...")

        weights = pd.read_csv(
            RESULTS / f"{model.lower()}_weights.csv",
            index_col=0,
            parse_dates=True
        )[ASSETS].sort_index()

        if weights.index.has_duplicates:
            raise ValueError(
                f"Duplicate weight dates for {model}."
            )

        if weights.isna().any().any():
            raise ValueError(
                f"Missing portfolio weights for {model}."
            )

        # Calculate turnover independently
        calculated = independent_turnover(
            weights,
            returns
        )

        # Load turnover recorded by the comparison script
        recorded = saved_turnover[
            model
        ].dropna()

        # Check identical rebalance dates
        assert calculated.index.equals(
            recorded.index
        ), f"{model}: Turnover dates do not match."

        # 142 monthly allocations, excluding initial purchase
        assert len(calculated) == 141, (
            f"{model}: Expected 141 rebalances."
        )

        # Validate calculations
        assert np.isfinite(
            calculated.to_numpy()
        ).all(), f"{model}: Non-finite turnover detected."

        assert (calculated >= 0).all(), (
            f"{model}: Negative turnover detected."
        )

        # Compare independently reconstructed turnover
        # against the original comparison results.
        differences = np.abs(
            calculated.to_numpy()
            - recorded.to_numpy()
        )

        max_difference = float(
            differences.max()
        )

        print(
            "Rebalances checked:",
            len(calculated)
        )

        print(
            "Maximum difference:",
            f"{max_difference:.12f}"
        )

        assert np.allclose(
            calculated.to_numpy(),
            recorded.to_numpy(),
            rtol=0,
            atol=1e-10
        ), (
            f"{model}: Independent turnover "
            "does not match saved results."
        )

        # Annualise one-way turnover
        years = len(weights) / 252

        annualised_turnover = (
            calculated.sum() / years
        )

        print(
            "Annualised turnover:",
            f"{annualised_turnover:.2%}"
        )

        print(
            f"PASS: {model} turnover independently verified."
        )

    print(
        "\nPASS: All turnover validation checks completed."
    )


if __name__ == "__main__":
    main()
