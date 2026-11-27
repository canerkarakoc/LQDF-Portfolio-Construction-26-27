
"""
LQDF Portfolio Construction
Phase 3 - Simulated Trade Order Generation

Convert MVO, HRP and equal-weight target allocations
into proposed dollar-value rebalancing orders.

This is a research simulation only.
No orders are transmitted to a broker.

Assumptions:
- Hypothetical portfolio NAV of $1,000,000
- Fractional dollar-value orders are permitted
- No transaction costs in the order generator
- Orders are calculated before each rebalance day
- Initial portfolio purchases are excluded
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "results" / "comparison"

OUTPUT_DIR = ROOT / "results" / "trade_orders"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = (
    ROOT / "data" / "processed"
    / "asset_returns_daily.csv"
)

ASSETS = ["VTI", "EFA", "AGG", "GLD", "VNQ"]

MODELS = ["MVO", "HRP", "Equal_Weight"]

PORTFOLIO_VALUE = 1_000_000.0


# --------------------------------------------------
# 2. Load historical returns and portfolio weights
# --------------------------------------------------

def load_returns():

    returns = pd.read_csv(
        DATA_FILE,
        index_col="Date",
        parse_dates=True
    )

    returns = returns[ASSETS].sort_index()

    if returns.index.has_duplicates:
        raise ValueError("Duplicate return dates.")

    if returns.isna().any().any():
        raise ValueError("Missing return observations.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Invalid return observations.")

    if (returns <= -1).any().any():
        raise ValueError("Invalid asset returns.")

    return returns


def load_weights(model):

    filename = f"{model.lower()}_weights.csv"

    weights = pd.read_csv(
        INPUT_DIR / filename,
        index_col=0,
        parse_dates=True
    )

    weights = weights[ASSETS].sort_index()

    if weights.index.has_duplicates:
        raise ValueError("Duplicate weight dates.")

    if weights.isna().any().any():
        raise ValueError("Missing portfolio weights.")

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Invalid portfolio weights.")

    if (weights < -1e-8).any().any():
        raise ValueError("Negative portfolio weights.")

    if not np.allclose(weights.sum(axis=1), 1.0):
        raise ValueError("Portfolio weights do not sum to 1.")

    return weights


# --------------------------------------------------
# 3. Identify monthly target allocations
# --------------------------------------------------

def get_monthly_targets(weights):

    monthly_targets = (
        weights
        .groupby(weights.index.to_period("M"))
        .head(1)
        .copy()
    )

    return monthly_targets


# --------------------------------------------------
# 4. Reconstruct pre-rebalance holdings
# --------------------------------------------------

def calculate_pre_rebalance_weights(
    previous_target,
    returns,
    previous_date,
    current_date
):
    """
    Reconstruct current holdings by compounding
    asset returns since the previous rebalance.

    Includes the previous rebalance day's return.
    Excludes the current rebalance day's return.
    """

    period_returns = returns.loc[
        (returns.index >= previous_date)
        & (returns.index < current_date),
        ASSETS
    ]

    if period_returns.empty:
        raise ValueError(
            "No returns between rebalance dates."
        )

    asset_growth = (
        1.0 + period_returns
    ).prod(axis=0)

    holding_values = (
        previous_target * asset_growth
    )

    total_value = holding_values.sum()

    if total_value <= 0:
        raise ValueError("Invalid portfolio value.")

    pre_rebalance_weights = (
        holding_values / total_value
    )

    return pre_rebalance_weights


# --------------------------------------------------
# 5. Generate proposed buy/sell orders
# --------------------------------------------------

def generate_orders(model, weights, returns):

    targets = get_monthly_targets(weights)

    if len(targets) < 2:
        raise ValueError("Insufficient rebalance dates.")

    if not weights.index.isin(returns.index).all():
        raise ValueError(
            "Portfolio dates missing from returns."
        )

    orders = []
    rebalance_summary = []

    dates = list(targets.index)

    # Exclude initial portfolio construction
    for i in range(1, len(dates)):

        previous_date = dates[i - 1]
        current_date = dates[i]

        previous_target = targets.loc[
            previous_date, ASSETS
        ]

        new_target = targets.loc[
            current_date, ASSETS
        ]

        pre_weights = calculate_pre_rebalance_weights(
            previous_target,
            returns,
            previous_date,
            current_date
        )

        # Change required for each asset
        weight_changes = new_target - pre_weights

        # Hypothetical dollar-value orders
        dollar_changes = (
            weight_changes * PORTFOLIO_VALUE
        )

        total_buys = 0.0
        total_sells = 0.0

        for asset in ASSETS:

            trade_value = float(
                dollar_changes[asset]
            )

            if abs(trade_value) < 1e-8:
                side = "HOLD"
                trade_value = 0.0

            elif trade_value > 0:
                side = "BUY"
                total_buys += trade_value

            else:
                side = "SELL"
                total_sells += abs(trade_value)

            orders.append({
                "Date": current_date,
                "Strategy": model,
                "Asset": asset,
                "Current_Weight": pre_weights[asset],
                "Target_Weight": new_target[asset],
                "Weight_Change": weight_changes[asset],
                "Side": side,
                "Trade_Value_USD": trade_value,
                "Absolute_Trade_USD": abs(trade_value),
            })

        one_way_turnover = (
            weight_changes.abs().sum() / 2.0
        )

        # Ensure buys and sells balance
        assert np.isclose(
            total_buys,
            total_sells,
            atol=1e-6
        ), (
            f"Unbalanced trades for {model} "
            f"on {current_date.date()}"
        )

        rebalance_summary.append({
            "Date": current_date,
            "Strategy": model,
            "Total_Buys_USD": total_buys,
            "Total_Sells_USD": total_sells,
            "One_Way_Turnover": one_way_turnover,
            "Largest_Target_Weight": new_target.max(),
            "Number_of_Buys": sum(
                dollar_changes > 1e-8
            ),
            "Number_of_Sells": sum(
                dollar_changes < -1e-8
            ),
        })

    return (
        pd.DataFrame(orders),
        pd.DataFrame(rebalance_summary)
    )


# --------------------------------------------------
# 6. Main execution and validation
# --------------------------------------------------

def main():

    print("LQDF - Simulated Trade Orders")
    print("-----------------------------")

    returns = load_returns()

    all_orders = []
    all_summaries = []

    for model in MODELS:

        print(f"\nProcessing {model}...")

        weights = load_weights(model)

        orders, summary = generate_orders(
            model,
            weights,
            returns
        )

        assert len(summary) == 141
        assert len(orders) == 141 * len(ASSETS)

        all_orders.append(orders)
        all_summaries.append(summary)

        print("Monthly rebalances:", len(summary))

        print(
            "Annualised-style mean monthly turnover:",
            f"{summary['One_Way_Turnover'].mean():.2%}"
        )

        print(
            "Largest proposed individual trade:",
            f"${orders['Absolute_Trade_USD'].max():,.2f}"
        )

        print(
            f"PASS: {model} orders generated."
        )

    # ----------------------------------------------
    # Combine research outputs
    # ----------------------------------------------

    orders_df = pd.concat(
        all_orders,
        ignore_index=True
    )

    summary_df = pd.concat(
        all_summaries,
        ignore_index=True
    )

    # ----------------------------------------------
    # Cross-check turnover against Phase 2
    # ----------------------------------------------

    saved_turnover = pd.read_csv(
        INPUT_DIR / "monthly_turnover.csv",
        index_col=0,
        parse_dates=True
    )

    for model in MODELS:

        calculated = (
            summary_df[
                summary_df["Strategy"] == model
            ]
            .set_index("Date")["One_Way_Turnover"]
            .sort_index()
        )

        expected = (
            saved_turnover[model]
            .dropna()
            .sort_index()
        )

        assert calculated.index.equals(
            expected.index
        )

        assert np.allclose(
            calculated.to_numpy(),
            expected.to_numpy(),
            rtol=0,
            atol=1e-10
        ), f"Turnover mismatch for {model}"

    print(
        "\nPASS: Proposed orders reproduce "
        "Phase 2 turnover."
    )

    # ----------------------------------------------
    # Save outputs
    # ----------------------------------------------

    orders_df.to_csv(
        OUTPUT_DIR / "simulated_trade_orders.csv",
        index=False
    )

    summary_df.to_csv(
        OUTPUT_DIR / "rebalance_summary.csv",
        index=False
    )

    print("\nTotal order records:", len(orders_df))

    print("\nResults saved to:")
    print(OUTPUT_DIR)

    print(
        "\nPASS: Simulated trade-order "
        "generation completed."
    )


if __name__ == "__main__":
    main()
