
"""
LQDF Portfolio Construction
Phase 3 - Institutional Feasibility & Scalability

Estimate the dollar trading requirements implied by
historical MVO, HRP and equal-weight rebalances.

This is a capacity screening exercise, not
a live trading or execution model.

The script calculates required ADV but does not
verify actual market volume or execution capacity.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "results" / "trade_orders"
OUTPUT_DIR = ROOT / "results" / "institutional_feasibility"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_NAV = 1_000_000.0

PORTFOLIO_SIZES = [
    1_000_000,
    100_000_000,
    1_000_000_000,
]

MAX_PARTICIPATION = 0.05

STRATEGIES = ["MVO", "HRP", "Equal_Weight"]


def load_orders():

    orders = pd.read_csv(
        INPUT_DIR / "simulated_trade_orders.csv",
        parse_dates=["Date"]
    )

    required = {
        "Date",
        "Strategy",
        "Asset",
        "Target_Weight",
        "Trade_Value_USD",
        "Absolute_Trade_USD",
    }

    if not required.issubset(orders.columns):
        raise ValueError("Missing order columns.")

    if orders.empty:
        raise ValueError("No orders available.")

    if orders.duplicated(
        ["Date", "Strategy", "Asset"]
    ).any():
        raise ValueError("Duplicate orders detected.")

    numeric = orders[
        ["Target_Weight",
         "Trade_Value_USD",
         "Absolute_Trade_USD"]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("Invalid order values.")

    if not np.allclose(
        np.abs(orders["Trade_Value_USD"]),
        orders["Absolute_Trade_USD"],
        atol=0.01
    ):
        raise ValueError("Trade values are inconsistent.")

    if not orders["Strategy"].isin(STRATEGIES).all():
        raise ValueError("Unexpected strategy detected.")

    return orders


def calculate_requirements(orders):

    scenarios = []

    for nav in PORTFOLIO_SIZES:

        scale_factor = nav / BASE_NAV

        scaled = orders.copy()

        scaled["Portfolio_NAV_USD"] = nav

        scaled["Scaled_Trade_USD"] = (
            scaled["Trade_Value_USD"] * scale_factor
        )

        scaled["Absolute_Scaled_Trade_USD"] = (
            scaled["Absolute_Trade_USD"] * scale_factor
        )

        # Required market ADV for a single-day trade
        # limited to 5% of that asset's daily dollar volume.
        scaled["Required_Dollar_ADV_USD"] = (
            scaled["Absolute_Scaled_Trade_USD"]
            / MAX_PARTICIPATION
        )

        scenarios.append(scaled)

    return pd.concat(scenarios, ignore_index=True)


def calculate_summary(scenarios):

    rows = []

    for (nav, strategy), group in scenarios.groupby(
        ["Portfolio_NAV_USD", "Strategy"]
    ):

        # Each rebalance has one record per asset.
        rebalance_trading = group.groupby("Date")[
            "Absolute_Scaled_Trade_USD"
        ].sum()

        largest_trade = float(
            group["Absolute_Scaled_Trade_USD"].max()
        )

        max_required_adv = float(
            group["Required_Dollar_ADV_USD"].max()
        )

        rows.append({
            "Portfolio_NAV_USD": nav,
            "Strategy": strategy,
            "Rebalances": group["Date"].nunique(),
            "Average_Gross_Trading_Per_Rebalance_USD":
                rebalance_trading.mean(),
            "Largest_Individual_Trade_USD":
                largest_trade,
            "Largest_Trade_Fraction_of_NAV":
                largest_trade / nav,
            "Required_ADV_for_Largest_Trade_USD":
                max_required_adv,
            "Average_Target_Concentration":
                group.groupby("Date")[
                    "Target_Weight"
                ].max().mean(),
        })

    return pd.DataFrame(rows)


def main():

    print("LQDF - Institutional Feasibility")
    print("--------------------------------")

    orders = load_orders()

    assert len(orders) == 2115

    scenarios = calculate_requirements(orders)

    summary = calculate_summary(scenarios)

    # Sanity checks
    assert len(scenarios) == (
        len(orders) * len(PORTFOLIO_SIZES)
    )

    assert len(summary) == (
        len(STRATEGIES) * len(PORTFOLIO_SIZES)
    )

    assert np.allclose(
        scenarios["Required_Dollar_ADV_USD"],
        scenarios["Absolute_Scaled_Trade_USD"]
        / MAX_PARTICIPATION
    )

    # Scaling NAV must not change portfolio weight
    # concentration or trade fractions of NAV.
    for strategy in STRATEGIES:

        subset = summary[
            summary["Strategy"] == strategy
        ]

        assert np.allclose(
            subset["Largest_Trade_Fraction_of_NAV"],
            subset["Largest_Trade_Fraction_of_NAV"].iloc[0]
        )

        assert np.allclose(
            subset["Average_Target_Concentration"],
            subset["Average_Target_Concentration"].iloc[0]
        )

    scenarios.to_csv(
        OUTPUT_DIR / "scaled_trade_requirements.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "institutional_capacity_summary.csv",
        index=False
    )

    print("\nParticipation limit: 5% of daily ADV")
    print("Execution horizon: 1 trading day")

    for nav in PORTFOLIO_SIZES:

        print(f"\nPortfolio NAV: ${nav:,.0f}")

        subset = summary[
            summary["Portfolio_NAV_USD"] == nav
        ]

        for _, row in subset.iterrows():

            print(f"\n{row['Strategy']}")

            print(
                "Largest individual trade:",
                f"${row['Largest_Individual_Trade_USD']:,.0f}"
            )

            print(
                "Minimum ADV needed for that trade:",
                f"${row['Required_ADV_for_Largest_Trade_USD']:,.0f}"
            )

            print(
                "Average largest target position:",
                f"{row['Average_Target_Concentration']:.2%}"
            )

    print("\nPASS: Scaling calculations validated.")

    print("\nFiles saved to:")
    print(OUTPUT_DIR)

    print(
        "\nNOTE: Actual ETF market liquidity "
        "has not yet been verified."
    )


if __name__ == "__main__":
    main()
