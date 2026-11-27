
"""
LQDF Portfolio Construction
Phase 3 - Transaction-Cost Sensitivity

Evaluate MVO, HRP and equal-weight portfolios under
alternative proportional transaction-cost assumptions.

Cost convention:
    - Cost in basis points per dollar traded
    - One-way turnover = half the total absolute trade
    - Cost = 2 * one-way turnover * cost rate
    - Initial purchase excluded
    - No additional transaction cost applied to SPY
    - No market impact or financing costs modelled
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "results" / "comparison"
OUTPUT_DIR = ROOT / "results" / "transaction_costs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["MVO", "HRP", "Equal_Weight"]
ALL_STRATEGIES = MODELS + ["SPY"]

COST_SCENARIOS_BPS = [0, 5, 10, 25]
TRADING_DAYS = 252


# --------------------------------------------------
# 2. Load and validate Phase 2 outputs
# --------------------------------------------------

def load_data():

    daily_returns = pd.read_csv(
        INPUT_DIR / "daily_returns.csv",
        index_col=0,
        parse_dates=True
    ).sort_index()

    monthly_turnover = pd.read_csv(
        INPUT_DIR / "monthly_turnover.csv",
        index_col=0,
        parse_dates=True
    ).sort_index()

    for name, frame in [
        ("Daily returns", daily_returns),
        ("Monthly turnover", monthly_turnover)
    ]:
        if frame.index.has_duplicates:
            raise ValueError(
                f"{name}: duplicate dates detected."
            )

    if not set(ALL_STRATEGIES).issubset(
        daily_returns.columns
    ):
        raise ValueError(
            "Missing strategies in daily returns."
        )

    if not set(MODELS).issubset(
        monthly_turnover.columns
    ):
        raise ValueError(
            "Missing models in turnover data."
        )

    daily_returns = daily_returns[ALL_STRATEGIES]
    monthly_turnover = monthly_turnover[MODELS]

    if daily_returns.isna().any().any():
        raise ValueError("Missing daily returns.")

    if monthly_turnover.isna().any().any():
        raise ValueError("Missing monthly turnover.")

    if not np.isfinite(
        daily_returns.to_numpy()
    ).all():
        raise ValueError("Invalid daily returns.")

    if not np.isfinite(
        monthly_turnover.to_numpy()
    ).all():
        raise ValueError("Invalid turnover values.")

    if (daily_returns <= -1).any().any():
        raise ValueError("Daily return below -100%.")

    if (monthly_turnover < 0).any().any():
        raise ValueError("Negative turnover detected.")

    if not monthly_turnover.index.isin(
        daily_returns.index
    ).all():
        raise ValueError(
            "Turnover dates missing from return data."
        )

    return daily_returns, monthly_turnover


# --------------------------------------------------
# 3. Apply transaction costs
# --------------------------------------------------

def apply_costs(gross_returns, turnover, cost_bps):
    """
    Deduct trading costs at rebalance dates.

    A cost of 10 bps means 0.10% of the dollar
    value traded, for both purchases and sales.

    The portfolio pays the cost immediately before
    earning the rebalance day's gross return.
    """

    cost_rate = cost_bps / 10000.0

    daily_turnover = turnover.reindex(
        gross_returns.index,
        fill_value=0.0
    )

    # Convert one-way turnover to total traded value
    daily_cost_fraction = (
        2.0 * daily_turnover * cost_rate
    )

    if (daily_cost_fraction >= 1).any():
        raise ValueError("Trading cost exceeds capital.")

    # Deduct costs, then apply the day's return
    net_returns = (
        (1.0 - daily_cost_fraction)
        * (1.0 + gross_returns)
        - 1.0
    )

    return net_returns, daily_cost_fraction


# --------------------------------------------------
# 4. Performance metrics
# --------------------------------------------------

def calculate_metrics(returns):

    growth = (1.0 + returns).cumprod()
    final_growth = float(growth.iloc[-1])

    annual_return = (
        final_growth ** (TRADING_DAYS / len(returns))
        - 1.0
    )

    annual_volatility = (
        returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    sharpe = (
        returns.mean() * TRADING_DAYS
        / annual_volatility
        if annual_volatility > 0
        else np.nan
    )

    wealth = np.r_[1.0, growth.to_numpy()]
    running_peak = np.maximum.accumulate(wealth)

    max_drawdown = (
        wealth / running_peak - 1.0
    ).min()

    return {
        "Annualised Return": annual_return,
        "Annualised Volatility": annual_volatility,
        "Sharpe (RF=0)": sharpe,
        "Maximum Drawdown": max_drawdown,
        "Final Growth of $1": final_growth,
    }


# --------------------------------------------------
# 5. Run sensitivity analysis
# --------------------------------------------------

def main():

    print("LQDF - Transaction Cost Analysis")
    print("--------------------------------")

    gross_returns, monthly_turnover = load_data()

    results = []
    net_return_series = {}

    for cost_bps in COST_SCENARIOS_BPS:

        print(f"\nTesting cost: {cost_bps} bps")

        for model in MODELS:

            net_returns, costs = apply_costs(
                gross_returns[model],
                monthly_turnover[model],
                cost_bps
            )

            metrics = calculate_metrics(net_returns)

            gross_metrics = calculate_metrics(
                gross_returns[model]
            )

            results.append({
                "Strategy": model,
                "Cost (bps)": cost_bps,
                **metrics,
                "Return Reduction (pp)": (
                    gross_metrics["Annualised Return"]
                    - metrics["Annualised Return"]
                ) * 100,
                "Total Cost Fraction": costs.sum(),
            })

            net_return_series[
                f"{model}_{cost_bps}bps"
            ] = net_returns

        # SPY is held as a buy-and-hold benchmark;
        # initial purchase costs are excluded.
        spy_returns = gross_returns["SPY"]

        results.append({
            "Strategy": "SPY",
            "Cost (bps)": cost_bps,
            **calculate_metrics(spy_returns),
            "Return Reduction (pp)": 0.0,
            "Total Cost Fraction": 0.0,
        })

        net_return_series[
            f"SPY_{cost_bps}bps"
        ] = spy_returns

    # ----------------------------------------------
    # Build output tables
    # ----------------------------------------------

    results_df = pd.DataFrame(results)

    net_returns_df = pd.DataFrame(
        net_return_series
    )

    growth_df = (
        1.0 + net_returns_df
    ).cumprod()

    # ----------------------------------------------
    # Internal validation
    # ----------------------------------------------

    for strategy in ALL_STRATEGIES:

        zero_cost = net_returns_df[
            f"{strategy}_0bps"
        ]

        assert np.allclose(
            zero_cost.to_numpy(),
            gross_returns[strategy].to_numpy(),
            rtol=0,
            atol=1e-12
        )

    print(
        "\nPASS: Zero-cost results reproduce "
        "Phase 2 returns."
    )

    # Costs should not increase final wealth.
    for model in MODELS:

        values = []

        for bps in COST_SCENARIOS_BPS:
            values.append(
                growth_df[
                    f"{model}_{bps}bps"
                ].iloc[-1]
            )

        assert np.all(
            np.diff(values) <= 1e-10
        ), f"Cost sensitivity failed for {model}"

    print(
        "PASS: Higher costs do not increase "
        "final portfolio wealth."
    )

    # ----------------------------------------------
    # Save results
    # ----------------------------------------------

    results_df.to_csv(
        OUTPUT_DIR / "cost_sensitivity_metrics.csv",
        index=False
    )

    net_returns_df.to_csv(
        OUTPUT_DIR / "net_daily_returns.csv",
        index_label="Date"
    )

    growth_df.to_csv(
        OUTPUT_DIR / "net_cumulative_growth.csv",
        index_label="Date"
    )

    # ----------------------------------------------
    # Print performance comparison
    # ----------------------------------------------

    print("\nAnnualised net returns (%):")

    table = results_df.pivot(
        index="Strategy",
        columns="Cost (bps)",
        values="Annualised Return"
    )

    print((table * 100).round(3).to_string())

    print("\nReturn reduction at 25 bps")
    print("(percentage points):")

    high_cost = results_df[
        results_df["Cost (bps)"] == 25
    ]

    print(
        high_cost[
            ["Strategy", "Return Reduction (pp)"]
        ].round(4).to_string(index=False)
    )

    print("\nFiles saved to:")
    print(OUTPUT_DIR)

    print(
        "\nPASS: Transaction-cost "
        "sensitivity analysis completed."
    )


if __name__ == "__main__":
    main()
