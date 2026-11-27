
"""
LQDF Portfolio Construction
Phase 3 - Market Regime Analysis

Compare MVO, HRP, Equal Weight and SPY across
predefined historical market environments.

All calculations use saved Phase 2 daily returns.

The analysis is descriptive:
- Regimes are selected in advance
- Transaction costs are excluded
- Sharpe ratios assume a 0% risk-free rate
- Short-period annualised statistics require caution
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "results" / "comparison"

OUTPUT_DIR = ROOT / "results" / "market_regimes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["MVO", "HRP", "Equal_Weight", "SPY"]
PORTFOLIO_MODELS = ["MVO", "HRP", "Equal_Weight"]

ASSETS = ["VTI", "EFA", "AGG", "GLD", "VNQ"]

TRADING_DAYS = 252

REGIMES = {
    "Pre_COVID": ("2017-01-01", "2019-12-31"),
    "COVID_Shock": ("2020-02-20", "2020-04-30"),
    "Inflation_2022": ("2022-01-01", "2022-12-31"),
    "Recovery_2023_2024": (
        "2023-01-01", "2024-12-31"
    ),
}


# --------------------------------------------------
# 2. Load results from Phase 2
# --------------------------------------------------

def load_data():

    returns = pd.read_csv(
        INPUT_DIR / "daily_returns.csv",
        index_col=0,
        parse_dates=True
    ).sort_index()

    returns = returns[STRATEGIES]

    if returns.empty:
        raise ValueError("No returns found.")

    if returns.index.has_duplicates:
        raise ValueError("Duplicate return dates.")

    if returns.isna().any().any():
        raise ValueError("Missing return observations.")

    if not np.isfinite(
        returns.to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid numerical returns.")

    if (returns <= -1).any().any():
        raise ValueError("Invalid return below -100%.")

    weights = {}

    for model in PORTFOLIO_MODELS:

        filename = f"{model.lower()}_weights.csv"

        frame = pd.read_csv(
            INPUT_DIR / filename,
            index_col=0,
            parse_dates=True
        ).sort_index()

        frame = frame[ASSETS]

        if not frame.index.equals(returns.index):
            raise ValueError(
                f"{model}: Weight dates do not "
                "match strategy return dates."
            )

        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(
                f"{model}: Invalid weights."
            )

        if (frame < -1e-8).any().any():
            raise ValueError(
                f"{model}: Negative weights."
            )

        if not np.allclose(
            frame.sum(axis=1), 1.0
        ):
            raise ValueError(
                f"{model}: Weights do not sum to one."
            )

        weights[model] = frame

    return returns, weights


# --------------------------------------------------
# 3. Calculate regime performance
# --------------------------------------------------

def calculate_performance(returns):

    n_days = len(returns)

    if n_days < 2:
        raise ValueError(
            "Insufficient observations in regime."
        )

    # Compounded cumulative total return
    growth = (1.0 + returns).cumprod()

    total_return = float(
        growth.iloc[-1] - 1.0
    )

    # Annualised volatility
    volatility = float(
        returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    # Preliminary Sharpe ratio (RF=0)
    sharpe = (
        float(
            returns.mean() * TRADING_DAYS
            / volatility
        )
        if volatility > 0
        else np.nan
    )

    # Drawdown starts from initial wealth = 1
    wealth = np.r_[1.0, growth.to_numpy()]

    running_peak = np.maximum.accumulate(
        wealth
    )

    maximum_drawdown = float(
        (wealth / running_peak - 1.0).min()
    )

    return {
        "Observations": n_days,
        "Total_Return": total_return,
        "Annualised_Volatility": volatility,
        "Sharpe_RF0": sharpe,
        "Maximum_Drawdown": maximum_drawdown,
        "Final_Growth_of_1": float(growth.iloc[-1]),
    }


# --------------------------------------------------
# 4. Calculate portfolio concentration
# --------------------------------------------------

def calculate_exposures(weights):

    hhi = weights.pow(2).sum(axis=1)

    largest_weight = weights.max(axis=1)

    dominant_asset = weights.idxmax(axis=1)

    return {
        "Average_AGG_Weight": float(
            weights["AGG"].mean()
        ),
        "Average_HHI": float(
            hhi.mean()
        ),
        "Average_Largest_Weight": float(
            largest_weight.mean()
        ),
        "AGG_Dominant_Fraction": float(
            (dominant_asset == "AGG").mean()
        ),
    }


# --------------------------------------------------
# 5. Run regime comparisons
# --------------------------------------------------

def main():

    print("LQDF - Market Regime Analysis")
    print("----------------------------")

    returns, weights = load_data()

    results = []
    exposure_results = []

    for regime, (start, end) in REGIMES.items():

        period = returns.loc[start:end]

        if period.empty:
            raise ValueError(
                f"No observations for {regime}."
            )

        print(f"\nAnalysing {regime}...")
        print(
            "Period:",
            period.index.min().date(),
            "to",
            period.index.max().date()
        )
        print("Trading days:", len(period))

        for strategy in STRATEGIES:

            metrics = calculate_performance(
                period[strategy]
            )

            results.append({
                "Regime": regime,
                "Strategy": strategy,
                **metrics
            })

        for model in PORTFOLIO_MODELS:

            period_weights = weights[model].loc[
                period.index
            ]

            exposures = calculate_exposures(
                period_weights
            )

            exposure_results.append({
                "Regime": regime,
                "Strategy": model,
                **exposures
            })

    # ----------------------------------------------
    # 6. Create result tables
    # ----------------------------------------------

    performance_df = pd.DataFrame(results)

    exposures_df = pd.DataFrame(
        exposure_results
    )

    return_table = performance_df.pivot(
        index="Strategy",
        columns="Regime",
        values="Total_Return"
    )

    volatility_table = performance_df.pivot(
        index="Strategy",
        columns="Regime",
        values="Annualised_Volatility"
    )

    drawdown_table = performance_df.pivot(
        index="Strategy",
        columns="Regime",
        values="Maximum_Drawdown"
    )

    agg_table = exposures_df.pivot(
        index="Strategy",
        columns="Regime",
        values="Average_AGG_Weight"
    )

    # ----------------------------------------------
    # 7. Validate results
    # ----------------------------------------------

    assert len(performance_df) == (
        len(REGIMES) * len(STRATEGIES)
    )

    assert len(exposures_df) == (
        len(REGIMES) * len(PORTFOLIO_MODELS)
    )

    assert np.allclose(
        performance_df["Final_Growth_of_1"],
        1.0 + performance_df["Total_Return"]
    )

    print(
        "\nPASS: All regime performance "
        "calculations completed."
    )

    # ----------------------------------------------
    # 8. Save research outputs
    # ----------------------------------------------

    performance_df.to_csv(
        OUTPUT_DIR / "regime_performance.csv",
        index=False
    )

    exposures_df.to_csv(
        OUTPUT_DIR / "regime_exposures.csv",
        index=False
    )

    return_table.to_csv(
        OUTPUT_DIR / "regime_returns_table.csv"
    )

    volatility_table.to_csv(
        OUTPUT_DIR / "regime_volatility_table.csv"
    )

    drawdown_table.to_csv(
        OUTPUT_DIR / "regime_drawdowns_table.csv"
    )

    agg_table.to_csv(
        OUTPUT_DIR / "regime_agg_exposure.csv"
    )

    # ----------------------------------------------
    # 9. Display key results
    # ----------------------------------------------

    print("\nCumulative returns by regime (%):")

    print(
        (return_table * 100)
        .round(2)
        .to_string()
    )

    print("\nAnnualised volatility by regime (%):")

    print(
        (volatility_table * 100)
        .round(2)
        .to_string()
    )

    print("\nMaximum drawdown by regime (%):")

    print(
        (drawdown_table * 100)
        .round(2)
        .to_string()
    )

    print("\nAverage AGG allocations (%):")

    print(
        (agg_table * 100)
        .round(2)
        .to_string()
    )

    print("\nResults saved to:")
    print(OUTPUT_DIR)

    print(
        "\nPASS: Market regime analysis completed."
    )


if __name__ == "__main__":
    main()
