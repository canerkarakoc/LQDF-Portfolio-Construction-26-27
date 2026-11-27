
"""
LQDF Portfolio Construction
Phase 3 - Statistical Uncertainty

Paired moving-block bootstrap of daily return differences.

Measures:
- Annualised arithmetic return difference
- 95% bootstrap confidence interval
- Approximate two-sided bootstrap p-value

This is an exploratory analysis, not a definitive
test of model superiority.

Transaction costs are excluded.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "results" / "comparison"
    / "daily_returns.csv"
)

OUTPUT_DIR = ROOT / "results" / "statistical_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISONS = [
    ("HRP", "MVO"),
    ("HRP", "Equal_Weight"),
    ("MVO", "Equal_Weight"),
    ("SPY", "Equal_Weight"),
]

TRADING_DAYS = 252
BLOCK_LENGTH = 21
N_BOOTSTRAPS = 3000
SEED = 42


def load_data():

    returns = pd.read_csv(
        INPUT_FILE,
        index_col=0,
        parse_dates=True
    ).sort_index()

    required = {
        name
        for pair in COMPARISONS
        for name in pair
    }

    if not required.issubset(returns.columns):
        raise ValueError("Missing strategy columns.")

    if returns.index.has_duplicates:
        raise ValueError("Duplicate trading dates.")

    if returns.isna().any().any():
        raise ValueError("Missing daily returns.")

    if not np.isfinite(
        returns.to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid daily returns.")

    if len(returns) < BLOCK_LENGTH:
        raise ValueError("Insufficient observations.")

    return returns


def bootstrap_comparison(differences, rng):
    """
    Circular moving-block bootstrap.

    Resample daily paired-return differences
    using contiguous blocks of 21 trading days.
    """

    differences = np.asarray(
        differences, dtype=float
    )

    n = len(differences)

    observed = (
        differences.mean() * TRADING_DAYS
    )

    bootstrap_estimates = np.empty(
        N_BOOTSTRAPS
    )

    number_of_blocks = int(
        np.ceil(n / BLOCK_LENGTH)
    )

    for b in range(N_BOOTSTRAPS):

        starts = rng.integers(
            0, n, size=number_of_blocks
        )

        indices = np.concatenate([
            (start + np.arange(BLOCK_LENGTH)) % n
            for start in starts
        ])[:n]

        bootstrap_estimates[b] = (
            differences[indices].mean()
            * TRADING_DAYS
        )

    # Percentile confidence interval
    lower, upper = np.percentile(
        bootstrap_estimates,
        [2.5, 97.5]
    )

    # Approximate two-sided test under zero mean:
    # centre bootstrap sampling errors around zero
    centred_errors = (
        bootstrap_estimates - observed
    )

    p_value = (
        np.count_nonzero(
            np.abs(centred_errors) >= abs(observed)
        ) + 1
    ) / (N_BOOTSTRAPS + 1)

    return {
        "Annualised_Difference": observed,
        "CI_95_Lower": lower,
        "CI_95_Upper": upper,
        "Approx_P_Value": p_value,
        "CI_Excludes_Zero": bool(
            lower > 0 or upper < 0
        ),
    }


def main():

    print("LQDF - Statistical Uncertainty")
    print("------------------------------")

    returns = load_data()

    print("\nDaily observations:", len(returns))
    print("Bootstrap samples:", N_BOOTSTRAPS)
    print("Block length:", BLOCK_LENGTH)

    rng = np.random.default_rng(SEED)

    results = []

    for strategy_a, strategy_b in COMPARISONS:

        differences = (
            returns[strategy_a]
            - returns[strategy_b]
        ).to_numpy()

        outcome = bootstrap_comparison(
            differences, rng
        )

        results.append({
            "Strategy_A": strategy_a,
            "Strategy_B": strategy_b,
            **outcome
        })

        print(
            f"\n{strategy_a} minus {strategy_b}"
        )

        print(
            "Annualised return difference:",
            f"{outcome['Annualised_Difference']:.2%}"
        )

        print(
            "95% confidence interval:",
            f"[{outcome['CI_95_Lower']:.2%}, "
            f"{outcome['CI_95_Upper']:.2%}]"
        )

        print(
            "Approximate two-sided p-value:",
            f"{outcome['Approx_P_Value']:.4f}"
        )

        print(
            "Interval excludes zero:",
            outcome["CI_Excludes_Zero"]
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "bootstrap_comparisons.csv",
        index=False
    )

    print("\nResults saved to:")
    print(OUTPUT_DIR)

    print(
        "\nPASS: Statistical uncertainty "
        "analysis completed."
    )


if __name__ == "__main__":
    main()
