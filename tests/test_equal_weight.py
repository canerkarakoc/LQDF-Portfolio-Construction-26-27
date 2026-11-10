
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Load the portfolio weights
weights = pd.read_csv(
    ROOT / "results/benchmarks/equal_weight_allocations.csv",
    index_col="Date",
    parse_dates=True
)

# Check 1: Portfolio weights always sum to 100%
assert np.allclose(weights.sum(axis=1), 1.0)
print("PASS: All portfolio weights sum to 100%")

# Check 2: Weights reset to 20% at the start of each month
first_day = weights.groupby(weights.index.to_period("M")).head(1)

assert np.allclose(first_day.to_numpy(), 0.20)
print("PASS: Portfolio resets to 20% per asset monthly")

# Check 3: Weights drift between monthly rebalances
within_month = weights.drop(index=first_day.index)

assert not within_month.empty
assert ((within_month - 0.20).abs() > 1e-8).any().any()
print("PASS: Portfolio weights drift between rebalances")

# Display sample weights
print("\nFirst trading day of each month:")
print(first_day.head())

print("\nExample of weight drift:")
print(weights.head(5).round(6))
