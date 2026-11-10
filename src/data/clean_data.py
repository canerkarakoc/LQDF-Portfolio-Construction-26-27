
from pathlib import Path
import pandas as pd

# Project directories
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["VTI", "EFA", "AGG", "GLD", "VNQ"]
BENCHMARK = "SPY"

# 1. Load historical adjusted prices
prices = pd.read_csv(
    RAW_DIR / "etf_prices_raw.csv",
    index_col="Date",
    parse_dates=True
)

print("Original shape:", prices.shape)

# 2. Check data quality
duplicates = prices.index.duplicated().sum()
missing = prices.isna().sum()
invalid = (prices <= 0).sum()

print("\nDuplicate dates:", duplicates)
print("\nMissing values:\n", missing)
print("\nNon-positive prices:\n", invalid)

# Stop if unexpected data problems are found
if duplicates > 0:
    raise ValueError("Duplicate dates detected.")

if missing.sum() > 0:
    raise ValueError("Missing prices detected.")

if invalid.sum() > 0:
    raise ValueError("Non-positive prices detected.")

# 3. Sort and align observations
prices = prices.sort_index()
prices = prices[ASSETS + [BENCHMARK]]

# 4. Calculate simple daily returns
returns = prices.pct_change(fill_method=None).dropna()

# 5. Separate asset and benchmark returns
asset_returns = returns[ASSETS]
benchmark_returns = returns[[BENCHMARK]]

# 6. Save processed files
prices.to_csv(PROCESSED_DIR / "etf_prices_clean.csv")

asset_returns.to_csv(
    PROCESSED_DIR / "asset_returns_daily.csv"
)

benchmark_returns.to_csv(
    PROCESSED_DIR / "benchmark_returns_daily.csv"
)

print("\nClean price shape:", prices.shape)
print("Daily returns shape:", returns.shape)

print("\nFirst five daily returns:")
print(returns.head())

print("\nReturn date range:")
print(returns.index.min(), "to", returns.index.max())

print("\nFiles saved successfully.")
