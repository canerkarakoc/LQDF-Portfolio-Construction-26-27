
from pathlib import Path

import pandas as pd
import yfinance as yf

# Project settings
TICKERS = ["VTI", "EFA", "AGG", "GLD", "VNQ", "SPY"]
START_DATE = "2010-01-01"
END_DATE = "2026-11-01"  # Exclusive: includes October 2026

# Save files relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Download adjusted daily prices
data = yf.download(
    tickers=TICKERS,
    start=START_DATE,
    end=END_DATE,
    interval="1d",
    auto_adjust=True,
    progress=False,
)

# With auto_adjust=True, Close is adjusted
prices = data["Close"].copy()
prices = prices.reindex(columns=TICKERS)
prices.index.name = "Date"

# Initial checks — do not clean the data yet
print("Dataset shape:", prices.shape)
print("\nFirst five observations:")
print(prices.head())

print("\nMissing values by asset:")
print(prices.isna().sum())

print("\nAvailable date range:")
print(prices.index.min(), "to", prices.index.max())

# Save raw downloaded prices
output_path = OUTPUT_DIR / "etf_prices_raw.csv"
prices.to_csv(output_path)

print(f"\nSaved raw prices to: {output_path}")
