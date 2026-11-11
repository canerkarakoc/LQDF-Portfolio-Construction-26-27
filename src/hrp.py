
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "asset_returns_daily.csv"
OUTPUT_DIR = ROOT / "results" / "hrp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["VTI", "EFA", "AGG", "GLD", "VNQ"]

START_DATE = "2015-01-01"
END_DATE = "2026-10-30"


# --------------------------------------------------
# 2. Load and validate daily returns
# --------------------------------------------------

def load_returns():
    """Load daily asset returns and select the agreed test period."""

    returns = pd.read_csv(
        DATA_FILE,
        index_col="Date",
        parse_dates=True,
    )

    missing_assets = [
        asset for asset in ASSETS
        if asset not in returns.columns
    ]

    if missing_assets:
        raise ValueError(
            f"Missing required assets: {missing_assets}"
        )

    returns = returns[ASSETS].sort_index()

    if returns.index.has_duplicates:
        raise ValueError("Duplicate dates found in returns data.")

    returns = returns.loc[START_DATE:END_DATE].copy()

    if returns.empty:
        raise ValueError("No observations in the selected period.")

    if returns.isna().any().any():
        raise ValueError("Missing asset returns detected.")

    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite returns detected.")

    if (returns <= -1).any().any():
        raise ValueError("Invalid return of -100% or below.")

    return returns.astype(float)


# --------------------------------------------------
# 3. Correlation and hierarchical clustering
# --------------------------------------------------


def calculate_clustering(returns):
    correlation = returns.corr()

    distance_values = np.sqrt(
        np.clip(
            (1.0 - correlation.to_numpy(dtype=float)) / 2.0,
            0.0,
            None
        )
    )

    np.fill_diagonal(distance_values, 0.0)

    condensed_distance = squareform(distance_values, checks=True)

    linkage_matrix = linkage(
        condensed_distance,
        method="single"
    )

    leaf_order = leaves_list(linkage_matrix)
    ordered_assets = returns.columns[leaf_order].tolist()

    return correlation, linkage_matrix, ordered_assets


    # Order assets according to dendrogram leaves
    ordered_indices = leaves_list(linkage_matrix)
    ordered_assets = correlation.columns[ordered_indices]

    return correlation, linkage_matrix, ordered_assets


# --------------------------------------------------
# 4. HRP allocation
# --------------------------------------------------

def get_cluster_variance(covariance, cluster):
    """Calculate variance of an inverse-variance weighted cluster."""

    cluster_covariance = covariance.loc[cluster, cluster]

    inverse_variance = 1 / np.diag(
        cluster_covariance.to_numpy()
    )

    weights = inverse_variance / inverse_variance.sum()

    variance = float(
        weights @ cluster_covariance.to_numpy() @ weights
    )

    return variance


def calculate_hrp_weights(returns, ordered_assets):
    """Calculate HRP weights using recursive bisection."""

    covariance = returns.cov()
    weights = pd.Series(1.0, index=ordered_assets)

    def recursive_bisection(cluster):
        if len(cluster) <= 1:
            return

        split = len(cluster) // 2

        left_cluster = cluster[:split]
        right_cluster = cluster[split:]

        left_variance = get_cluster_variance(
            covariance, left_cluster
        )

        right_variance = get_cluster_variance(
            covariance, right_cluster
        )

        total_variance = left_variance + right_variance

        if total_variance <= 0:
            raise ValueError("Invalid cluster variance.")

        # Allocate more weight to the lower-risk cluster
        left_allocation = 1 - (
            left_variance / total_variance
        )

        weights.loc[left_cluster] *= left_allocation
        weights.loc[right_cluster] *= (1 - left_allocation)

        recursive_bisection(left_cluster)
        recursive_bisection(right_cluster)

    recursive_bisection(list(ordered_assets))

    # Return weights in the original asset order
    weights = weights.reindex(ASSETS)
    weights = weights / weights.sum()

    return weights


# --------------------------------------------------
# 5. Save visualisations
# --------------------------------------------------

def save_heatmap(correlation, ordered_assets):
    """Save the reordered asset correlation heatmap."""

    ordered_correlation = correlation.loc[
        ordered_assets, ordered_assets
    ]

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        ordered_correlation,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
    )

    plt.title("Asset Return Correlation Matrix (2015–2026)")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_dendrogram(linkage_matrix, assets):
    """Save the hierarchical clustering dendrogram."""

    plt.figure(figsize=(10, 6))

    dendrogram(
        linkage_matrix,
        labels=assets,
        leaf_rotation=0,
        leaf_font_size=11,
    )

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Assets")
    plt.ylabel("Correlation Distance")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "dendrogram.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# --------------------------------------------------
# 6. Main execution
# --------------------------------------------------

def main():
    returns = load_returns()

    correlation, linkage_matrix, ordered_assets = (
        calculate_clustering(returns)
    )

    weights = calculate_hrp_weights(
        returns, ordered_assets
    )

    # Save results
    correlation.to_csv(
        OUTPUT_DIR / "correlation_matrix.csv"
    )

    weights.rename("HRP_Weight").to_csv(
        OUTPUT_DIR / "hrp_weights.csv"
    )

    save_heatmap(correlation, ordered_assets)
    save_dendrogram(linkage_matrix, ASSETS)

    # Display results
    print("\nHRP analysis completed successfully.")
    print(f"Testing period: {returns.index.min().date()} "
          f"to {returns.index.max().date()}")
    print(f"Observations: {len(returns)}")

    print("\nHierarchical asset ordering:")
    print(list(ordered_assets))

    print("\nPreliminary HRP portfolio weights:")
    print((weights * 100).round(2).astype(str) + "%")

    print(f"\nTotal weight: {weights.sum():.6f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
