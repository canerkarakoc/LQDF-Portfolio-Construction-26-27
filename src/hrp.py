
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform


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
    """
    Calculate correlation, primary distance and secondary distance,
    then perform hierarchical clustering.

    The primary distance measures the dissimilarity between each
    pair of assets based on their correlation.

    The secondary distance measures the dissimilarity between
    the primary distance profiles of each pair of assets.
    """

    # 1. Calculate the correlation matrix
    correlation = returns.corr()

    # 2. Calculate the primary distance matrix
    primary_distance_values = np.sqrt(
        np.clip(
            (1.0 - correlation.to_numpy(dtype=float)) / 2.0,
            0.0,
            None
        )
    )

    np.fill_diagonal(primary_distance_values, 0.0)

    # 3. Calculate the secondary distance matrix
    # Compare the primary distance profiles of each asset.
    secondary_distance_values = squareform(
        pdist(
            primary_distance_values.T,
            metric="euclidean"
        )
    )

    # 4. Convert the secondary distance matrix into
    # the condensed format required by SciPy linkage.
    condensed_secondary_distance = squareform(
        secondary_distance_values,
        checks=True
    )

    # 5. Perform hierarchical clustering using single linkage
    linkage_matrix = linkage(
        condensed_secondary_distance,
        method="single"
    )

    # 6. Obtain the quasi-diagonal asset ordering
    leaf_order = leaves_list(linkage_matrix)

    ordered_assets = returns.columns[leaf_order].tolist()

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

    # Return weights in the original input order
    weights = weights.reindex(returns.columns)
    weights = weights / weights.sum()

    return weights


def calculate_weights(returns_window):
    """Calculate HRP weights for a window of asset returns."""

    _, _, ordered_assets = calculate_clustering(returns_window)
    return calculate_hrp_weights(returns_window, ordered_assets)

def diagnose_hrp_splits(returns, ordered_assets):
    """Print cluster variances and allocations at every HRP split."""

    covariance = returns.cov()

    def inspect(cluster, level=0):

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

        left_allocation = (
            right_variance / total_variance
        )

        right_allocation = (
            left_variance / total_variance
        )

        indent = "    " * level

        print(f"\n{indent}Split: {cluster}")
        print(f"{indent}Left cluster: {left_cluster}")
        print(f"{indent}Right cluster: {right_cluster}")
        print(f"{indent}Left variance: {left_variance:.8f}")
        print(f"{indent}Right variance: {right_variance:.8f}")
        print(f"{indent}Left allocation: {left_allocation:.2%}")
        print(f"{indent}Right allocation: {right_allocation:.2%}")

        inspect(left_cluster, level + 1)
        inspect(right_cluster, level + 1)

    inspect(list(ordered_assets))



# --------------------------------------------------
# 5. Save visualisations
# --------------------------------------------------


def save_heatmap(correlation, ordered_assets, period_name):
    """Save the reordered asset correlation heatmap for a period."""

    ordered_correlation = correlation.loc[
        ordered_assets, ordered_assets
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    plt.title(f"Asset Return Correlation Matrix ({period_name})")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"correlation_heatmap_{period_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_dendrogram(linkage_matrix, assets, period_name):
    """Save the hierarchical clustering dendrogram for a period."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    dendrogram(
        linkage_matrix,
        labels=assets,
        leaf_rotation=0,
        leaf_font_size=11,
    )

    plt.title(f"Hierarchical Clustering Dendrogram ({period_name})")
    plt.xlabel("Assets")
    plt.ylabel("Secondary Distance")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"dendrogram_{period_name}.png",
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

    # Portfolio concentration diagnostics
    maximum_weight = weights.max()
    hhi = (weights ** 2).sum()
    effective_holdings = 1 / hhi

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

    print("\n HRP portfolio weights:")
    print((weights * 100).round(2).astype(str) + "%")

    print("\n HRP portfolio diagnostics:")
    print("\nPortfolio concentration diagnostics:")
    print(f"Maximum allocation: {maximum_weight:.2%}")
    print(f"HHI: {hhi:.4f}")
    print(f"Effective holdings: {effective_holdings:.2f}")
    print(f"\nTotal weight: {weights.sum():.6f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
