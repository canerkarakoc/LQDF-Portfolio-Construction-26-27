
# --------------------------------------------------
# HRP Historical Period Analysis
# --------------------------------------------------

import pandas as pd
from src import hrp
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Define historical periods
# --------------------------------------------------

PERIODS = {
    "2015-2018": ("2015-01-01", "2018-12-31"),
    "2019-2022": ("2019-01-01", "2022-12-31"),
    "2023-2026": ("2023-01-01", "2026-10-30"),
}


# --------------------------------------------------
# 2. Main analysis
# --------------------------------------------------

def main():

    # Load historical asset returns
    returns = hrp.load_returns()

    # Store results from each period
    weights_results = {}
    diagnostics_results = {}
    clustering_results = {}

    # --------------------------------------------------
    # 3. Calculate HRP weights for each period
    # --------------------------------------------------

    for period, (start, end) in PERIODS.items():

        print(f"\nAnalysing period: {period}")

        # Select returns for this historical period
        period_returns = returns.loc[start:end]

        if len(period_returns) < 2:
            raise ValueError(
                f"Not enough observations for {period}"
            )

        # Calculate clustering
        correlation, linkage_matrix, ordered_assets = (
            hrp.calculate_clustering(period_returns)
        )
        # Save this period's correlation heatmap and dendrogram
        hrp.save_heatmap(
            correlation,
            ordered_assets,
            period,
        )

        hrp.save_dendrogram(
            linkage_matrix,
            period_returns.columns.tolist(),
            period,
        )

        # Calculate HRP portfolio weights
        weights = hrp.calculate_hrp_weights(
            period_returns, ordered_assets
        )

        # Store portfolio weights
        weights_results[period] = weights

        # Calculate concentration diagnostics
        maximum_weight = weights.max()
        hhi = (weights ** 2).sum()
        effective_holdings = 1 / hhi

        diagnostics_results[period] = {
            "Maximum allocation": maximum_weight,
            "HHI": hhi,
            "Effective holdings": effective_holdings,
            "Total weight": weights.sum(),
            "Observations": len(period_returns),
        }

        # Store clustering order
        clustering_results[period] = ", ".join(
            ordered_assets
        )
        
        # Display results
        print("\nHRP portfolio weights:")
        print((weights * 100).round(2).astype(str) + "%")

        print(f"\nMaximum allocation: {maximum_weight:.2%}")
        print(f"HHI: {hhi:.4f}")
        print(f"Effective holdings: {effective_holdings:.2f}")
        print(f"Observations: {len(period_returns)}")

    # --------------------------------------------------
    # 4. Create comparison tables
    # --------------------------------------------------

    weights_comparison = pd.DataFrame(weights_results)

    diagnostics_comparison = pd.DataFrame(
        diagnostics_results
    ).T

    clustering_comparison = pd.DataFrame.from_dict(
        clustering_results,
        orient="index",
        columns=["Clustering order"]
    )

    # --------------------------------------------------
    # 5. Save results
    # --------------------------------------------------

    output_dir = hrp.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_comparison.to_csv(
        output_dir / "historical_weights_comparison.csv"
    )

    diagnostics_comparison.to_csv(
        output_dir / "historical_diagnostics.csv"
    )

    clustering_comparison.to_csv(
        output_dir / "historical_clustering_comparison.csv"
    )
    ax = (weights_comparison * 100).plot.bar(
        figsize=(9, 5), rot=0, width=0.8
    )
    ax.set_title("HRP portfolio weights by historical period")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Portfolio weight (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="Historical period")

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_dir / "historical_weights_comparison.png", dpi=300)
    plt.close(fig)
    # --------------------------------------------------
    # 6. Display comparisons
    # --------------------------------------------------

    print("\n======================================")
    print("HISTORICAL HRP WEIGHT COMPARISON")
    print("======================================")

    print((weights_comparison * 100).round(2))

    print("\nHISTORICAL CONCENTRATION DIAGNOSTICS")
    print(diagnostics_comparison.round(4))

    print("\nHISTORICAL CLUSTERING ORDER")
    print(clustering_comparison)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
