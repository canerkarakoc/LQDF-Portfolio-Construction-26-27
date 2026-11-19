import numpy as np

from src import hrp


def test_no_unexpected_allocations(sample_returns):

    _, _, ordered_assets = hrp.calculate_clustering(
        sample_returns
    )

    weights = hrp.calculate_hrp_weights(
        sample_returns, ordered_assets
    )

    # Every asset should receive a valid weight
    assert np.isfinite(weights).all()

    assert (weights >= 0).all()

    assert (weights <= 1).all()

    assert (weights > 0).all()

    # Calculate concentration
    hhi = (weights ** 2).sum()

    effective_holdings = 1 / hhi

    maximum_weight = weights.max()

    print("\nAllocation analysis:")
    print("Maximum asset weight:", maximum_weight)
    print("HHI:", hhi)
    print("Effective holdings:", effective_holdings)

    # Effective holdings must be between 1 and 5
    assert 1 - 1e-9 <= effective_holdings <= 5 + 1e-9