import numpy as np

from src import hrp


def test_weights_sum_to_one(sample_returns):
    # Calculate clustering and asset ordering
    correlation, linkage_matrix, ordered_assets = (
        hrp.calculate_clustering(sample_returns)
    )

    # Calculate HRP weights
    weights = hrp.calculate_hrp_weights(
        sample_returns, ordered_assets
    )

    # Check that total allocation equals 100%
    assert np.isclose(weights.sum(), 1.0, atol=1e-9)

def test_weights_are_non_negative(sample_returns):
    _, _, ordered_assets = hrp.calculate_clustering(sample_returns)
    
    # Check weights non negative
    weights = hrp.calculate_hrp_weights(
        sample_returns, ordered_assets
    )

    assert (weights >= 0).all()

# Check no infinite values
def test_weights_are_finite(sample_returns):
    _, _, ordered_assets = hrp.calculate_clustering(sample_returns)

    weights = hrp.calculate_hrp_weights(
        sample_returns, ordered_assets
    )

    assert np.isfinite(weights).all()


def test_calculate_weights_returns_valid_weights_in_input_order(
    sample_returns,
):
    returns = sample_returns[sample_returns.columns[::-1]]

    weights = hrp.calculate_weights(returns)

    assert weights.index.tolist() == returns.columns.tolist()
    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    assert np.isclose(weights.sum(), 1.0, atol=1e-9)