import numpy as np

from src import hrp


def test_correlation_matrix_validity(sample_returns):

    correlation, _, _ = hrp.calculate_clustering(
        sample_returns
    )

    # Check for missing or infinite values
    assert np.isfinite(correlation.to_numpy()).all()

    # Check symmetry
    np.testing.assert_allclose(
        correlation,
        correlation.T
    )

    # Check diagonal equals one
    np.testing.assert_allclose(
        np.diag(correlation), 1.0
    )

    # Check correlations are between -1 and 1
    assert (correlation.to_numpy() >= -1 - 1e-10).all()
    assert (correlation.to_numpy() <= 1 + 1e-10).all()


def test_linkage_matrix_validity(sample_returns):

    _, linkage_matrix, _ = hrp.calculate_clustering(
        sample_returns
    )

    n_assets = len(sample_returns.columns)

    # There should be N-1 merging operations
    assert linkage_matrix.shape == (n_assets - 1, 4)

    # Linkage distances must be finite and non-negative
    assert np.isfinite(linkage_matrix).all()
    assert (linkage_matrix[:, 2] >= 0).all()


def test_asset_ordering(sample_returns):

    _, _, ordered_assets = hrp.calculate_clustering(
        sample_returns
    )

    # Every asset must appear exactly once
    assert len(ordered_assets) == len(sample_returns.columns)

    assert set(ordered_assets) == set(sample_returns.columns)

    assert len(set(ordered_assets)) == len(ordered_assets)