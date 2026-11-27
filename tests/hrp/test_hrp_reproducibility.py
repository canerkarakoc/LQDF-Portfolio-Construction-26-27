import numpy as np

from src import hrp


def test_hrp_reproducibility(sample_returns):

    # First model run
    corr1, linkage1, order1 = hrp.calculate_clustering(
        sample_returns
    )

    weights1 = hrp.calculate_hrp_weights(
        sample_returns, order1
    )

    # Second model run using identical data
    corr2, linkage2, order2 = hrp.calculate_clustering(
        sample_returns
    )

    weights2 = hrp.calculate_hrp_weights(
        sample_returns, order2
    )

    # Compare clustering outputs
    assert order1 == order2

    np.testing.assert_allclose(
        linkage1, linkage2, atol=1e-10
    )

    # Compare final portfolio weights
    np.testing.assert_allclose(
        weights1.values,
        weights2.values,
        atol=1e-10
    )