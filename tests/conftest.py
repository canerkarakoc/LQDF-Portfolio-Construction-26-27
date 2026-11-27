import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_returns():

    # Reproducible random number generator
    rng = np.random.default_rng(42)

    assets = ["VTI", "EFA", "AGG", "GLD", "VNQ"]

    # Generate 300 days of synthetic asset returns
    returns = rng.normal(
        loc=0,
        scale=[0.012, 0.014, 0.004, 0.010, 0.016],
        size=(300, 5)
    )

    returns = pd.DataFrame(
        returns,
        columns=assets,
        index=pd.date_range(
            start="2020-01-01",
            periods=300,
            freq="B"
        )
    )

    return returns