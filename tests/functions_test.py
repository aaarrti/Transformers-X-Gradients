import pytest
import numpy as np
from sklearn import linear_model

from transformers_gradients.functions import ridge_regression


@pytest.fixture(scope="session")
def ridge_inputs():
    return (
        (np.random.default_rng(42).random(size=(6, 29)) >= 0.5),
        np.random.default_rng(42).random(size=6, dtype=float),
        np.random.default_rng(42).random(size=6, dtype=float),
    )


@pytest.fixture
def ridge_expected(ridge_inputs):
    model = linear_model.Ridge(alpha=1.0, solver="cholesky", random_state=42)
    model = model.fit(ridge_inputs[0], ridge_inputs[1], sample_weight=ridge_inputs[2])
    return model.coef_  # noqa


def test_ridge_regression(ridge_inputs, ridge_expected):
    result = ridge_regression(
        ridge_inputs[0], ridge_inputs[1], sample_weight=ridge_inputs[2]
    )
    assert result.shape == ridge_expected.shape
    assert not np.isnan(result).any()
    assert np.allclose(result, ridge_expected, atol=0.01)
