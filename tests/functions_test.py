import pytest
import tensorflow as tf
from safetensors.tensorflow import load_file

from transformers_gradients import normalize_sum_to_1
from transformers_gradients.assertions import assert_numerics
from transformers_gradients.functions import ridge_regression


@pytest.fixture
def explanation_tensor():
    return load_file("tests/data/explanations_tensor.safetensors")["a_batch"]


@pytest.fixture(scope="session")
def ridge_inputs():
    return load_file("tests/data/ridge_input.safetensors")


@pytest.fixture
def ridge_expected(ridge_inputs):
    return load_file("tests/data/ridge_expected.safetensors")["expected"]


def test_ridge_regression(ridge_inputs, ridge_expected):
    result = ridge_regression(
        ridge_inputs["X"], ridge_inputs["Y"], ridge_inputs["sample_weight"]
    )
    assert result.shape == ridge_expected.shape
    assert_numerics(result)
    tf.debugging.assert_near(result, ridge_expected, atol=0.01)


def test_normalise_scores(explanation_tensor):
    result = normalize_sum_to_1(tf.cast(explanation_tensor, tf.float16))
    assert_numerics(result)
    ex_sum = tf.reduce_sum(result, axis=1)
    tf.debugging.assert_near(ex_sum, 1.0)
