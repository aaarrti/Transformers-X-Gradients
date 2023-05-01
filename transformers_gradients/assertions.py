import tensorflow as tf


def assert_numerics(arr):
    try:
        tf.debugging.check_numerics(arr, message="NaN are not allowed.")
    except Exception as e:
        assert "Tensor had NaN values" in e.message
