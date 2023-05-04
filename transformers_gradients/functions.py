import numpy as np
import tensorflow as tf
from typing import TypeVar
from transformers_gradients.utils import is_xla_compatible_platform

T = TypeVar("T")


@tf.function(
    reduce_retracing=True,
    jit_compile=is_xla_compatible_platform(),
    experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
)
def logits_for_labels(logits: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
    # Matrix with indexes like [ [0,y_0], [1, y_1], ...]
    with tf.name_scope("logits_for_labels"):
        logits = tf.cast(logits, tf.float16)
        indexes = tf.transpose(
            tf.stack(
                [
                    tf.range(tf.shape(logits)[0], dtype=tf.int32),
                    tf.cast(y_batch, tf.int32),
                ]
            ),
            [1, 0],
        )
        return tf.gather_nd(logits, indexes)


@tf.function(
    reduce_retracing=True,
    jit_compile=is_xla_compatible_platform(),
    experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
)
def default_attention_mask(x_batch: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("default_attention_mask"):
        return tf.ones(
            tf.gather(tf.shape(x_batch), tf.constant([0, 1])), dtype=tf.int32
        )


@tf.function(
    reduce_retracing=True,
    jit_compile=is_xla_compatible_platform(),
    experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
)
def weighted_average(x: tf.Tensor, weights: tf.Tensor, axis=None) -> tf.Tensor:
    with tf.name_scope("weighted_average"):
        return tf.reduce_sum(weights * x, axis=axis) / tf.reduce_sum(weights, axis=axis)


@tf.function(
    reduce_retracing=True,
    jit_compile=is_xla_compatible_platform(),
    experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
)
def ridge_regression(
    X: tf.Tensor,
    y: tf.Tensor,
    sample_weight: tf.Tensor,
    alpha: tf.Tensor = tf.constant(1.0),
) -> tf.Tensor:
    # Preprocess data
    with tf.name_scope("ridge_regression"):
        y = tf.cast(y, dtype=X.dtype)
        sample_weight = tf.cast(sample_weight, dtype=X.dtype)
        X_offset = weighted_average(
            X, axis=0, weights=tf.expand_dims(sample_weight, axis=1)
        )
        X -= X_offset
        y_offset = weighted_average(y, axis=0, weights=sample_weight)
        y = y - y_offset
        y = tf.expand_dims(y, axis=1)
        # Rescale data
        sample_weight_sqrt = tf.sqrt(sample_weight)
        sw_matrix = tf.linalg.diag(sample_weight_sqrt)
        X = tf.matmul(sw_matrix, X, a_is_sparse=True)
        y = tf.matmul(sw_matrix, y, a_is_sparse=True)
        # Create kernel
        K = tf.matmul(X, X, transpose_b=True)
        # Apply penalty
        penalty = tf.cast(
            tf.linalg.diag(tf.repeat(alpha, tf.shape(K)[0])), dtype=K.dtype
        )
        K = K + penalty
        # Solve
        dual_coef = tf.linalg.solve(K, y)
        coef = tf.transpose(tf.matmul(X, dual_coef, transpose_a=True), [1, 0])
        return coef[0]


def normalize_sum_to_1(scores: T) -> T:
    if isinstance(scores, (tf.Tensor, np.ndarray)):
        return _normalize_sum_to_1(scores)

    from transformers_gradients.lib_types import Explanation

    return Explanation(
        tokens=scores.tokens,
        scores=_normalize_sum_to_1(tf.expand_dims(scores.scores, 0))[0],
    )


@tf.function(
    reduce_retracing=True,
    jit_compile=is_xla_compatible_platform(),
    experimental_autograph_options=tf.autograph.experimental.Feature.ALL,
)
def _normalize_sum_to_1(scores: tf.Tensor) -> tf.Tensor:
    """Makes the absolute values sum to 1."""
    og_dtype = scores.dtype
    # float 16 will cause overflow
    scores = tf.cast(scores, tf.float32)
    scores = scores + tf.keras.backend.epsilon()
    scores = tf.transpose(
        tf.transpose(scores, [1, 0]) / tf.reduce_sum(tf.abs(scores), -1), [1, 0]
    )
    return tf.cast(scores, og_dtype)
