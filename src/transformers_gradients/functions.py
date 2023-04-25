import tensorflow as tf

from transformers_gradients.utils import is_xla_compatible_platform


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def multiplicative_noise(arr: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("multiplicative_noise"):
        return tf.multiply(arr, noise)


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def additive_noise(arr: tf.Tensor, noise: tf.Tensor):
    with tf.name_scope("additive_noise"):
        return tf.add(arr, noise)


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def logits_for_labels(logits: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
    # Matrix with indexes like [ [0,y_0], [1, y_1], ...]
    with tf.name_scope("logits_for_labels"):
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


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def default_attention_mask(x_batch: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("default_attention_mask"):
        return tf.ones(
            tf.gather(tf.shape(x_batch), tf.constant([0, 1])), dtype=tf.int32
        )


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def zeros_baseline(arr: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("zeros_baseline"):
        return tf.zeros_like(arr)


# @tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def sample_masks(num_samples: int, num_features: int, seed: int = 42):
    with tf.name_scope("sample_masks"):
        positions = tf.tile(
            tf.expand_dims(tf.range(num_features, dtype=tf.int32), 0), (num_samples, 1)
        )
        permutations = tf.vectorized_map(tf.random.shuffle, positions)
        num_disabled_features = tf.random.uniform(
            minval=1,
            maxval=num_features + 1,
            shape=tf.shape(positions),
            seed=seed,
            dtype=tf.int32,
        )
        return tf.math.greater_equal(permutations, num_disabled_features)


# @tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def mask_tokens(
    token_ids: tf.Tensor, masks: tf.Tensor, mask_token_id: tf.Tensor
) -> tf.Tensor:
    with tf.name_scope("mask_tokens"):
        ids_batch = tf.repeat(tf.expand_dims(token_ids, 0), tf.shape(masks)[0], axis=0)
        masks = tf.cast(masks, tf.int32)
        return (ids_batch * (tf.ones_like(masks) - masks)) + (masks * mask_token_id)


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def weighted_average(x: tf.Tensor, weights: tf.Tensor, axis=None) -> tf.Tensor:
    with tf.name_scope("weighted_average"):
        return tf.reduce_sum(weights * x, axis=axis) / tf.reduce_sum(weights, axis=axis)


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def ridge_regression(
    X: tf.Tensor,
    y: tf.Tensor,
    sample_weight: tf.Tensor,
    alpha: tf.Tensor = tf.constant(1.0),
) -> tf.Tensor:
    # Preprocess data
    with tf.name_scope("ridge_regression"):
        X = tf.cast(X, dtype=tf.float64)
        y = tf.cast(y, dtype=tf.float64)
        sample_weight = tf.cast(sample_weight, dtype=tf.float64)
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
