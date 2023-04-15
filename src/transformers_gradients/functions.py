import tensorflow as tf

from transformers_gradients.types import BaselineFn
from transformers_gradients.util import is_xla_compatible_platform


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def pseudo_interpolate(x: tf.Tensor, num_steps: tf.Tensor) -> tf.Tensor:
    og_shape = tf.convert_to_tensor(tf.shape(x))
    new_shape = tf.concat([[num_steps + tf.constant(1)], og_shape], axis=0)
    x = tf.broadcast_to(x, new_shape)
    flat_shape = tf.concat([tf.constant([-1]), og_shape[1:]], axis=0)
    x = tf.reshape(x, flat_shape)
    return x


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def multiplicative_noise(arr: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
    return tf.multiply(arr, noise)


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def additive_noise(arr: tf.Tensor, noise: tf.Tensor):
    return tf.add(arr, noise)


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def logits_for_labels(logits: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
    # Matrix with indexes like [ [0,y_0], [1, y_1], ...]
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
def bounding_shape(arr: tf.Tensor) -> tf.Tensor:
    return tf.constant([tf.shape(arr)[0], tf.shape(arr)[1]])


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def zeros_baseline(arr: tf.Tensor) -> tf.Tensor:
    return tf.zeros_like(arr)


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def _interpolate_inputs(
    baseline: tf.Tensor, target: tf.Tensor, num_steps: int
) -> tf.Tensor:
    """Gets num_step linearly interpolated inputs from baseline to target."""
    delta = target - baseline
    scales = tf.linspace(0, 1, num_steps + 1)[:, tf.newaxis, tf.newaxis]
    scales = tf.cast(scales, dtype=delta.dtype)
    shape = tf.convert_to_tensor(
        [num_steps + 1, tf.shape(delta)[0], tf.shape(delta)[1]]
    )
    deltas = scales * tf.broadcast_to(delta, shape)
    interpolated_inputs = baseline + deltas
    return interpolated_inputs


def interpolate_inputs(
    x_batch: tf.Tensor, num_steps: int, baseline_fn: BaselineFn
) -> tf.Tensor:
    return tf.vectorized_map(
        lambda i: _interpolate_inputs(baseline_fn(i), i, tf.constant(num_steps)),
        x_batch,
    )


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def broadcast_expand_dims(x: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    return tf.broadcast_to(x, tf.stack([tf.shape(target)[0], tf.shape(x)[0]]))


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def exponential_kernel(distance: tf.Tensor, kernel_width: tf.Tensor = 25) -> tf.Tensor:
    return tf.sqrt(
        tf.exp(-(distance ** tf.constant(2.0)) / kernel_width ** tf.constant(2.0))
    )
