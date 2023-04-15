from __future__ import annotations

import platform
from typing import TypeVar, Callable, Dict, List, Tuple

import tensorflow as tf
from transformers import PreTrainedTokenizerBase

T = TypeVar("T")


def get_input_ids(
    tokenizer: PreTrainedTokenizerBase, x_batch: List[str]
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """Do batch encode, unpack input ids and other forward-pass kwargs."""
    encoded_input = tokenizer(x_batch, padding="longest", return_tensors="tf").data
    return encoded_input.pop("input_ids"), encoded_input


def value_or_default(value: T | None, default_factory: Callable[[], T]) -> T:
    if value is not None:
        return value
    else:
        return default_factory()


def is_xla_compatible_platform() -> bool:
    """Determine if host is xla-compatible."""
    return not (platform.system() == "Darwin" and "arm" in platform.processor().lower())


def as_tensor(arr) -> tf.Tensor:
    if isinstance(arr, (tf.Tensor, Callable)):  # type: ignore
        return arr
    else:
        return tf.convert_to_tensor(arr)


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
def bounding_shape(arr):
    return tf.constant([tf.shape(arr)[0], tf.shape(arr)[1]])


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def zeros_baseline(arr):
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


def interpolate_inputs(x_batch, num_steps, baseline_fn):
    return tf.map_fn(
        lambda i: _interpolate_inputs(baseline_fn(i), i, tf.constant(num_steps)),
        x_batch,
    )


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def multiplicative_noise(arr, noise):
    return tf.multiply(arr, noise)
