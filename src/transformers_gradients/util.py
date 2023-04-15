from __future__ import annotations

import platform
from functools import wraps
from typing import TypeVar, Callable, Dict, List, Tuple

import tensorflow as tf
from transformers import PreTrainedTokenizerBase

T = TypeVar("T")


def encode_inputs(
    tokenizer: PreTrainedTokenizerBase, x_batch: List[str]
) -> Tuple[tf.Tensor, tf.Tensor | None]:
    """Do batch encode, unpack input ids and other forward-pass kwargs."""
    encoded_input = tokenizer(x_batch, padding="longest", return_tensors="tf").data
    return encoded_input.pop("input_ids"), encoded_input.get("attention_mask")


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


def tensor_inputs(func):
    @wraps(func)
    def wrapper(
        model: UserObject,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        **kwargs,
    ):
        x_batch = as_tensor(x_batch)
        y_batch = as_tensor(y_batch)
        attention_mask = value_or_default(
            attention_mask, lambda: tf.ones(bounding_shape(x_batch), dtype=tf.int32)
        )
        attention_mask = as_tensor(attention_mask)
        return func(model, x_batch, y_batch, attention_mask, **kwargs)

    return wrapper
