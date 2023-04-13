from __future__ import annotations

import platform
from typing import TypeVar, Callable, Dict, List, Tuple

import tensorflow as tf
from cachetools import cached
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


@cached(key=lambda f: f.__name__, cache={})
def cached_tf_function(func):
    return tf.function(
        func, reduce_retracing=True, jit_compile=is_xla_compatible_platform()
    )
