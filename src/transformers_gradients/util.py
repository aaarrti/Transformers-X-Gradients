from __future__ import annotations

import platform
from typing import TypeVar, Callable, Dict, NamedTuple

import tensorflow as tf
from transformers import PreTrainedTokenizerBase

T = TypeVar("T")
R = TypeVar("R")


def get_input_ids(
    tokenizer: PreTrainedTokenizerBase, x_batch: list[str]
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
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


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def as_tensor(arr) -> tf.Tensor:
    if isinstance(arr, (tf.Tensor, NamedTuple, Callable)):
        return arr
    else:
        return tf.convert_to_tensor(arr)


def map_dict(
    dictionary: Dict[str, T],
    value_mapper: Callable[[T], R],
    key_mapper: Callable[[str], str] = lambda x: x,
) -> Dict[str, R]:
    """Applies func to values in dict. Additionally, if provided can also map keys."""
    result = {}
    for k, v in dictionary.items():
        result[key_mapper(k)] = value_mapper(v)
    return result
