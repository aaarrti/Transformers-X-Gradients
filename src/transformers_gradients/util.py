from __future__ import annotations
from typing import Optional, TypeVar, Callable, List, Tuple, Dict, TYPE_CHECKING
import platform

import tensorflow as tf
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from transformers_gradients.types import EncodedBatch, TokenizerI

T = TypeVar("T")


def get_input_ids(
    tokenizer: TokenizerI, x_batch: list[str]
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Do batch encode, unpack input ids and other forward-pass kwargs."""
    encoded_input = tokenizer.batch_encode(x_batch)
    encoded_input = dict(encoded_input)
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
    if not isinstance(arr, tf.Tensor):
        return arr
    else:
        return tf.convert_to_tensor(arr)
