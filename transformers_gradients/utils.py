from __future__ import annotations

import platform
from types import ModuleType
from typing import TypeVar, Callable, List, Tuple, Mapping

import tensorflow as tf
from transformers import PreTrainedTokenizerBase

from transformers_gradients.lib_types import ExplainFn, ApplyNoiseFn

T = TypeVar("T")
R = TypeVar("R")


def encode_inputs(
    tokenizer: PreTrainedTokenizerBase, x_batch: List[str], **kwargs
) -> Tuple[tf.Tensor, tf.Tensor | None]:
    """Do batch encode, unpack input ids and other forward-pass kwargs."""
    encoded_input = tokenizer(
        x_batch, padding="longest", return_tensors="tf", **kwargs
    ).data
    return encoded_input.pop("input_ids"), encoded_input.get("attention_mask")


def value_or_default(value: T | None, default_factory: Callable[[], T]) -> T:
    if value is not None:
        return value
    else:
        return default_factory()


def map_optional(val: T | None, func: Callable[[T], R]) -> R | None:
    """Apply func to value if not None, otherwise return None."""
    if val is None:
        return None
    return func(val)


def is_xla_compatible_platform() -> bool:
    """Determine if host is xla-compatible."""
    return not (platform.system() == "Darwin" and "arm" in platform.processor().lower())


def as_tensor(arr) -> tf.Tensor:
    if isinstance(arr, (tf.Tensor, Callable)):  # type: ignore
        return arr
    else:
        return tf.convert_to_tensor(arr)


def resolve_baseline_explain_fn(
    module: ModuleType, explain_fn: str | ExplainFn
) -> ExplainFn:
    if isinstance(explain_fn, Callable):  # type: ignore
        return explain_fn  # type: ignore

    method_mapping = {
        "IntGrad": module.integrated_gradients,
        "GradNorm": module.gradient_norm,
        "GradXInput": module.gradient_x_input,
    }
    if explain_fn not in method_mapping:
        raise ValueError(
            f"Unknown XAI method {explain_fn}, supported are {list(method_mapping.keys())}"
        )
    return method_mapping[explain_fn]  # type: ignore


def resolve_noise_fn(
    noise_fn: ApplyNoiseFn | Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    if isinstance(noise_fn, Callable):
        return noise_fn

    if noise_fn == "multiplicative":
        return tf.multiply
    if noise_fn == "additive":
        return tf.add

    raise ValueError(
        f"Unknown noise_fn: {noise_fn}, supported are additive, multiplicative"
    )


def mapping_to_config(config, cls):
    if isinstance(config, Mapping):
        return cls(**config)
    else:
        return config


def is_mixed_precision_supported_device():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        cc = gpu_details.get("compute_capability")
        if cc:
            return cc >= (7, 0)
    return False
