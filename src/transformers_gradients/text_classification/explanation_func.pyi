from __future__ import annotations

from typing import overload, List

import tensorflow as tf
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

from transformers_gradients.config import (
    IntGradConfig,
    NoiseGradConfig,
    NoiseGradPlusPlusConfig,
    SmoothGradConfing,
)
from transformers_gradients.types import Explanation

@overload
def gradient_norm(
    model: TFPreTrainedModel,
    x_batch: List[str],
    y_batch: tf.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Explanation]: ...
@overload
def gradient_norm(
    model: TFPreTrainedModel, x_batch: tf.Tensor, y_batch: tf.Tensor, **kwargs
) -> tf.Tensor: ...
@overload
def gradient_x_input(
    model: TFPreTrainedModel,
    x_batch: List[str],
    y_batch: tf.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> tf.Tensor: ...
@overload
def gradient_x_input(
    model: TFPreTrainedModel, x_batch: tf.Tensor, y_batch: tf.Tensor, **kwargs
) -> tf.Tensor: ...
@overload
def integrated_gradients(
    model: TFPreTrainedModel,
    x_batch: List[str],
    y_batch: tf.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    config: IntGradConfig | None = None,
) -> List[Explanation]: ...
@overload
def integrated_gradients(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    config: IntGradConfig | None = None,
    **kwargs,
) -> tf.Tensor: ...
@overload
def smooth_grad(
    model: TFPreTrainedModel,
    x_batch: List[str],
    y_batch: tf.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    config: SmoothGradConfing | None = None,
) -> List[Explanation]: ...
@overload
def smooth_grad(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    config: SmoothGradConfing | None = None,
    **kwargs,
) -> tf.Tensor: ...
@overload
def noise_grad(
    model: TFPreTrainedModel,
    x_batch: List[str],
    y_batch: tf.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    config: NoiseGradConfig | None = None,
) -> List[Explanation]: ...
@overload
def noise_grad(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    config: NoiseGradConfig | None = None,
    **kwargs,
) -> tf.Tensor: ...
@overload
def noise_grad_plus_plus(
    model: TFPreTrainedModel,
    x_batch: List[str],
    y_batch: tf.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    config: NoiseGradPlusPlusConfig | None = None,
) -> List[Explanation]: ...
@overload
def noise_grad_plus_plus(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    config: NoiseGradPlusPlusConfig | None = None,
    **kwargs,
) -> List[Explanation]: ...
