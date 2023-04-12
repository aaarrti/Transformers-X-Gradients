from typing import overload, List, Optional

import tensorflow as tf
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

from transformers_gradients.types import (
    Explanation,
    IntGradConfig,
    NoiseGradConfig,
    NoiseGradPlusPlusConfig,
    SmoothGradConfing,
)

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
def gradient_x_input(
    model: TFPreTrainedModel,
    x_batch: List[str],
    y_batch: tf.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Explanation]: ...
@overload
def gradient_x_input(
    model: TFPreTrainedModel, x_batch: tf.Tensor, y_batch: tf.Tensor, **kwargs
) -> tf.Tensor: ...
def integrated_gradients(
    model: TFPreTrainedModel,
    x_batch: List[str],
    y_batch: tf.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    config: Optional[IntGradConfig] = None,
) -> List[Explanation]: ...
@overload
def integrated_gradients(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    config: Optional[IntGradConfig] = None,
    **kwargs,
) -> tf.Tensor: ...
def smooth_grad(
    model: TFPreTrainedModel,
    x_batch: List[str],
    y_batch: tf.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    config: Optional[SmoothGradConfing] = None,
) -> List[Explanation]: ...
@overload
def smooth_grad(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    config: Optional[SmoothGradConfing] = None,
    **kwargs,
) -> tf.Tensor: ...
def noise_grad(
    model: TFPreTrainedModel,
    x_batch: List[str],
    y_batch: tf.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    config: Optional[NoiseGradConfig] = None,
) -> List[Explanation]: ...
@overload
def noise_grad(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    config: Optional[NoiseGradConfig] = None,
    **kwargs,
) -> tf.Tensor: ...
def noise_grad_plus_plus(
    model: TFPreTrainedModel,
    x_batch: List[str],
    y_batch: tf.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    config: Optional[NoiseGradPlusPlusConfig] = None,
) -> List[Explanation]: ...
@overload
def noise_grad_plus_plus(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    config: Optional[NoiseGradPlusPlusConfig] = None,
    **kwargs,
) -> List[Explanation]: ...
