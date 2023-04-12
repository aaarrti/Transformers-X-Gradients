from __future__ import annotations

from typing import (
    Callable,
    Protocol,
    overload,
    runtime_checkable,
)

import tensorflow as tf
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

BaselineFn = Callable[[tf.Tensor], tf.Tensor]
Explanation = tuple[list[str], tf.Tensor]
ApplyNoiseFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


@runtime_checkable
class ExplainFn(Protocol):
    @overload
    def __call__(
        self,
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        tokenizer: None = None,
        *args,
        **kwargs,
    ) -> tf.Tensor:
        ...

    @overload
    def __call__(
        self,
        model: TFPreTrainedModel,
        x_batch: list[str],
        y_batch: tf.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        *args,
        **kwargs,
    ) -> list[Explanation]:
        ...

    def __call__(
        self,
        model: TFPreTrainedModel,
        x_batch: list[str] | tf.Tensor,
        y_batch: tf.Tensor,
        tokenizer: PreTrainedTokenizerBase | None,
        *args,
        **kwargs,
    ) -> list[Explanation] | tf.Tensor:
        ...
