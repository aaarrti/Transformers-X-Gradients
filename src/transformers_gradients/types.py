from __future__ import annotations

from typing import Callable, Protocol, overload, runtime_checkable, Tuple, List

import tensorflow as tf
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

BaselineFn = Callable[[tf.Tensor], tf.Tensor]
Explanation = Tuple[List[str], tf.Tensor]
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
        x_batch: List[str],
        y_batch: tf.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        *args,
        **kwargs,
    ) -> List[Explanation]:
        ...

    def __call__(  # type: ignore
        self,
        model: TFPreTrainedModel,
        x_batch: List[str] | tf.Tensor,
        y_batch: tf.Tensor,
        tokenizer: PreTrainedTokenizerBase | None,
        *args,
        **kwargs,
    ) -> List[Explanation] | tf.Tensor:
        ...
