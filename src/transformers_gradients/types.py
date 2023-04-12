from __future__ import annotations
from typing import (
    NamedTuple,
    Callable,
    Union,
    Protocol,
    Optional,
    overload,
    runtime_checkable,
    Literal,
)
import tensorflow as tf

from transformers import TFPreTrainedModel, PreTrainedTokenizerBase
from transformers_gradients.util import value_or_default, is_xla_compatible_platform

BaselineFn = Callable[[tf.Tensor], tf.Tensor]
Explanation = tuple[list[str], tf.Tensor]
ApplyNoiseFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


@runtime_checkable
class ExplainFn(Protocol):
    @overload
    def __call__(
        self,
        model: ModelI,
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
        model: ModelI,
        x_batch: list[str],
        y_batch: tf.Tensor,
        tokenizer: TokenizerI,
        *args,
        **kwargs,
    ) -> list[Explanation]:
        ...

    def __call__(
        self,
        model: ModelI,
        x_batch: list[str] | tf.Tensor,
        y_batch: tf.Tensor,
        tokenizer: TokenizerI | None,
        *args,
        **kwargs,
    ) -> list[Explanation] | tf.Tensor:
        ...


@runtime_checkable
class ModelI(Protocol):
    model: tf.keras.Model

    def __call__(self, inputs_embeds: tf.Tensor | None, **kwargs) -> tf.Tensor:
        ...

    def embedding_lookup(self, input_ids: tf.Tensor) -> EncodedBatch:
        ...


@runtime_checkable
class TokenizerI(Protocol):
    def convert_ids_to_tokens(self, input_ids: list[int]) -> list[str]:
        ...

    def batch_encode(self, x_batch: list[str]) -> dict[str, tf.Tensor]:
        ...
