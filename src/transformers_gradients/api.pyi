from __future__ import annotations
from typing import overload, List
import tensorflow as tf
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase
from transformers_gradients.types import (
    Explanation,
    IntGradConfig,
    SmoothGradConfing,
    NoiseGradPlusPlusConfig,
    NoiseGradConfig,
    LimeConfig,
)

class text_classification(object):
    # ----------------------------------------------------------------------------
    @staticmethod
    def gradient_norm(
        model: TFPreTrainedModel, x_batch, y_batch: tf.Tensor, **kwargs
    ): ...
    @staticmethod
    @overload
    def gradient_norm(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
    ) -> tf.Tensor: ...
    @staticmethod
    @overload
    def gradient_norm(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
    ) -> List[Explanation]: ...

    # ----------------------------------------------------------------------------

    @staticmethod
    def gradient_x_input(
        model: TFPreTrainedModel, x_batch, y_batch: tf.Tensor, **kwargs
    ): ...
    @staticmethod
    @overload
    def gradient_x_input(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
    ) -> tf.Tensor: ...
    @staticmethod
    @overload
    def gradient_x_input(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
    ) -> List[Explanation]: ...

    # ----------------------------------------------------------------------------

    @staticmethod
    def integrated_gradients(
        model: TFPreTrainedModel,
        x_batch,
        y_batch: tf.Tensor,
        *,
        config: IntGradConfig | None = None,
        **kwargs,
    ): ...
    @staticmethod
    @overload
    def integrated_gradients(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
        config: IntGradConfig | None = None,
    ) -> tf.Tensor: ...
    @staticmethod
    @overload
    def integrated_gradients(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
        config: IntGradConfig | None = None,
    ) -> List[Explanation]: ...

    # ----------------------------------------------------------------------------

    @staticmethod
    def smooth_grad(
        model: TFPreTrainedModel,
        x_batch,
        y_batch: tf.Tensor,
        *,
        config: SmoothGradConfing | None = None,
        **kwargs,
    ): ...
    @staticmethod
    @overload
    def smooth_grad(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
        config: SmoothGradConfing | None = None,
    ) -> tf.Tensor: ...
    @staticmethod
    @overload
    def smooth_grad(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
        config: SmoothGradConfing | None = None,
    ) -> List[Explanation]: ...

    # ----------------------------------------------------------------------------

    @staticmethod
    def noise_grad(
        model: TFPreTrainedModel,
        x_batch,
        y_batch: tf.Tensor,
        *,
        config: NoiseGradConfig | None = None,
        **kwargs,
    ) -> tf.Tensor: ...
    @staticmethod
    @overload
    def noise_grad(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
        config: NoiseGradConfig | None = None,
    ) -> tf.Tensor: ...
    @staticmethod
    @overload
    def noise_grad(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
        config: NoiseGradConfig | None = None,
    ) -> List[Explanation]: ...

    # ----------------------------------------------------------------------------

    @staticmethod
    def noise_grad_plus_plus(
        model: TFPreTrainedModel,
        x_batch,
        y_batch: tf.Tensor,
        *,
        config: NoiseGradPlusPlusConfig | None = None,
        **kwargs,
    ): ...
    @staticmethod
    @overload
    def noise_grad_plus_plus(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
        config: NoiseGradPlusPlusConfig | None = None,
    ) -> tf.Tensor: ...
    @staticmethod
    @overload
    def noise_grad_plus_plus(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
        config: NoiseGradPlusPlusConfig | None = None,
    ) -> List[Explanation]: ...

    # ----------------------------------------------------------------------------

    @staticmethod
    def lime(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        config: LimeConfig | None = None,
    ) -> List[Explanation]: ...
