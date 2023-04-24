from __future__ import annotations
from typing import overload, List
import tensorflow as tf

from transformers_gradients.types import (
    Explanation,
    IntGradConfig,
    NoiseGradConfig,
    SmoothGradConfing,
    NoiseGradPlusPlusConfig,
)
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase


class text_classification(object):
    # ----------------------------------------------------------------------------
    @staticmethod
    @overload
    def gradient_norm(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
    ) -> tf.Tensor:
        ...

    @staticmethod
    @overload
    def gradient_norm(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
    ) -> List[Explanation]:
        ...

    @staticmethod
    def gradient_norm(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import gradient_norm

        return gradient_norm(*args, **kwargs)

    # ----------------------------------------------------------------------------
    @staticmethod
    @overload
    def gradient_x_input(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
    ) -> tf.Tensor:
        ...

    @staticmethod
    @overload
    def gradient_x_input(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
    ) -> List[Explanation]:
        ...

    @staticmethod
    def gradient_x_input(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import gradient_x_input

        return gradient_x_input(*args, **kwargs)

    # ----------------------------------------------------------------------------
    @staticmethod
    @overload
    def integrated_gradients(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
        config: IntGradConfig | None = None,
    ) -> tf.Tensor:
        ...

    @staticmethod
    @overload
    def integrated_gradients(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
        config: IntGradConfig | None = None,
    ) -> List[Explanation]:
        ...

    @staticmethod
    def integrated_gradients(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import (
            integrated_gradients,
        )

        return integrated_gradients(*args, **kwargs)

    # ----------------------------------------------------------------------------
    @staticmethod
    @overload
    def smooth_grad(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
        config: SmoothGradConfing | None = None,
    ) -> tf.Tensor:
        ...

    @staticmethod
    @overload
    def smooth_grad(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
        config: SmoothGradConfing | None = None,
    ) -> List[Explanation]:
        ...

    @staticmethod
    def smooth_grad(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import smooth_grad

        return smooth_grad(*args, **kwargs)

    # ----------------------------------------------------------------------------
    @staticmethod
    @overload
    def noise_grad(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
        config: NoiseGradConfig | None = None,
    ) -> tf.Tensor:
        ...

    @staticmethod
    @overload
    def noise_grad(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
        config: NoiseGradConfig | None = None,
    ) -> List[Explanation]:
        ...

    @staticmethod
    def noise_grad(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import smooth_grad

        return smooth_grad(*args, **kwargs)

    # ----------------------------------------------------------------------------
    @staticmethod
    @overload
    def noise_grad_plus_plus(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
        config: NoiseGradPlusPlusConfig | None = None,
    ) -> tf.Tensor:
        ...

    @staticmethod
    @overload
    def noise_grad_plus_plus(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
        config: NoiseGradPlusPlusConfig | None = None,
    ) -> List[Explanation]:
        ...

    @staticmethod
    def noise_grad_plus_plus(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import (
            noise_grad_plus_plus,
        )

        return noise_grad_plus_plus(*args, **kwargs)

    # ----------------------------------------------------------------------------
    @staticmethod
    def lime(*args, **kwargs):
        from transformers_gradients.tasks.text_classification import lime

        return lime(*args, **kwargs)
