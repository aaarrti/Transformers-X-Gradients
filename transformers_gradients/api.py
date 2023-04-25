from __future__ import annotations
from typing import overload, List, TYPE_CHECKING
import tensorflow as tf

if TYPE_CHECKING:
    from transformers_gradients.lib_types import (
        Explanation,
        IntGradConfig,
        NoiseGradConfig,
        SmoothGradConfing,
        NoiseGradPlusPlusConfig,
        LimeConfig,
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
    def gradient_norm(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        attention_mask: tf.Tensor | None = None,
    ) -> List[Explanation] | tf.Tensor:
        from transformers_gradients.tasks.text_classification import gradient_norm

        return gradient_norm(
            model, x_batch, y_batch, tokenizer=tokenizer, attention_mask=attention_mask
        )

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
    def gradient_x_input(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        attention_mask: tf.Tensor | None = None,
    ) -> List[Explanation] | tf.Tensor:
        from transformers_gradients.tasks.text_classification import gradient_x_input

        return gradient_x_input(
            model, x_batch, y_batch, tokenizer=tokenizer, attention_mask=attention_mask
        )

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
    def integrated_gradients(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        attention_mask: tf.Tensor | None = None,
        config: IntGradConfig | None = None,
    ) -> List[Explanation] | tf.Tensor:
        from transformers_gradients.tasks.text_classification import (
            integrated_gradients,
        )

        return integrated_gradients(
            model,
            x_batch,
            y_batch,
            tokenizer=tokenizer,
            attention_mask=attention_mask,
            config=config,
        )

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
    def smooth_grad(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        attention_mask: tf.Tensor | None = None,
        config: SmoothGradConfing | None = None,
    ) -> List[Explanation] | tf.Tensor:
        from transformers_gradients.tasks.text_classification import smooth_grad

        return smooth_grad(
            model,
            x_batch,
            y_batch,
            tokenizer=tokenizer,
            attention_mask=attention_mask,
            config=config,
        )

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
    def noise_grad(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        attention_mask: tf.Tensor | None = None,
        config: NoiseGradConfig | None = None,
    ) -> List[Explanation] | tf.Tensor:
        from transformers_gradients.tasks.text_classification import smooth_grad

        return smooth_grad(
            model,
            x_batch,
            y_batch,
            tokenizer=tokenizer,
            attention_mask=attention_mask,
            config=config,
        )

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
    def noise_grad_plus_plus(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        attention_mask: tf.Tensor | None = None,
        config: NoiseGradPlusPlusConfig | None = None,
    ) -> List[Explanation] | tf.Tensor:
        from transformers_gradients.tasks.text_classification import (
            noise_grad_plus_plus,
        )

        return noise_grad_plus_plus(
            model,
            x_batch,
            y_batch,
            tokenizer=tokenizer,
            attention_mask=attention_mask,
            config=config,
        )

    # ----------------------------------------------------------------------------
    @staticmethod
    def lime(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        config: LimeConfig | None = None,
    ) -> List[Explanation]:
        from transformers_gradients.tasks.text_classification import lime

        return lime(model, x_batch, y_batch, tokenizer=tokenizer, config=config)
