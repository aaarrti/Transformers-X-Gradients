from __future__ import annotations
from types import SimpleNamespace
from typing import overload, List, TYPE_CHECKING, Mapping, Callable


if TYPE_CHECKING:
    import tensorflow as tf
    from transformers_gradients.lib_types import (
        Explanation,
        NoiseGradConfig,
        SmoothGradConfing,
        FusionGradConfig,
        LimeConfig,
    )
    from transformers import TFPreTrainedModel, PreTrainedTokenizerBase


class text_classification(SimpleNamespace):
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
        """
        A baseline GradientNorm text-classification explainer.
        The implementation is based on https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L38.
        GradientNorm explanation algorithm is:
            - Convert inputs to models latent representations.
            - Execute forwards pass
            - Retrieve logits for y_batch.
            - Compute gradient of logits with respect to input embeddings.
            - Compute L2 norm of gradients.

        References:
        ----------
        - https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L38

        Parameters
        ----------
        model:
            A model, which is subject to explanation.
        x_batch:
            A batch of plain text inputs or their embeddings, which are subjects to explanation.
        y_batch:
            A batch of labels, which are subjects to explanation.
        attention_mask:
            Optional attention mask to use, in case input embeddings were provided.
        tokenizer:
            Tokenizer, must be provided for plain-text inputs.

        Returns
        -------
        a_batch:
            List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

        """
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
        """
        A baseline GradientXInput text-classification explainer.
         The implementation is based on https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L108.
         GradientXInput explanation algorithm is:
            - Convert inputs to models latent representations.
            - Execute forwards pass
            - Retrieve logits for y_batch.
            - Compute gradient of logits with respect to input embeddings.
            - Compute vector dot product between input embeddings and gradients.


        References:
        ----------
        - https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L108

        Parameters
        ----------
        model:
            A model, which is subject to explanation.
        x_batch:
            A batch of plain text inputs or their embeddings, which are subjects to explanation.
        y_batch:
            A batch of labels, which are subjects to explanation.
        attention_mask:
            Optional attention mask to use, in case input embeddings were provided.
        tokenizer:
            Tokenizer, must be provided for plain-text inputs.

        Returns
        -------
        a_batch:
            List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

        """

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
        num_steps: int = 10,
        baseline_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
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
        num_steps: int = 10,
        baseline_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
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
        num_steps: int = 10,
        baseline_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    ) -> List[Explanation] | tf.Tensor:
        """
        A baseline Integrated Gradients text-classification explainer. Integrated Gradients explanation algorithm is:
            - Convert inputs to models latent representations.
            - For each x, y in x_batch, y_batch
            - Generate num_steps samples interpolated from baseline to x.
            - Execute forwards pass.
            - Retrieve logits for y.
            - Compute gradient of logits with respect to interpolated samples.
            - Estimate integral over interpolated samples using trapezoid rule.
        In practise, we combine all interpolated samples in one batch, to avoid executing forward and backward passes
        in for-loop. This means potentially, that batch size selected for this XAI method should be smaller than usual.

        References:
        ----------
        - https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L108
        - Sundararajan et al., 2017, Axiomatic Attribution for Deep Networks, https://arxiv.org/pdf/1703.01365.pdf

        Parameters
        ----------
        model:
            A model, which is subject to explanation.
        x_batch:
            A batch of plain text inputs or their embeddings, which are subjects to explanation.
        y_batch:
            A batch of labels, which are subjects to explanation.
        attention_mask:
            Optional attention mask to use, in case input embeddings were provided.
        tokenizer:
            Tokenizer, must be provided for plain-text inputs.
        num_steps:
            Number of interpolated samples, which should be generated, default=10.
        baseline_fn:
            Function used to created baseline values, by default will create zeros tensor. Alternatively, e.g.,
            embedding for [UNK] token could be used.

        Returns
        -------
        a_batch:
            List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

        Examples
        -------
        Specifying [UNK] token as baseline:

        >>> unk_token_embedding = model.embedding_lookup([model.tokenizer.unk_token_id])[0, 0]
        >>> unknown_baseline_function = tf.function(lambda x: unk_token_embedding)
        >>> text_classification.integrated_gradients(..., ..., ..., baseline_fn=unknown_baseline_function)

        """
        from transformers_gradients.tasks.text_classification import (
            integrated_gradients,
        )

        return integrated_gradients(
            model,
            x_batch,
            y_batch,
            tokenizer=tokenizer,
            attention_mask=attention_mask,
            num_steps=num_steps,
            baseline_fn=baseline_fn,
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
        config: SmoothGradConfing | Mapping[str, ...] | None = None,
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
        config: SmoothGradConfing | Mapping[str, ...] | None = None,
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
        config: SmoothGradConfing | Mapping[str, ...] | None = None,
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
        config: NoiseGradConfig | Mapping[str, ...] | None = None,
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
        config: NoiseGradConfig | Mapping[str, ...] | None = None,
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
        config: NoiseGradConfig | Mapping[str, ...] | None = None,
    ) -> List[Explanation] | tf.Tensor:
        """
        NoiseGrad is a state-of-the-art gradient based XAI method, which enhances baseline explanation function
        by adding stochasticity to model's weights. The implementation is based
        on https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.

        Parameters
        ----------
        model:
            A model, which is subject to explanation.
        x_batch:
            A batch of plain text inputs or their embeddings, which are subjects to explanation.
        y_batch:
            A batch of labels, which are subjects to explanation.
        attention_mask:
            Optional attention mask to use, in case input embeddings were provided.
        tokenizer:
            Tokenizer, must be provided for plain-text inputs.
        config:
            Optional config specifying hyperparameters.

        Returns
        -------
        a_batch:
            List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.


        Examples
        -------
        Passing kwargs to baseline explanation function:

        >>> import functools
        >>> ig_config = IntGradConfig(num_steps=22)
        >>> explain_fn = functools.partial(text_classification.integrated_gradients, config=ig_config)
        >>> ng_config = NoiseGradConfig(explain_fn=explain_fn)
        >>> text_classification.noise_grad(config=ng_config)

        References
        -------
        - https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.
        - Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne, 2021, NoiseGrad: enhancing explanations by introducing stochasticity to model weights, https://arxiv.org/abs/2106.10185

        """
        from transformers_gradients.tasks.text_classification import noise_grad

        return noise_grad(
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
    def fusion_grad(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
        config: FusionGradConfig | Mapping[str, ...] | None = None,
    ) -> tf.Tensor:
        ...

    @staticmethod
    @overload
    def fusion_grad(
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
        config: FusionGradConfig | Mapping[str, ...] | None = None,
    ) -> List[Explanation]:
        ...

    @staticmethod
    def fusion_grad(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        attention_mask: tf.Tensor | None = None,
        config: FusionGradConfig | Mapping[str, ...] | None = None,
    ) -> List[Explanation] | tf.Tensor:
        """
        FusionGrad is a fusion of NoiseGrad and SmoothGrad methods.

        Parameters
        ----------
        model:
            A model, which is subject to explanation.
        x_batch:
            A batch of plain text inputs or their embeddings, which are subjects to explanation.
        y_batch:
            A batch of labels, which are subjects to explanation.
        attention_mask:
            Optional attention mask to use, in case input embeddings were provided.
        tokenizer:
            Tokenizer, must be provided for plain-text inputs.
        config:
            Optional config specifying hyper parameters.

        Returns
        -------
        a_batch:
            List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.


        Examples
        -------
        Passing kwargs to baseline explanation function:

        References
        -------
        - https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.
        - Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne, 2021, NoiseGrad: enhancing explanations by introducing stochasticity to model weights, https://arxiv.org/abs/2106.10185

        """
        from transformers_gradients.tasks.text_classification import (
            fusion_grad,
        )

        return fusion_grad(
            model,
            x_batch,
            y_batch,
            tokenizer=tokenizer,
            attention_mask=attention_mask,
            config=config,
        )

    @staticmethod
    @overload
    def noise_grad_plus_plus(
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        *,
        attention_mask: tf.Tensor | None = None,
        config: FusionGradConfig | Mapping[str, ...] | None = None,
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
        config: FusionGradConfig | Mapping[str, ...] | None = None,
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
        config: FusionGradConfig | Mapping[str, ...] | None = None,
    ) -> List[Explanation] | tf.Tensor:
        """NoiseGrad++ is an alternative name for FusionGrad."""
        from transformers_gradients.tasks.text_classification import (
            fusion_grad,
        )

        return fusion_grad(
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
        x_batch: List[str],
        y_batch: tf.Tensor,
        *,
        tokenizer: PreTrainedTokenizerBase,
        config: LimeConfig | Mapping[str, ...] | None = None,
    ) -> List[Explanation]:
        """
        LIME explains classifiers by returning a feature attribution score
        for each input feature. It works as follows:

        1) Sample perturbation masks. First the number of masked features is sampled
            (uniform, at least 1), and then that number of features are randomly chosen
            to be masked out (without replacement).
        2) Get preMappingions from the model for those perturbations. Use these as labels.
        3) Fit a linear model to associate the input positions indicated by the binary
            mask with the resulting preMappinged label.

        The resulting feature importance scores are the linear model coefficients for
        the requested output class or (in case of regression) the output score.

        This is a reimplementation of the original https://github.com/marcotcr/lime
        and is tested for compatibility. This version supports only plain text input.

        Parameters
        ----------
        model:
            A model, which is subject to explanation.
        x_batch:
            A batch of plain text inputs or their embeddings, which are subjects to explanation.
        y_batch:
            A batch of labels, which are subjects to explanation.
        tokenizer:
            Tokenizer.
        config:
            Optional config specifying hyper parameters.

        Returns
        -------
        a_batch:
            List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

        Examples
        -------

        References
        -------
        - https://github.com/marcotcr/lime
        - Marco Tulio Ribeiro and Sameer Singh and Carlos Guestrin, 2016, "Why Should I Trust You?": Explaining the PreMappingions of Any Classifier, https://arxiv.org/abs/1602.04938


        """
        from transformers_gradients.tasks.text_classification import lime

        return lime(model, x_batch, y_batch, tokenizer=tokenizer, config=config)
