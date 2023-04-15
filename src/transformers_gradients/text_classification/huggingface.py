from __future__ import annotations

from functools import wraps, partial
from typing import List, Callable

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions.normal import Normal
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

from transformers_gradients.functions import (
    logits_for_labels,
    interpolate_inputs,
    zeros_baseline,
    multiplicative_noise,
    pseudo_interpolate,
    broadcast_expand_dims,
)
from transformers_gradients.types import (
    IntGradConfig,
    NoiseGradPlusPlusConfig,
    NoiseGradConfig,
    SmoothGradConfing,
    BaselineFn,
    BaselineExplainFn,
    ExplainFn,
    ApplyNoiseFn,
)
from transformers_gradients.util import (
    value_or_default,
    encode_inputs,
    as_tensor,
    tensor_inputs,
)


def plain_text_hook(func):
    @wraps(func)
    def wrapper(
        model: TFPreTrainedModel,
        x_batch: List[str] | tf.Tensor,
        y_batch: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ):
        if not isinstance(x_batch[0], str):
            return func(
                model, as_tensor(x_batch), as_tensor(y_batch), attention_mask, **kwargs
            )

        input_ids, attention_mask = encode_inputs(tokenizer, x_batch)
        embeddings = model.get_input_embeddings()(input_ids)
        scores = func(model, embeddings, as_tensor(y_batch), attention_mask, **kwargs)
        return [
            (tokenizer.convert_ids_to_tokens(list(i)), j)
            for i, j in zip(input_ids, scores)
        ]

    return wrapper


@plain_text_hook
@tensor_inputs
def gradient_norm(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor | None = None,
) -> tf.Tensor:
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

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        logits = model(
            None, inputs_embeds=x_batch, training=False, attention_mask=attention_mask
        ).logits
        logits_for_label = logits_for_labels(logits, y_batch)

    grads = tape.gradient(logits_for_label, x_batch)
    return tf.linalg.norm(grads, axis=-1)


@plain_text_hook
@tensor_inputs
def gradient_x_input(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor | None = None,
) -> tf.Tensor:
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

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        logits = model(
            None, inputs_embeds=x_batch, training=False, attention_mask=attention_mask
        ).logits
        logits_for_label = logits_for_labels(logits, y_batch)
    grads = tape.gradient(logits_for_label, x_batch)
    return tf.math.reduce_sum(x_batch * grads, axis=-1)


@plain_text_hook
@tensor_inputs
def integrated_gradients(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor | None = None,
    config: IntGradConfig | None = None,
    baseline_fn: BaselineFn | None = None,
) -> tf.Tensor:
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

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    Examples
    -------
    Specifying [UNK] token as baseline:

    >>> unk_token_embedding = model.embedding_lookup([model.tokenizer.unk_token_id])[0, 0]
    >>> unknown_token_baseline_function = tf.function(lambda x: unk_token_embedding)
    >>> config = IntGradConfig(baseline_fn=unknown_token_baseline_function)
    >>> integrated_gradients(..., ..., ..., config=config)

    """
    config = value_or_default(config, lambda: IntGradConfig())
    baseline_fn = value_or_default(baseline_fn, lambda: zeros_baseline)
    interpolated_embeddings = interpolate_inputs(x_batch, config.num_steps, baseline_fn)

    if config.batch_interpolated_inputs:
        return _integrated_gradients_batched(
            model,
            interpolated_embeddings,
            y_batch,
            attention_mask,
            tf.constant(config.num_steps),
        )
    else:
        return _integrated_gradients_iterative(
            model,
            interpolated_embeddings,
            y_batch,
            attention_mask,
        )


@plain_text_hook
@tensor_inputs
def smooth_grad(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor | None = None,
    config: SmoothGradConfing | None = None,
    explain_fn: ExplainFn | BaselineExplainFn = "IntGrad",
    noise_fn: ApplyNoiseFn | None = None,
) -> tf.Tensor:
    config = value_or_default(config, lambda: SmoothGradConfing())
    explain_fn = resolve_baseline_explain_fn(explain_fn)
    apply_noise_fn = value_or_default(noise_fn, lambda: multiplicative_noise)

    explanations_array = tf.TensorArray(
        x_batch.dtype,
        size=config.n,
        clear_after_read=True,
        colocate_with_first_write_call=True,
    )

    noise_dist = Normal(config.mean, config.std)

    def noise_fn(x):
        noise = noise_dist.sample(tf.shape(x))
        return apply_noise_fn(x, noise)

    for n in tf.range(config.n):
        noisy_x = noise_fn(x_batch)
        explanation = explain_fn(model, noisy_x, y_batch, attention_mask)
        explanations_array = explanations_array.write(n, explanation)

    scores = tf.reduce_mean(explanations_array.stack(), axis=0)
    explanations_array.close()
    return scores


@plain_text_hook
@tensor_inputs
def noise_grad(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor | None = None,
    config: NoiseGradConfig | None = None,
    explain_fn: ExplainFn | BaselineExplainFn = "IntGrad",
    noise_fn: ApplyNoiseFn | None = None,
) -> tf.Tensor:
    """
    NoiseGrad++ is a state-of-the-art gradient based XAI method, which enhances baseline explanation function
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
    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.


    Examples
    -------
    Passing kwargs to baseline explanation function:

    >>> import functools
    >>> ig_config = IntGradConfig(num_steps=22)
    >>> explain_fn = functools.partial(integrated_gradients, config=ig_config)
    >>> ng_config = NoiseGradConfig(explain_fn=explain_fn)
    >>> noise_grad_plus_plus(config=ng_config)

    References
    -------
    - https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.
    - Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne, 2021, NoiseGrad: enhancing explanations by introducing stochasticity to model weights, https://arxiv.org/abs/2106.10185

    """

    config = value_or_default(config, lambda: NoiseGradConfig())
    explain_fn = resolve_baseline_explain_fn(explain_fn)
    apply_noise_fn = value_or_default(noise_fn, lambda: multiplicative_noise)
    original_weights = model.weights.copy()

    explanations_array = tf.TensorArray(
        x_batch.dtype,
        size=config.n,
        clear_after_read=True,
        colocate_with_first_write_call=True,
    )

    noise_dist = Normal(config.mean, config.std)

    def noise_fn(x):
        noise = noise_dist.sample(tf.shape(x))
        return apply_noise_fn(x, noise)

    for n in tf.range(config.n):
        noisy_weights = tf.nest.map_structure(
            noise_fn,
            original_weights,
        )
        model.set_weights(noisy_weights)

        explanation = explain_fn(model, x_batch, y_batch, attention_mask)
        explanations_array = explanations_array.write(n, explanation)

    scores = tf.reduce_mean(explanations_array.stack(), axis=0)
    explanations_array.close()
    model.set_weights(original_weights)
    return scores


@plain_text_hook
@tensor_inputs
def noise_grad_plus_plus(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor | None = None,
    config: NoiseGradPlusPlusConfig | None = None,
    explain_fn="IntGrad",
    noise_fn=None,
) -> tf.Tensor:
    """
    NoiseGrad++ is a state-of-the-art gradient based XAI method, which enhances baseline explanation function
    by adding stochasticity to model's weights and model's inputs. The implementation is based
    on https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs or their embeddings, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    config:
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
    config = value_or_default(config, lambda: NoiseGradPlusPlusConfig())
    sg_config = SmoothGradConfing(
        n=config.m,
        mean=config.sg_mean,
        std=config.sg_std,
    )
    sg_explain_fn = partial(
        smooth_grad, config=sg_config, explain_fn=explain_fn, noise_fn=noise_fn
    )
    ng_config = NoiseGradConfig(
        n=config.n,
        mean=config.mean,
    )
    return noise_grad(
        model,
        x_batch,
        y_batch,
        attention_mask,
        config=ng_config,
        explain_fn=sg_explain_fn,
        noise_fn=noise_fn,
    )


# ----------------------- IntGrad ------------------------


def _integrated_gradients_batched(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor,
    num_steps: tf.Tensor,
) -> tf.Tensor:
    num_steps = tf.constant(num_steps)
    shape = tf.shape(x_batch)
    batch_size = shape[0]

    interpolated_embeddings = tf.reshape(
        tf.cast(x_batch, dtype=tf.float32),
        [-1, shape[2], shape[3]],
    )
    interpolated_y_batch = pseudo_interpolate(y_batch, num_steps)
    interpolated_mask = pseudo_interpolate(attention_mask, num_steps)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_embeddings)
        logits = model(
            None,
            inputs_embeds=interpolated_embeddings,
            training=False,
            attention_mask=interpolated_mask,
        ).logits
        logits_for_label = logits_for_labels(logits, interpolated_y_batch)

    grads = tape.gradient(logits_for_label, interpolated_embeddings)
    grads_shape = tf.shape(grads)
    grads = tf.reshape(
        grads, [batch_size, num_steps + 1, grads_shape[1], grads_shape[2]]
    )
    return tf.linalg.norm(tfp.math.trapz(grads, axis=1), axis=-1)


def _integrated_gradients_iterative(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor,
) -> tf.Tensor:
    batch_size = tf.shape(x_batch)[0]
    scores = tf.TensorArray(
        x_batch.dtype,
        size=batch_size,
        clear_after_read=True,
        colocate_with_first_write_call=True,
    )

    for i in tf.range(batch_size):
        interpolated_embeddings = x_batch[i]

        attention_mask_i = broadcast_expand_dims(
            attention_mask[i], interpolated_embeddings
        )

        with tf.GradientTape() as tape:
            tape.watch(interpolated_embeddings)
            logits = model(
                None,
                inputs_embeds=interpolated_embeddings,
                training=False,
                attention_mask=attention_mask_i,
            ).logits
            logits_for_label = logits[:, y_batch[i]]

        grads = tape.gradient(logits_for_label, interpolated_embeddings)
        score = tf.linalg.norm(tfp.math.trapz(grads, axis=0), axis=-1)
        scores = scores.write(i, score)

    scores_stack = scores.stack()
    scores.close()
    return scores_stack


# --------------------- utils ----------------------


def resolve_baseline_explain_fn(explain_fn):
    if isinstance(explain_fn, Callable):
        return explain_fn  # type: ignore

    method_mapping = {
        "IntGrad": integrated_gradients,
        "GradNorm": gradient_norm,
        "GradXInput": gradient_x_input,
    }
    if explain_fn not in method_mapping:
        raise ValueError(
            f"Unknown XAI method {explain_fn}, supported are {list(method_mapping.keys())}"
        )
    return method_mapping[explain_fn]
