from __future__ import annotations

from functools import partial, singledispatch, wraps
from typing import Callable, Dict, List, Optional, Union
import gc

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions.normal import Normal
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

from transformers_gradients.types import (
    Explanation,
    ModelI,
    TokenizerI,
    BaselineFn,
)
from transformers_gradients.config import (
    IntGradConfig,
    NoiseGradPlusPlusConfig,
    NoiseGradConfig,
    SmoothGradConfing,
    resolve_baseline_explain_fn,
)
from transformers_gradients.util import (
    value_or_default,
    is_xla_compatible_platform,
    get_input_ids,
    as_tensor,
)


def plain_text_hook(func):
    @wraps(func)
    def wrapper(
        model: ModelI,
        x_batch: list[str] | tf.Tensor,
        y_batch: tf.Tensor,
        tokenizer: TokenizerI | None = None,
        *args,
        **kwargs,
    ):
        if isinstance(x_batch[0], str):
            input_ids, predict_kwargs = get_input_ids(tokenizer, x_batch)
            embeddings = model.embedding_lookup(input_ids)
            scores = func(model, embeddings, y_batch, *args, **predict_kwargs)
            return [
                (tokenizer.convert_ids_to_tokens(list(i)), j)
                for i, j in zip(input_ids, scores)
            ]
        else:
            return func(model, x_batch, y_batch, *args, **kwargs)

    return wrapper


@plain_text_hook
@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def gradient_norm(
    model: ModelI,
    x_batch: list[str] | tf.Tensor,
    y_batch: tf.Tensor,
    tokenizer: TokenizerI | None = None,
    **kwargs,
) -> list[Explanation] | tf.Tensor:
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
    tokenizer:
        Optional tokenizer, which should be passed in case x_batch is plain-text.

    kwargs:
        If x_batch is embeddings, kwargs can be used to pass, additional forward pass kwargs, e.g., attention mask.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    x_batch = as_tensor(x_batch)
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        logits = model(inputs_embeds=x_batch, **kwargs)
        logits_for_label = logits_for_labels(logits, y_batch)

    grads = tape.gradient(logits_for_label, x_batch)
    return tf.linalg.norm(grads, axis=-1)


@plain_text_hook
@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def gradient_x_input(
    model: ModelI,
    x_batch: list[str] | tf.Tensor,
    y_batch: tf.Tensor,
    tokenizer: TokenizerI | None = None,
    **kwargs,
) -> list[Explanation] | tf.Tensor:
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
    kwargs:
        If x_batch is embeddings, kwargs can be used to pass, additional forward pass kwargs, e.g., attention mask.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    x_batch = as_tensor(x_batch)
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        logits = model(inputs_embeds=x_batch, **kwargs)
        logits_for_label = logits_for_labels(logits, y_batch)
    grads = tape.gradient(logits_for_label, x_batch)
    return tf.math.reduce_sum(x_batch * grads, axis=-1)


@plain_text_hook
@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def integrated_gradients(
    model: ModelI,
    x_batch: list[str] | tf.Tensor,
    y_batch: tf.Tensor,
    tokenizer: TokenizerI | None = None,
    config: IntGradConfig | None = None,
    **kwargs,
) -> list[Explanation] | tf.Tensor:
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
    config:

    kwargs:
        If x_batch is embeddings, kwargs can be used to pass, additional forward pass kwargs, e.g., attention mask.

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
    x_batch = as_tensor(x_batch)
    if config.batch_interpolated_inputs:
        return _integrated_gradients_batched(
            model,
            x_batch,
            y_batch,
            config.num_steps,
            config.baseline_fn,
            **kwargs,
        )
    else:
        return _integrated_gradients_iterative(
            model,
            x_batch,
            y_batch,
            config.num_steps,
            config.baseline_fn,
            **kwargs,
        )


@plain_text_hook
@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def smooth_grad(
    model: ModelI,
    x_batch: list[str] | tf.Tensor,
    y_batch: tf.Tensor,
    tokenizer: TokenizerI | None = None,
    config: SmoothGradConfing | None = None,
    **kwargs,
) -> list[Explanation] | tf.Tensor:
    config = value_or_default(config, lambda: SmoothGradConfing())
    explain_fn = resolve_baseline_explain_fn(config.explain_fn)

    explanations_array = tf.TensorArray(
        x_batch.dtype,
        size=config.n,
        clear_after_read=True,
        colocate_with_first_write_call=True,
    )

    noise_dist = Normal(config.mean, config.std)

    def noise_fn(x):
        noise = noise_dist.sample(tf.shape(x))
        return config.noise_fn(x, noise)

    for n in tf.range(config.n):
        noisy_x = noise_fn(x_batch)
        explanation = explain_fn(model, noisy_x, y_batch, **kwargs)
        explanations_array = explanations_array.write(n, explanation)

    scores = tf.reduce_mean(explanations_array.stack(), axis=0)
    explanations_array.close()
    return scores


@plain_text_hook
def noise_grad(
    model: ModelI,
    x_batch: list[str] | tf.Tensor,
    y_batch: tf.Tensor,
    tokenizer: TokenizerI | None = None,
    config: NoiseGradConfig | None = None,
    **kwargs,
) -> list[Explanation] | tf.Tensor:
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
    config:

    kwargs:
        If x_batch is embeddings, kwargs can be used to pass, additional forward pass kwargs, e.g., attention mask.

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

    config = value_or_default(config, lambda: NoiseGradPlusPlusConfig())
    explain_fn = resolve_baseline_explain_fn(config.explain_fn)
    original_weights = model.model.weights.copy()

    explanations_array = tf.TensorArray(
        x_batch.dtype,
        size=config.n,
        clear_after_read=True,
        colocate_with_first_write_call=True,
    )

    noise_dist = Normal(config.mean, config.std)

    def noise_fn(x):
        noise = noise_dist.sample(tf.shape(x))
        return config.noise_fn(x, noise)

    for n in tf.range(config.n):
        noisy_weights = tf.nest.map_structure(
            noise_fn,
            original_weights,
        )
        model.model.set_weights(noisy_weights)

        explanation = explain_fn(model, x_batch, y_batch, **kwargs)
        explanations_array = explanations_array.write(n, explanation)

    scores = tf.reduce_mean(explanations_array.stack(), axis=0)
    model.model.set_weights(original_weights)
    explanations_array.close()
    tf.keras.backend.clear_session()
    gc.collect()
    return scores


@plain_text_hook
def noise_grad_plus_plus(
    model: ModelI,
    x_batch: list[str] | tf.Tensor,
    y_batch: tf.Tensor,
    tokenizer: TokenizerI | None = None,
    config: NoiseGradPlusPlusConfig | None = None,
    **kwargs,
) -> list[Explanation] | tf.Tensor:
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

    kwargs:
        If x_batch is embeddings, kwargs can be used to pass, additional forward pass kwargs, e.g., attention mask.

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
    explain_fn = resolve_baseline_explain_fn(config.explain_fn)

    original_weights = model.model.get_weights().copy()

    noise_dist = Normal(config.mean, config.std)
    sg_noise_dist = Normal(config.sg_mean, config.sg_std)

    explanations_array = tf.TensorArray(
        x_batch.dtype,
        size=config.n * config.m,
        clear_after_read=True,
        colocate_with_first_write_call=True,
    )

    def noise_fn(x):
        noise = noise_dist.sample(tf.shape(x))
        return config.noise_fn(x, noise)

    def sg_noise_fn(x):
        noise = sg_noise_dist.sample(tf.shape(x))
        return config.noise_fn(x, noise)

    for n in tf.range(config.n):
        noisy_weights = tf.nest.map_structure(noise_fn, original_weights)
        model.weights = noisy_weights

        for m in tf.range(config.m):
            noisy_embeddings = sg_noise_fn(x_batch)
            explanation = explain_fn(model, noisy_embeddings, y_batch, **kwargs)  # type: ignore # noqa
            explanations_array = explanations_array.write(n + m * config.m, explanation)

    scores = tf.reduce_mean(explanations_array.stack(), axis=0)
    model.model.set_weights(original_weights)
    return scores


# ----------------------- IntGrad ------------------------


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def _integrated_gradients_batched(
    model: ModelI,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    num_steps: int,
    baseline_fn: BaselineFn,
    **kwargs,
):
    interpolated_embeddings = tf.vectorized_map(
        lambda i: interpolate_inputs(baseline_fn(i), i, num_steps), x_batch
    )

    shape = tf.shape(interpolated_embeddings)
    batch_size = shape[0]

    interpolated_embeddings = tf.reshape(
        tf.cast(interpolated_embeddings, dtype=tf.float32),
        [-1, shape[2], shape[3]],
    )

    def pseudo_interpolate(x, n):
        og_shape = tf.convert_to_tensor(tf.shape(x))
        new_shape = tf.concat([tf.constant([n + 1]), og_shape], axis=0)
        x = tf.broadcast_to(x, new_shape)
        flat_shape = tf.concat([tf.constant([-1]), og_shape[1:]], axis=0)
        x = tf.reshape(x, flat_shape)
        return x

    interpolated_kwargs = tf.nest.map_structure(
        partial(pseudo_interpolate, n=num_steps), kwargs
    )
    interpolated_y_batch = pseudo_interpolate(y_batch, num_steps)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_embeddings)
        # model(None, inputs_embeds=x_batch, training=False, **kwargs).logits
        logits = model(interpolated_embeddings, **interpolated_kwargs)
        logits_for_label = logits_for_labels(logits, interpolated_y_batch)

    grads = tape.gradient(logits_for_label, interpolated_embeddings)
    grads_shape = tf.shape(grads)
    grads = tf.reshape(
        grads, [batch_size, num_steps + 1, grads_shape[1], grads_shape[2]]
    )
    return tf.linalg.norm(tfp.math.trapz(grads, axis=1), axis=-1)


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def _integrated_gradients_iterative(
    model: ModelI,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    num_steps: int,
    baseline_fn: BaselineFn,
    **kwargs,
) -> tf.Tensor:
    interpolated_embeddings_batch = tf.map_fn(
        lambda x: interpolate_inputs(baseline_fn(x), x, num_steps),
        x_batch,
    )

    batch_size = tf.shape(interpolated_embeddings_batch)[0]

    scores = tf.TensorArray(
        x_batch.dtype,
        size=batch_size,
        clear_after_read=True,
        colocate_with_first_write_call=True,
    )

    def pseudo_interpolate(x, embeds):
        return tf.broadcast_to(x, (tf.shape(embeds)[0], *x.shape))

    for i in tf.range(batch_size):
        interpolated_embeddings = interpolated_embeddings_batch[i]

        interpolated_kwargs = tf.nest.map_structure(
            lambda x: pseudo_interpolate(x, interpolated_embeddings),
            {k: v[i] for k, v in kwargs.items()},
        )
        with tf.GradientTape() as tape:
            tape.watch(interpolated_embeddings)
            # model(None, inputs_embeds=x_batch, training=False, **kwargs).logits
            logits = model(interpolated_embeddings, **interpolated_kwargs)
            logits_for_label = logits[:, y_batch[i]]

        grads = tape.gradient(logits_for_label, interpolated_embeddings)
        score = tf.linalg.norm(tfp.math.trapz(grads, axis=0), axis=-1)
        scores = scores.write(i, score)

    scores_stack = scores.stack()
    scores.close()
    return scores_stack


# --------------------- utils ----------------------


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def logits_for_labels(logits: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
    # Matrix with indexes like [ [0,y_0], [1, y_1], ...]
    indexes = tf.transpose(
        tf.stack(
            [
                tf.range(tf.shape(logits)[0], dtype=tf.int32),
                tf.cast(y_batch, tf.int32),
            ]
        ),
        [1, 0],
    )
    return tf.gather_nd(logits, indexes)


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def interpolate_inputs(
    baseline: tf.Tensor, target: tf.Tensor, num_steps: int
) -> tf.Tensor:
    """Gets num_step linearly interpolated inputs from baseline to target."""
    delta = target - baseline
    scales = tf.linspace(0, 1, num_steps + 1)[:, tf.newaxis, tf.newaxis]
    scales = tf.cast(scales, dtype=delta.dtype)
    shape = tf.convert_to_tensor(
        [num_steps + 1, tf.shape(delta)[0], tf.shape(delta)[1]]
    )
    deltas = scales * tf.broadcast_to(delta, shape)
    interpolated_inputs = baseline + deltas
    return interpolated_inputs
