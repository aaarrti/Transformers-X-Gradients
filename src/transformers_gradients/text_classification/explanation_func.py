from __future__ import annotations

import gc
from functools import wraps, partial
from operator import itemgetter

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions.normal import Normal
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

from transformers_gradients.config import (
    IntGradConfig,
    NoiseGradPlusPlusConfig,
    NoiseGradConfig,
    SmoothGradConfing,
    resolve_baseline_explain_fn,
)
from transformers_gradients.types import (
    Explanation,
)
from transformers_gradients.util import (
    value_or_default,
    is_xla_compatible_platform,
    get_input_ids,
    as_tensor,
    map_dict,
)


def plain_text_hook(func):
    @wraps(func)
    def wrapper(
        model: TFPreTrainedModel,
        x_batch: list[str] | tf.Tensor,
        y_batch: tf.Tensor,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ):
        if isinstance(x_batch[0], str):
            input_ids, predict_kwargs = get_input_ids(tokenizer, x_batch)
            embeddings = model.get_input_embeddings()(input_ids)
            scores = func(
                model, embeddings, as_tensor(y_batch), **kwargs, **predict_kwargs
            )
            return [
                (tokenizer.convert_ids_to_tokens(list(i)), j)
                for i, j in zip(input_ids, scores)
            ]
        else:
            return func(model, as_tensor(x_batch), as_tensor(y_batch), **kwargs)

    return wrapper


@plain_text_hook
@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def gradient_norm(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    **kwargs,
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

    kwargs:
        If x_batch is embeddings, kwargs can be used to pass, additional forward pass kwargs, e.g., attention mask.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """

    @tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
    def grad_norm_fn(xx_batch, yy_batch, **kwargs):
        with tf.GradientTape() as tape:
            tape.watch(xx_batch)
            logits = model(
                None, inputs_embeds=xx_batch, training=False, **kwargs
            ).logits
            logits_for_label = logits_for_labels(logits, yy_batch)

        grads = tape.gradient(logits_for_label, xx_batch)
        return tf.linalg.norm(grads, axis=-1)

    return grad_norm_fn(x_batch, y_batch, **kwargs)


@plain_text_hook
def gradient_x_input(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    **kwargs,
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
    kwargs:
        If x_batch is embeddings, kwargs can be used to pass, additional forward pass kwargs, e.g., attention mask.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """

    @tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
    def grad_x_input_fn(xx_batch, yy_batch, **kwargs):
        with tf.GradientTape() as tape:
            tape.watch(xx_batch)
            logits = model(
                None, inputs_embeds=xx_batch, training=False, **kwargs
            ).logits
            logits_for_label = logits_for_labels(logits, yy_batch)
        grads = tape.gradient(logits_for_label, xx_batch)
        return tf.math.reduce_sum(xx_batch * grads, axis=-1)

    return grad_x_input_fn(x_batch, y_batch, **kwargs)


@plain_text_hook
def integrated_gradients(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    config: IntGradConfig | None = None,
    **kwargs,
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
    interpolated_embeddings = tf.vectorized_map(
        lambda i: interpolate_inputs(
            config.baseline_fn(i), i, tf.constant(config.num_steps)
        ),
        x_batch,
    )

    if config.batch_interpolated_inputs:
        return _integrated_gradients_batched(
            model,
            interpolated_embeddings,
            y_batch,
            tf.constant(config.num_steps),
            **kwargs,
        )
    else:
        return _integrated_gradients_iterative(
            model,
            interpolated_embeddings,
            y_batch,
            **kwargs,
        )


@plain_text_hook
# @tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def smooth_grad(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    config: SmoothGradConfing | None = None,
    **kwargs,
) -> list[Explanation] | tf.Tensor:
    config = value_or_default(config, lambda: SmoothGradConfing())
    explain_fn = resolve_baseline_explain_fn(config.explain_fn)

    def smooth_grad_fn(xx_batch, yy_batch, n, mean, std, **kwargs):
        explanations_array = tf.TensorArray(
            xx_batch.dtype,
            size=n,
            clear_after_read=True,
            colocate_with_first_write_call=True,
        )

        noise_dist = Normal(mean, std)

        def noise_fn(x):
            noise = noise_dist.sample(tf.shape(x))
            return config.noise_fn(x, noise)

        for n in tf.range(n):
            noisy_x = noise_fn(xx_batch)
            explanation = explain_fn(model, noisy_x, yy_batch, **kwargs)
            explanations_array = explanations_array.write(n, explanation)

        scores = tf.reduce_mean(explanations_array.stack(), axis=0)
        explanations_array.close()
        return scores

    return smooth_grad_fn(
        x_batch,
        y_batch,
        tf.constant(config.n),
        tf.constant(config.mean),
        tf.constant(config.std),
        **kwargs,
    )


@plain_text_hook
def noise_grad(
    model: TFPreTrainedModel,
    x_batch: list[str] | tf.Tensor,
    y_batch: tf.Tensor,
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

    config = value_or_default(config, lambda: NoiseGradConfig())
    explain_fn = resolve_baseline_explain_fn(config.explain_fn)
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
        return config.noise_fn(x, noise)

    for n in tf.range(config.n):
        noisy_weights = tf.nest.map_structure(
            noise_fn,
            original_weights,
        )
        model.set_weights(noisy_weights)

        explanation = explain_fn(model, x_batch, y_batch, **kwargs)
        explanations_array = explanations_array.write(n, explanation)

    scores = tf.reduce_mean(explanations_array.stack(), axis=0)
    model.set_weights(original_weights)
    explanations_array.close()
    tf.keras.backend.clear_session()
    gc.collect()
    return scores


@plain_text_hook
def noise_grad_plus_plus(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
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
    sg_config = SmoothGradConfing(
        n=config.m,
        mean=config.sg_mean,
        std=config.sg_std,
        explain_fn=config.explain_fn,
        noise_fn=config.noise_fn,
    )
    ng_config = NoiseGradConfig(
        n=config.n,
        mean=config.mean,
        noise_fn=config.noise_fn,
        explain_fn=partial(smooth_grad, config=sg_config),
    )
    return noise_grad(model, x_batch, y_batch, config=ng_config, **kwargs)


# ----------------------- IntGrad ------------------------


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def _integrated_gradients_batched(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    num_steps: int,
    **kwargs,
):
    @tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
    def int_grad_fn(xx_batch, yy_batch, nnum_steps, **kwargs):
        shape = tf.shape(xx_batch)
        batch_size = shape[0]

        interpolated_embeddings = tf.reshape(
            tf.cast(xx_batch, dtype=tf.float32),
            [-1, shape[2], shape[3]],
        )

        def pseudo_interpolate(x):
            og_shape = tf.convert_to_tensor(tf.shape(x))
            new_shape = tf.concat([[nnum_steps + 1], og_shape], axis=0)
            x = tf.broadcast_to(x, new_shape)
            flat_shape = tf.concat([tf.constant([-1]), og_shape[1:]], axis=0)
            x = tf.reshape(x, flat_shape)
            return x

        interpolated_kwargs = tf.nest.map_structure(pseudo_interpolate, kwargs)
        interpolated_y_batch = pseudo_interpolate(yy_batch)

        with tf.GradientTape() as tape:
            tape.watch(interpolated_embeddings)
            logits = model(
                None,
                inputs_embeds=interpolated_embeddings,
                training=False,
                **interpolated_kwargs,
            ).logits
            logits_for_label = logits_for_labels(logits, interpolated_y_batch)

        grads = tape.gradient(logits_for_label, interpolated_embeddings)
        grads_shape = tf.shape(grads)
        grads = tf.reshape(
            grads, [batch_size, nnum_steps + 1, grads_shape[1], grads_shape[2]]
        )
        return tf.linalg.norm(tfp.math.trapz(grads, axis=1), axis=-1)

    return int_grad_fn(x_batch, y_batch, num_steps, **kwargs)


def _integrated_gradients_iterative(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    **kwargs,
) -> tf.Tensor:
    batch_size = tf.shape(x_batch)[0]
    scores = tf.TensorArray(
        x_batch.dtype,
        size=batch_size,
        clear_after_read=True,
        colocate_with_first_write_call=True,
    )

    def pseudo_interpolate(x, embeds):
        return tf.broadcast_to(x, (tf.shape(embeds)[0], *x.shape))

    for i in tf.range(batch_size):
        interpolated_embeddings = x_batch[i]

        interpolated_kwargs = tf.nest.map_structure(
            lambda x: pseudo_interpolate(x, interpolated_embeddings),
            map_dict(kwargs, itemgetter(i)),
        )
        with tf.GradientTape() as tape:
            tape.watch(interpolated_embeddings)
            logits = model(
                None,
                inputs_embeds=interpolated_embeddings,
                training=False,
                **interpolated_kwargs,
            ).logits
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
    baseline: tf.Tensor, target: tf.Tensor, num_steps: tf.Tensor
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
