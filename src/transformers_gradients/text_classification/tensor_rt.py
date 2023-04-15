from __future__ import annotations

from functools import wraps, partial
from typing import Callable, List

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
)
from tensorflow_probability.python.distributions.normal import Normal
from transformers import PreTrainedTokenizerBase

from transformers_gradients.functions import (
    logits_for_labels,
    interpolate_inputs,
    zeros_baseline,
    multiplicative_noise,
    pseudo_interpolate,
)
from transformers_gradients.types import (
    SmoothGradConfing,
    IntGradConfig,
    NoiseGradConfig,
    NoiseGradPlusPlusConfig,
    UserObject,
    BaselineFn,
    ApplyNoiseFn,
    ExplainFn,
    BaselineExplainFn,
)
from transformers_gradients.util import value_or_default, encode_inputs, tensor_inputs


# It was designed with TensorRT in mind, but fused TensorRT kernel don't have gradients, e.g.,
# LookupError: No gradient defined for operation'TRTEngineOp_000_001' (op type: TRTEngineOp).
# But it still can be used with saved model API, for a bit of speed up.


def plain_text_hook(func):
    @wraps(func)
    def wrapper(
        model: UserObject,
        x_batch: List[str] | tf.Tensor,
        y_batch: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        embeddings_lookup_fn: Callable[[UserObject, tf.Tensor], tf.Tensor]
        | None = None,
        **kwargs,
    ):
        if not isinstance(x_batch[0], str):
            return func(model, x_batch, y_batch, attention_mask, **kwargs)

        if tokenizer is None or embeddings_lookup_fn is None:
            raise ValueError

        input_ids, attention_mask = encode_inputs(tokenizer, x_batch)
        embeddings = embeddings_lookup_fn(model, input_ids)
        scores = func(model, embeddings, y_batch, attention_mask, **kwargs)
        return [
            (tokenizer.convert_ids_to_tokens(list(i)), j)
            for i, j in zip(input_ids, scores)
        ]

    return wrapper


@plain_text_hook
@tensor_inputs
def gradient_norm(
    model: UserObject,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor,
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        logits = model.signatures[DEFAULT_SERVING_SIGNATURE_DEF_KEY](
            inputs_embeds=x_batch, attention_mask=attention_mask
        )["classifier"]
        logits_for_label = logits_for_labels(logits, y_batch)

    grads = tape.gradient(logits_for_label, x_batch)
    return tf.linalg.norm(grads, axis=-1)


@plain_text_hook
@tensor_inputs
def gradient_x_input(
    model: UserObject,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor,
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        logits = model.signatures[DEFAULT_SERVING_SIGNATURE_DEF_KEY](
            inputs_embeds=x_batch, attention_mask=attention_mask
        )["classifier"]
        logits_for_label = logits_for_labels(logits, y_batch)
    grads = tape.gradient(logits_for_label, x_batch)
    return tf.math.reduce_sum(x_batch * grads, axis=-1)


@plain_text_hook
@tensor_inputs
def integrated_gradients(
    model: UserObject,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor,
    config: IntGradConfig | None = None,
    baseline_fn: BaselineFn | None = None,
) -> tf.Tensor:
    config = value_or_default(config, lambda: IntGradConfig())
    baseline_fn = value_or_default(baseline_fn, lambda: zeros_baseline)

    interpolated_x = interpolate_inputs(x_batch, config.num_steps, baseline_fn)

    shape = tf.shape(interpolated_x)
    batch_size = shape[0]

    interpolated_x = tf.reshape(
        tf.cast(interpolated_x, dtype=tf.float32),
        [-1, shape[2], shape[3]],
    )
    interpolated_attention_mask = pseudo_interpolate(
        attention_mask, tf.constant(config.num_steps)
    )
    interpolated_y_batch = pseudo_interpolate(y_batch, tf.constant(config.num_steps))

    with tf.GradientTape() as tape:
        tape.watch(interpolated_x)
        logits = model.signatures[DEFAULT_SERVING_SIGNATURE_DEF_KEY](
            inputs_embeds=interpolated_x, attention_mask=interpolated_attention_mask
        )["classifier"]
        logits_for_label = logits_for_labels(logits, interpolated_y_batch)

    grads = tape.gradient(logits_for_label, interpolated_x)
    grads_shape = tf.shape(grads)
    grads = tf.reshape(
        grads, [batch_size, config.num_steps + 1, grads_shape[1], grads_shape[2]]
    )
    return tf.linalg.norm(tfp.math.trapz(grads, axis=1), axis=-1)


@plain_text_hook
@tensor_inputs
def smooth_grad(
    model: UserObject,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor,
    config: SmoothGradConfing | None = None,
    explain_fn: ExplainFn | BaselineExplainFn = "IntGrad",
    noise_fn=None,
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


# ------------------------ NoiseGrad ---------------------------------


@plain_text_hook
@tensor_inputs
def noise_grad(
    model: UserObject,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor,
    config: NoiseGradConfig | None = None,
    explain_fn: ExplainFn | BaselineExplainFn = "IntGrad",
    noise_fn: ApplyNoiseFn | None = None,
) -> tf.Tensor:
    config = value_or_default(config, lambda: NoiseGradConfig())
    explain_fn = resolve_baseline_explain_fn(explain_fn)
    apply_noise_fn = value_or_default(noise_fn, lambda: multiplicative_noise)
    original_weights = model.variables.copy()

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
        model.variables = noisy_weights

        explanation = explain_fn(model, x_batch, y_batch, attention_mask)
        explanations_array = explanations_array.write(n, explanation)

    scores = tf.reduce_mean(explanations_array.stack(), axis=0)
    model.variables = original_weights
    explanations_array.close()
    return scores


@plain_text_hook
@tensor_inputs
def noise_grad_plus_plus(
    model: UserObject,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor,
    config: NoiseGradPlusPlusConfig | None = None,
    explain_fn: ExplainFn | BaselineExplainFn = "IntGrad",
    noise_fn: ApplyNoiseFn | None = None,
) -> tf.Tensor:
    config = value_or_default(config, lambda: NoiseGradPlusPlusConfig())
    base_explain_fn = resolve_baseline_explain_fn(explain_fn)
    sg_config = SmoothGradConfing(
        n=config.m,
        mean=config.sg_mean,
        std=config.sg_std,
    )

    explain_fn = partial(
        smooth_grad, config=sg_config, explain_fn=base_explain_fn, noise_fn=noise_fn
    )
    ng_config = NoiseGradConfig(n=config.n, mean=config.mean, std=config.std)
    return noise_grad(
        model,
        x_batch,
        y_batch,
        attention_mask=attention_mask,
        config=ng_config,
        explain_fn=explain_fn,
        noise_fn=noise_fn,
    )


def resolve_baseline_explain_fn(explain_fn: ExplainFn | BaselineExplainFn) -> ExplainFn:
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
