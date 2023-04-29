from __future__ import annotations

import sys
from functools import wraps, partial
from typing import List, Mapping

import tensorflow as tf
import tensorflow_probability as tfp
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

from transformers_gradients.functions import (
    logits_for_labels,
    sample_masks,
    mask_tokens,
    ridge_regression,
)
from transformers_gradients.lib_types import (
    FusionGradConfig,
    NoiseGradConfig,
    SmoothGradConfing,
    LimeConfig,
    Explanation,
    BaselineFn,
)
from transformers_gradients.utils import (
    value_or_default,
    encode_inputs,
    as_tensor,
    resolve_baseline_explain_fn,
    resolve_noise_fn,
    mapping_to_config,
)


# ----------------------------------------------------------------------------


def tensor_inputs(func):
    from transformers_gradients.functions import default_attention_mask

    @wraps(func)
    def wrapper(
        model,
        x_batch,
        y_batch,
        *,
        attention_mask=None,
        **kwargs,
    ):
        x_batch = as_tensor(x_batch)
        y_batch = as_tensor(y_batch)
        attention_mask = value_or_default(
            attention_mask, partial(default_attention_mask, x_batch)
        )
        attention_mask = as_tensor(attention_mask)
        return func(model, x_batch, y_batch, attention_mask=attention_mask, **kwargs)

    return wrapper


def plain_text_inputs(func):
    @wraps(func)
    def wrapper(
        model: TFPreTrainedModel,
        x_batch: List[str] | tf.Tensor,
        y_batch,
        *,
        attention_mask=None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ):
        if not isinstance(x_batch[0], str):
            return func(
                model,
                as_tensor(x_batch),
                as_tensor(y_batch),
                attention_mask=attention_mask,
                **kwargs,
            )

        if tokenizer is None:
            raise ValueError("Must provide tokenizer for plain-text inputs.")

        input_ids, attention_mask = encode_inputs(tokenizer, x_batch)
        embeddings = model.get_input_embeddings()(input_ids)
        scores = func(
            model,
            embeddings,
            as_tensor(y_batch),
            attention_mask=attention_mask,
            **kwargs,
        )
        from transformers_gradients import config

        if config.return_raw_scores:
            return scores
        return [
            (tokenizer.convert_ids_to_tokens(list(i)), j)  # type: ignore
            for i, j in zip(input_ids, scores)
        ]

    return wrapper


# ----------------------------------------------------------------------------


@plain_text_inputs
@tensor_inputs
def gradient_norm(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor | None = None,
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        logits = model(
            None, inputs_embeds=x_batch, training=False, attention_mask=attention_mask
        ).logits
        logits_for_label = logits_for_labels(logits, y_batch)

    grads = tape.gradient(logits_for_label, x_batch)
    return tf.linalg.norm(grads, axis=-1)


@plain_text_inputs
@tensor_inputs
def gradient_x_input(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor | None = None,
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        logits = model(
            None, inputs_embeds=x_batch, training=False, attention_mask=attention_mask
        ).logits
        logits_for_label = logits_for_labels(logits, y_batch)
    grads = tape.gradient(logits_for_label, x_batch)
    return tf.math.reduce_sum(x_batch * grads, axis=-1)


@plain_text_inputs
@tensor_inputs
def integrated_gradients(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor | None = None,
    *,
    num_steps: int = 10,
    baseline_fn: BaselineFn | None = None,
) -> tf.Tensor:
    baseline_fn = value_or_default(baseline_fn, lambda: tf.zeros_like)
    num_steps = tf.constant(num_steps)

    baseline = baseline_fn(x_batch)
    dtype = x_batch.dtype
    interpolated_embeddings = tfp.math.batch_interp_regular_1d_grid(
        x=tf.cast(tf.range(num_steps + 1), dtype=dtype),
        x_ref_min=tf.cast(0, dtype=dtype),
        x_ref_max=tf.cast(num_steps, dtype=dtype),
        y_ref=[baseline, x_batch],
        axis=0,
    )

    interpolated_grads = tf.TensorArray(
        size=num_steps + 1,
        dtype=interpolated_embeddings.dtype,
        clear_after_read=True,
    )

    # While at first it may seem a better idea to concatenate all inputs in one new batch,
    # at practise in real use the batch size already is maximized,
    # and create new axis based on num steps, will causes shapes not divisible by 8.
    for i in tf.range(num_steps):
        interpolation_step = interpolated_embeddings[i]

        with tf.GradientTape() as tape:
            tape.watch(interpolation_step)
            logits = model(
                None,
                inputs_embeds=interpolation_step,
                training=False,
                attention_mask=attention_mask,
            ).logits
            logits = logits_for_labels(logits, y_batch)

        grads = tape.gradient(logits, interpolation_step)
        interpolated_grads = interpolated_grads.write(i, grads)

    interpolated_grads_tensor = interpolated_grads.stack()
    interpolated_grads.mark_used()
    interpolated_grads.close()

    # Compute grad norm for each interpolation step.
    interpolated_scores = tf.linalg.norm(interpolated_grads_tensor, axis=-1)

    scores = tfp.math.trapz(interpolated_scores, axis=0)
    return scores


@plain_text_inputs
@tensor_inputs
def smooth_grad(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    *,
    attention_mask: tf.Tensor | None = None,
    config: SmoothGradConfing | Mapping[str, ...] | None = None,
) -> tf.Tensor:
    config = mapping_to_config(config, SmoothGradConfing)
    config = value_or_default(config, lambda: SmoothGradConfing())
    explain_fn = resolve_baseline_explain_fn(sys.modules[__name__], config.explain_fn)
    apply_noise_fn = resolve_noise_fn(config.noise_fn)  # type: ignore

    explanations_array = tf.TensorArray(
        x_batch.dtype,
        size=config.n,
        clear_after_read=True,
    )

    noise_dist = tfp.distributions.Normal(config.mean, config.std)

    def noise_fn(x):
        noise = noise_dist.sample(tf.shape(x))
        return apply_noise_fn(x, tf.cast(noise, dtype=x.dtype))

    for n in tf.range(config.n):
        noisy_x = noise_fn(x_batch)
        explanation = explain_fn(model, noisy_x, y_batch, attention_mask=attention_mask)
        explanations_array = explanations_array.write(n, explanation)

    scores = tf.reduce_mean(explanations_array.stack(), axis=0)
    explanations_array.close()
    return scores


@plain_text_inputs
@tensor_inputs
def noise_grad(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    *,
    attention_mask: tf.Tensor | None = None,
    config: NoiseGradConfig | Mapping[str, ...] | None = None,
) -> tf.Tensor:
    config = mapping_to_config(config, NoiseGradConfig)
    config = value_or_default(config, lambda: NoiseGradConfig())
    explain_fn = resolve_baseline_explain_fn(sys.modules[__name__], config.explain_fn)
    apply_noise_fn = resolve_noise_fn(config.noise_fn)  # type: ignore

    original_weights = model.weights.copy()

    explanations_array = tf.TensorArray(
        x_batch.dtype,
        size=config.n,
        clear_after_read=True,
    )

    noise_dist = tfp.distributions.Normal(config.mean, config.std)

    def noise_fn(x):
        noise = noise_dist.sample(tf.shape(x))
        return apply_noise_fn(x, tf.cast(noise, dtype=x.dtype))

    for n in tf.range(config.n):
        noisy_weights = tf.nest.map_structure(
            noise_fn,
            original_weights,
        )
        model.set_weights(noisy_weights)

        explanation = explain_fn(model, x_batch, y_batch, attention_mask=attention_mask)
        explanations_array = explanations_array.write(n, explanation)

    scores = tf.reduce_mean(explanations_array.stack(), axis=0)
    explanations_array.close()
    model.set_weights(original_weights)
    return scores


@plain_text_inputs
@tensor_inputs
def fusion_grad(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    *,
    attention_mask: tf.Tensor | None = None,
    config: FusionGradConfig | Mapping[str, ...] | None = None,
) -> tf.Tensor:
    config = mapping_to_config(config, FusionGradConfig)
    config = value_or_default(config, lambda: FusionGradConfig())
    sg_config = SmoothGradConfing(
        n=config.m,
        mean=config.sg_mean,
        std=config.sg_std,
        explain_fn=config.explain_fn,
        noise_fn=config.noise_fn,
    )
    sg_explain_fn = partial(smooth_grad, config=sg_config)
    ng_config = NoiseGradConfig(
        n=config.n,
        mean=config.mean,
        explain_fn=sg_explain_fn,  # noqa
        noise_fn=config.noise_fn,
    )
    return noise_grad(
        model,
        x_batch,
        y_batch,
        attention_mask=attention_mask,
        config=ng_config,
    )


# ---------------------------- LIME ----------------------------


def lime(
    model: TFPreTrainedModel,
    x_batch: List[str],
    y_batch: tf.Tensor,
    *,
    tokenizer: PreTrainedTokenizerBase,
    config: LimeConfig | Mapping[str, ...] | None = None,
) -> List[Explanation]:
    config = mapping_to_config(config, LimeConfig)
    config = value_or_default(config, lambda: LimeConfig())
    distance_scale = tf.constant(config.distance_scale)
    mask_token_id = tokenizer.convert_tokens_to_ids(config.mask_token)

    num_samples = tf.constant(config.num_samples)
    a_batch = []

    encoded_inputs = tokenizer(x_batch, return_tensors="tf", padding="longest").data

    for i, y in enumerate(y_batch):
        ids = encoded_inputs["input_ids"][i]
        masks = sample_masks(num_samples - 1, len(ids), seed=42)
        if masks.shape[0] != num_samples - 1:
            raise ValueError("Expected num_samples + 1 masks.")

        all_true_mask = tf.ones_like(masks[0], dtype=tf.bool)
        masks = tf.concat([tf.expand_dims(all_true_mask, 0), masks], axis=0)

        perturbations = mask_tokens(ids, masks, mask_token_id)

        attention_mask = tf.repeat(
            encoded_inputs["attention_mask"][i, tf.newaxis],
            tf.shape(perturbations)[0],
            axis=0,
        )

        logits = model(perturbations, attention_mask=attention_mask).logits
        outputs = logits[:, y]
        distances = tf.keras.losses.cosine_similarity(
            tf.cast(all_true_mask, dtype=tf.float32), tf.cast(masks, dtype=tf.float32)
        )
        distances = distance_scale * distances
        distances = tfp.math.psd_kernels.ExponentiatedQuadratic(
            length_scale=25.0
        ).apply(distances[:, tf.newaxis], tf.zeros_like(distances[:, tf.newaxis]))
        score = ridge_regression(masks, outputs, sample_weight=distances)
        a_batch.append((tokenizer.convert_ids_to_tokens(ids), score))

    return a_batch
