from __future__ import annotations

import sys
from functools import wraps, partial
from typing import List, Mapping

import tensorflow as tf
from keras.engine.data_adapter import TensorLikeDataAdapter
import tensorflow_probability as tfp
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase


from transformers_gradients.functions import (
    logits_for_labels,
    zeros_baseline,
    multiplicative_noise,
    sample_masks,
    mask_tokens,
    ridge_regression,
)
from transformers_gradients.lib_types import (
    IntGradConfig,
    NoiseGradPlusPlusConfig,
    NoiseGradConfig,
    SmoothGradConfing,
    LimeConfig,
    Explanation,
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

# TODO implement batch size limit as hook for all methods


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
    config: IntGradConfig | Mapping[str, ...] | None = None,
) -> tf.Tensor:
    config = mapping_to_config(config, IntGradConfig)
    config = value_or_default(config, lambda: IntGradConfig())
    baseline_fn = value_or_default(config.baseline_fn, lambda: zeros_baseline)
    num_steps = tf.constant(config.num_steps)

    if (
        config.batch_size_limit >= len(x_batch) * num_steps
        or not config.batch_interpolated_inputs
    ):
        baseline = baseline_fn(x_batch)
        interpolated_embeddings = tfp.math.batch_interp_regular_1d_grid(
            x=tf.cast(tf.range(num_steps + 1), dtype=tf.float32),
            x_ref_min=tf.cast(0, dtype=tf.float32),
            x_ref_max=tf.cast(num_steps, dtype=tf.float32),
            y_ref=[x_batch, baseline],
            axis=0,
        )

        if config.batch_interpolated_inputs:
            return integrated_gradients_batched(
                model,
                interpolated_embeddings,
                y_batch,
                attention_mask,
                num_steps,
            )
        else:
            return integrated_gradients_iterative(
                model,
                interpolated_embeddings,
                y_batch,
                attention_mask,
            )

    x_batch_shards = TensorLikeDataAdapter(
        (x_batch, attention_mask),
        y_batch,
        batch_size=config.batch_size_limit // num_steps,
    )
    a_batch = tf.TensorArray(
        dtype=x_batch.dtype,
        size=x_batch_shards.get_size(),
        clear_after_read=True,
        dynamic_size=True,
        element_shape=[None, None],
        infer_shape=False,
    )
    for i, ((x, am), y) in enumerate(x_batch_shards.get_dataset()):
        baseline = baseline_fn(x)
        interpolated_embeddings = tfp.math.batch_interp_regular_1d_grid(
            x=tf.cast(tf.range(num_steps + 1), dtype=tf.float32),
            x_ref_min=tf.cast(0, dtype=tf.float32),
            x_ref_max=tf.cast(num_steps, dtype=tf.float32),
            y_ref=[x, baseline],
            axis=0,
        )
        a = integrated_gradients_batched(
            model, interpolated_embeddings, y, am, num_steps
        )
        a_batch = a_batch.write(i, a)

    return a_batch.concat()


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
    apply_noise_fn = value_or_default(config.noise_fn, lambda: multiplicative_noise)
    apply_noise_fn = resolve_noise_fn(apply_noise_fn)  # type: ignore

    explanations_array = tf.TensorArray(
        x_batch.dtype,
        size=config.n,
        clear_after_read=True,
    )

    noise_dist = tfp.distributions.Normal(config.mean, config.std)

    def noise_fn(x):
        noise = noise_dist.sample(tf.shape(x))
        return apply_noise_fn(x, noise)

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
    apply_noise_fn = value_or_default(config.noise_fn, lambda: multiplicative_noise)
    apply_noise_fn = resolve_noise_fn(apply_noise_fn)  # type: ignore

    original_weights = model.weights.copy()

    explanations_array = tf.TensorArray(
        x_batch.dtype,
        size=config.n,
        clear_after_read=True,
    )

    noise_dist = tfp.distributions.Normal(config.mean, config.std)

    def noise_fn(x):
        noise = noise_dist.sample(tf.shape(x))
        return apply_noise_fn(x, noise)

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
def noise_grad_plus_plus(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    *,
    attention_mask: tf.Tensor | None = None,
    config: NoiseGradPlusPlusConfig | Mapping[str, ...] | None = None,
) -> tf.Tensor:
    config = mapping_to_config(config, NoiseGradPlusPlusConfig)
    config = value_or_default(config, lambda: NoiseGradPlusPlusConfig())
    sg_config = SmoothGradConfing(
        n=config.m,
        mean=config.sg_mean,
        std=config.sg_std,
        explain_fn=config.explain_fn,
        noise_fn=config.noise_fn,
    )
    sg_explain_fn = partial(smooth_grad, config=sg_config)
    ng_config = NoiseGradConfig(
        n=config.n, mean=config.mean, explain_fn=sg_explain_fn, noise_fn=config.noise_fn
    )
    return noise_grad(
        model,
        x_batch,
        y_batch,
        attention_mask=attention_mask,
        config=ng_config,
    )


# ----------------------- IntGrad ------------------------


def integrated_gradients_batched(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor,
    num_steps: tf.Tensor,
) -> tf.Tensor:
    num_steps = tf.constant(num_steps)
    shape = tf.shape(x_batch)
    batch_size = shape[1]

    interpolated_embeddings = tf.reshape(
        tf.cast(x_batch, dtype=tf.float32),
        [-1, shape[2], shape[3]],
    )
    interpolated_y_batch = tf.repeat(y_batch, num_steps + 1)
    interpolated_mask = tf.repeat(attention_mask, num_steps + 1, axis=0)

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


def integrated_gradients_iterative(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor,
) -> tf.Tensor:
    batch_size = tf.shape(x_batch)[1]
    scores = tf.TensorArray(
        x_batch.dtype,
        size=batch_size,
        clear_after_read=True,
    )

    for i in tf.range(batch_size):
        interpolated_embeddings = x_batch[i]

        attention_mask_i = tf.repeat(
            tf.expand_dims(attention_mask[i], axis=0),
            tf.shape(interpolated_embeddings)[0],
            axis=0,
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

    for i, y in enumerate(y_batch):
        ids = tokenizer(x_batch[i], return_tensors="tf")["input_ids"][0]
        masks = sample_masks(num_samples - 1, len(ids), seed=42)
        if masks.shape[0] != num_samples - 1:
            raise ValueError("Expected num_samples + 1 masks.")

        all_true_mask = tf.ones_like(masks[0], dtype=tf.bool)
        masks = tf.concat([tf.expand_dims(all_true_mask, 0), masks], axis=0)

        perturbations = mask_tokens(ids, masks, mask_token_id)
        logits = model(perturbations).logits
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
