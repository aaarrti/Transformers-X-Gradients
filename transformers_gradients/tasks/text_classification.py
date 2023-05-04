from __future__ import annotations

import sys
from functools import wraps, partial
from typing import List, Mapping, Callable

import tensorflow as tf
import tensorflow_probability as tfp
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

from transformers_gradients.functions import (
    logits_for_labels,
    ridge_regression,
)
from transformers_gradients.lib_types import (
    FusionGradConfig,
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


def normalise_scores(func):
    from transformers_gradients.functions import normalize_sum_to_1
    from transformers_gradients import config

    @wraps(func)
    def wrapper(*args, **kwargs):
        scores = func(*args, **kwargs)
        if config.normalize_scores:
            scores = normalize_sum_to_1(scores)
        return scores

    return wrapper


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
            Explanation(
                tokens=tuple(tokenizer.convert_ids_to_tokens(list(i))), scores=j
            )  # type: ignore
            for i, j in zip(input_ids, scores)
        ]

    return wrapper


# ----------------------------------------------------------------------------


@plain_text_inputs
@tensor_inputs
@normalise_scores
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
@normalise_scores
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
@normalise_scores
def integrated_gradients(
    model: TFPreTrainedModel,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    attention_mask: tf.Tensor | None = None,
    *,
    num_steps: int = 10,
    baseline_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
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
        size=min(len(x_batch), num_steps + 1),
        dtype=interpolated_embeddings.dtype,
        clear_after_read=True,
    )

    iterate_over_interpolation_samples = len(x_batch) >= num_steps

    if not iterate_over_interpolation_samples:
        interpolated_embeddings = tf.transpose(interpolated_embeddings, [1, 0, 2, 3])

    iterator = (
        tf.range(num_steps)
        if iterate_over_interpolation_samples
        else tf.range(len(x_batch))
    )

    for i in iterator:
        x = interpolated_embeddings[i]
        if iterate_over_interpolation_samples:
            am = attention_mask
        else:
            am = tf.repeat(
                tf.expand_dims(attention_mask[i], 0), (num_steps + 1), axis=0
            )

        with tf.GradientTape() as tape:
            tape.watch(x)
            logits = model(
                None,
                inputs_embeds=x,
                training=False,
                attention_mask=am,
            ).logits
            if iterate_over_interpolation_samples:
                logits = logits_for_labels(logits, y_batch)
            else:
                logits = logits[:, y_batch[i]]

        grads = tape.gradient(logits, x)

        interpolated_grads = interpolated_grads.write(i, grads)

    interpolated_grads_tensor = interpolated_grads.stack()
    interpolated_grads.mark_used()
    interpolated_grads.close()

    integration_axis = 0 if iterate_over_interpolation_samples else 1

    return tf.math.reduce_sum(
        tfp.math.trapz(interpolated_grads_tensor, axis=integration_axis)
        * (x_batch - baseline),
        axis=-1,
    )


@plain_text_inputs
@tensor_inputs
@normalise_scores
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
    explanations_array.mark_used()
    explanations_array.close()
    return scores


@plain_text_inputs
@tensor_inputs
@normalise_scores
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
    explanations_array.mark_used()
    explanations_array.close()
    model.set_weights(original_weights)
    return scores


@plain_text_inputs
@tensor_inputs
@normalise_scores
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
    from transformers_gradients import config as lib_config
    from transformers_gradients.functions import normalize_sum_to_1

    config = mapping_to_config(config, LimeConfig)
    config = value_or_default(config, lambda: LimeConfig())
    distance_scale = tf.constant(config.distance_scale)
    mask_token_id = tokenizer.convert_tokens_to_ids(config.mask_token)

    num_samples = tf.constant(config.num_samples)

    encoded_inputs = tokenizer(x_batch, return_tensors="tf", padding="longest").data

    def sample_masks(num_features: int):
        with tf.name_scope("sample_masks"):
            positions = tf.tile(
                tf.expand_dims(tf.range(num_features, dtype=tf.int32), 0),
                (num_samples, 1),
            )
            permutations = tf.vectorized_map(tf.random.shuffle, positions)
            num_disabled_features = tf.random.uniform(
                minval=1,
                maxval=num_features + 1,
                shape=tf.shape(positions),
                dtype=tf.int32,
            )
            return tf.math.greater_equal(permutations, num_disabled_features)

    def mask_tokens(token_ids: tf.Tensor, mmasks: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("mask_tokens"):
            ids_batch = tf.repeat(
                tf.expand_dims(token_ids, 0), tf.shape(mmasks)[0], axis=0
            )
            mmasks = tf.cast(mmasks, tf.int32)
            return (ids_batch * (tf.ones_like(mmasks) - mmasks)) + (
                mmasks * mask_token_id
            )

    scores_array = tf.TensorArray(
        dtype=tf.float32, size=len(x_batch), clear_after_read=True
    )

    for i, y in enumerate(y_batch):
        ids = encoded_inputs["input_ids"][i]
        masks = sample_masks(num_samples - 1)
        if masks.shape[0] != num_samples - 1:
            raise ValueError("Expected num_samples + 1 masks.")

        all_true_mask = tf.ones_like(masks[0], dtype=tf.bool)
        masks = tf.concat([tf.expand_dims(all_true_mask, 0), masks], axis=0)

        perturbations = mask_tokens(ids, masks)

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
        scores_array = scores_array.write(i, score)

    scores_batch = scores_array.concat()
    scores_array.mark_used()
    scores_array.close()

    if lib_config.normalize_scores:
        scores_batch = normalize_sum_to_1(scores_batch)
    if lib_config.return_raw_scores:
        return scores_batch

    a_batch = [
        Explanation(tokens=tuple(tokenizer.convert_ids_to_tokens(i)), scores=j)
        for i, j in zip(encoded_inputs["input_ids"], scores_batch)
    ]

    return a_batch
