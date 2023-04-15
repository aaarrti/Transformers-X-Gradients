from __future__ import annotations

import tempfile
from functools import wraps, partial
from typing import Protocol, runtime_checkable, Mapping, Callable, List

import tensorflow as tf
import tensorflow_probability as tfp
from absl import logging as log
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
)
from tensorflow.python.trackable.data_structures import ListWrapper
from tensorflow.python.types.core import GenericFunction
from tensorflow_probability.python.distributions.normal import Normal
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase


from transformers_gradients.config import (
    SmoothGradConfing,
    IntGradConfig,
    NoiseGradConfig,
    NoiseGradPlusPlusConfig,
    ModelConfig,
)
from transformers_gradients.util import (
    logits_for_labels,
    as_tensor,
    bounding_shape,
    value_or_default,
    interpolate_inputs,
    zeros_baseline,
    multiplicative_noise,
    is_xla_compatible_platform,
    get_input_ids,
)


@runtime_checkable
class ModelFn(Protocol):
    def __call__(
        self, *, inputs_embeds: tf.Tensor, attention_mask: tf.Tensor
    ) -> Mapping[str, tf.Tensor]:
        ...


@runtime_checkable
class UserObject(Protocol):
    signatures: Mapping[str, ModelFn]
    variables: ListWrapper[tf.Variable]


def tensor_inputs(func):
    @wraps(func)
    def wrapper(
        model: UserObject,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        **kwargs,
    ):
        x_batch = as_tensor(x_batch)
        y_batch = as_tensor(y_batch)
        attention_mask = value_or_default(
            attention_mask, lambda: tf.ones(bounding_shape(x_batch), dtype=tf.int32)
        )
        attention_mask = as_tensor(attention_mask)
        return func(model, x_batch, y_batch, attention_mask, **kwargs)

    return wrapper


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

        input_ids, predict_kwargs = get_input_ids(tokenizer, x_batch)
        attention_mask = predict_kwargs.get("attention_mask")
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
    baseline_fn=None,
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
    explain_fn="IntGrad",
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
    explain_fn="IntGrad",
    noise_fn=None,
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
    explain_fn="IntGrad",
    noise_fn=None,
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


# ----------------------------------------------------------------------


def convert_graph_to_tensor_rt(
    model: TFPreTrainedModel, fallback_to_saved_model: bool
) -> GenericFunction:
    with tempfile.TemporaryDirectory() as tmpdir:
        tf.saved_model.save(model, f"{tmpdir}/saved_model")

        try:
            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=f"{tmpdir}/saved_model", use_dynamic_shape=True
            )
            converter.convert()
            converter.save(f"{tmpdir}/tensor_rt")
            tensor_rt_func = tf.saved_model.load(f"{tmpdir}/tensor_rt")
            return tensor_rt_func
        except RuntimeError as e:
            if not fallback_to_saved_model:
                raise e
            log.error(
                f"Failed to convert model to TensoRT: {e}, falling back to TF saved model."
            )
            return tf.saved_model.load(f"{tmpdir}/saved_model")


def build_embeddings_model(
    hf_model: TFPreTrainedModel, config: ModelConfig | None = None
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(
        shape=[None, 768], dtype=tf.float32, name="inputs_embeds"
    )
    mask_in = tf.keras.layers.Input(shape=[None], dtype=tf.int32, name="attention_mask")

    if config is None:
        model_family = hf_model.base_model_prefix
        embeddings_dim = getattr(hf_model, model_family).embeddings.dim
        num_hidden_layers = getattr(hf_model, model_family).num_hidden_layers
        config = ModelConfig(model_family, num_hidden_layers, embeddings_dim)

    distilbert_output = getattr(hf_model, config.model_family).transformer(
        inputs,
        mask_in,
        [None] * config.num_hidden_layers,
        False,
        False,
        False,
    )
    hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
    pooled_output = hidden_state[:, 0]  # (bs, dim)
    pooled_output = hf_model.pre_classifier(pooled_output)  # (bs, dim)
    pooled_output = hf_model.dropout(pooled_output, training=False)  # (bs, dim)
    logits = hf_model.classifier(pooled_output)  # (bs, dim)

    new_model = tf.keras.Model(
        inputs={"inputs_embeds": inputs, "attention_mask": mask_in}, outputs=[logits]
    )
    # Build graph
    new_model(
        {
            "inputs_embeds": tf.random.uniform([8, 10, config.embeddings_dim]),
            "attention_mask": tf.ones([8, 10], dtype=tf.int32),
        }
    )
    return new_model


# ---------------------------------------------------------------


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


@tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
def pseudo_interpolate(x, num_steps):
    og_shape = tf.convert_to_tensor(tf.shape(x))
    new_shape = tf.concat([[num_steps + 1], og_shape], axis=0)
    x = tf.broadcast_to(x, new_shape)
    flat_shape = tf.concat([tf.constant([-1]), og_shape[1:]], axis=0)
    x = tf.reshape(x, flat_shape)
    return x
