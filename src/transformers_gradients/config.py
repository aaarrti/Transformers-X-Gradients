from typing import NamedTuple, Union, Callable
import tensorflow as tf


from transformers_gradients.types import BaselineFn, ExplainFn, ApplyNoiseFn
from transformers_gradients.util import is_xla_compatible_platform


class LibConfig(NamedTuple):
    seed: int = 42
    log_level: str = "INFO"


class IntGradConfig(NamedTuple):
    """
    num_steps:
        Number of interpolated samples, which should be generated, default=10.
    baseline_fn:
        Function used to created baseline values, by default will create zeros tensor. Alternatively, e.g.,
        embedding for [UNK] token could be used.
    batch_interpolated_inputs:
        Indicates if interpolated inputs should be stacked into 1 bigger batch.
        This speeds up the explanation, however can be very memory intensive.
    """

    num_steps: int = 10
    baseline_fn: BaselineFn = tf.function(
        reduce_retracing=True, jit_compile=is_xla_compatible_platform()
    )(lambda x: tf.zeros_like(x, dtype=x.dtype))
    batch_interpolated_inputs: bool = True


class NoiseGradConfig(NamedTuple):
    """
    mean:
        Mean of normal distribution, from which noise applied to model's weights is sampled, default=1.0.
    std:
        Standard deviation of normal distribution, from which noise applied to model's weights is sampled, default=0.2.
    n:
        Number of times noise is applied to weights, default=10.
    explain_fn:
        Baseline explanation function. If string provided must be one of GradNorm, GradXInput, IntGrad, default=IntGrad.
        Passing additional kwargs is not supported, please use partial application from functools package instead.
    noise_fn:
        Function to apply noise, default=multiplication.
    seed:
        PRNG seed used for noise generating distributions.
    """

    n: int = 10
    mean: float = 1.0
    std: float = 0.2
    explain_fn: Union[ExplainFn, str] = "IntGrad"
    noise_fn: ApplyNoiseFn = tf.function(
        reduce_retracing=True, jit_compile=is_xla_compatible_platform()
    )(lambda a, b: a * b)


class SmoothGradConfing(NamedTuple):
    """
    mean:
        Mean of normal distribution, from which noise applied to input embeddings is sampled, default=0.0.
    std:
        Standard deviation of normal distribution, from which noise applied to input embeddings is sampled, default=0.4.
    n:
        Number of times noise is applied to input embeddings, default=10
    explain_fn:
        Baseline explanation function. If string provided must be one of GradNorm, GradXInput, IntGrad, default=IntGrad.
        Passing additional kwargs is not supported, please use partial application from functools package instead.
    noise_fn:
        Function to apply noise, default=multiplication.
    seed:
        PRNG seed used for noise generating distributions.
    """

    n: int = 10
    mean: float = 1.0
    std = 0.2
    explain_fn: Union[ExplainFn, str] = "IntGrad"
    noise_fn: ApplyNoiseFn = tf.function(
        reduce_retracing=True, jit_compile=is_xla_compatible_platform()
    )(lambda a, b: a * b)


class NoiseGradPlusPlusConfig(NamedTuple):
    """
    mean:
        Mean of normal distribution, from which noise applied to model's weights is sampled, default=1.0.
    sg_mean:
        Mean of normal distribution, from which noise applied to input embeddings is sampled, default=0.0.
    std:
        Standard deviation of normal distribution, from which noise applied to model's weights is sampled, default=0.2.
    sg_std:
        Standard deviation of normal distribution, from which noise applied to input embeddings is sampled, default=0.4.
    n:
        Number of times noise is applied to weights, default=10.
      m:
        Number of times noise is applied to input embeddings, default=10
    explain_fn:
        Baseline explanation function. If string provided must be one of GradNorm, GradXInput, IntGrad, default=IntGrad.
        Passing additional kwargs is not supported, please use partial application from functools package instead.
    noise_fn:
        Function to apply noise, default=multiplication.

    seed:
        PRNG seed used for noise generating distributions.
    """

    n: int = 10
    m: int = 10
    mean: float = 1.0
    sg_mean: float = 0.0
    std: float = 0.2
    sg_std: float = 0.4
    explain_fn: Union[ExplainFn, str] = "IntGrad"
    noise_fn: ApplyNoiseFn = tf.function(
        reduce_retracing=True, jit_compile=is_xla_compatible_platform()
    )(lambda a, b: a * b)


def update_config(**kwargs):
    config = LibConfig()
    tf.random.set_seed(config.seed)


def resolve_baseline_explain_fn(explain_fn):
    if isinstance(explain_fn, Callable):
        return explain_fn  # type: ignore

    if explain_fn in ("SmoothGrad", "NoiseGrad", "NoiseGrad++"):
        raise ValueError(f"Can't use NoiseGrad as baseline xai method for NoiseGrad.")

    from transformers_gradients.text_classification.explanation_func import (
        integrated_gradients,
        gradient_norm,
        gradient_x_input,
    )

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
