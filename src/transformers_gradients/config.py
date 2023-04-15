from typing import NamedTuple

import tensorflow as tf


class LibConfig(NamedTuple):
    seed: int = 42
    log_level: str = "INFO"


class ModelConfig(tf.experimental.ExtensionType):
    model_family: str
    num_hidden_layers: int
    embeddings_dim: int


class IntGradConfig(tf.experimental.ExtensionType):
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
    batch_interpolated_inputs: bool = True


class NoiseGradConfig(tf.experimental.ExtensionType):
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
    std: float = 0.0055


class SmoothGradConfing(tf.experimental.ExtensionType):
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
    std: float = 0.0055


class NoiseGradPlusPlusConfig(tf.experimental.ExtensionType):
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
    std: float = 0.0055
    sg_std: float = 0.05


def update_config(**kwargs):
    config = LibConfig()
    tf.random.set_seed(config.seed)
