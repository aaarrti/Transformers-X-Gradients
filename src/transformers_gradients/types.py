from __future__ import annotations

from typing import (
    Callable,
    Protocol,
    overload,
    runtime_checkable,
    Tuple,
    List,
    Literal,
    NamedTuple,
)

import tensorflow as tf
from tensorflow.python.trackable.data_structures import ListWrapper
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

BaselineFn = Callable[[tf.Tensor], tf.Tensor]
Explanation = Tuple[List[str], tf.Tensor]
ApplyNoiseFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
BaselineExplainFn = Literal["GradNorm", "GradXInput", "IntGrad"]
DistanceFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
KernelFn = Callable[[tf.Tensor], tf.Tensor]


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


@runtime_checkable
class ExplainFn(Protocol):
    @overload
    def __call__(
        self,
        model: TFPreTrainedModel,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        attention_mask: tf.Tensor | None,
        tokenizer: None = None,
        *args,
        **kwargs,
    ) -> tf.Tensor:
        ...

    @overload
    def __call__(
        self,
        model: TFPreTrainedModel,
        x_batch: List[str],
        y_batch: tf.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        *args,
        **kwargs,
    ) -> List[Explanation]:
        ...

    def __call__(  # type: ignore
        self,
        model: TFPreTrainedModel,
        x_batch: List[str] | tf.Tensor,
        y_batch: tf.Tensor,
        tokenizer: PreTrainedTokenizerBase | None,
        *args,
        **kwargs,
    ) -> List[Explanation] | tf.Tensor:
        ...


class LibConfig(NamedTuple):
    prng_seed: int = 42
    log_level: str = "INFO"
    log_format: str = "%(asctime)s:[%(filename)s:%(lineno)s->%(funcName)s()]:%(levelname)s: %(message)s"


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


class LimeConfig(tf.experimental.ExtensionType):
    alpha: float = 1.0
    solver: str = "cholesky"
    num_samples: int = 1000
    mask_token: str = "[UNK]"
    distance_scale: float = 100.0
    batch_size: int = 256
