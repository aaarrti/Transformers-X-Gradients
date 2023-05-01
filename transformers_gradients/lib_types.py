from __future__ import annotations

from typing import (
    Protocol,
    overload,
    runtime_checkable,
    Tuple,
    List,
    Literal,
    NamedTuple,
)

import tensorflow as tf
from pydantic import BaseSettings, Field
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

ApplyNoiseFn = Literal["additive", "multiplicative"]
BaselineExplainFn = Literal["GradNorm", "GradXInput", "IntGrad"]
ColorMappingStrategy = Literal["global", "row-wise"]
RgbRange = Literal[1, 255]


class Explanation(tf.experimental.ExtensionType):
    tokens: Tuple[str, ...]
    scores: tf.Tensor


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


class LibConfig(BaseSettings):
    """
    Configuration for libraries behaviour.

    Attributes
    ----------

    prng_seed:
        Manual seed to use for PRNGs.
    log_level:
        Log level for builtin logging module.
    log_format:
        Format for log messages.
    return_raw_scores:
        If set to true, explanation functions will return scores only event for plain-text inputs.


    Examples
    ----------
    >>> from transformers_gradients import update_config
    >>> update_config(prng_seed=100)
    """

    prng_seed: int = Field(42, env="TG_PRNG_SEED")
    log_level: str = Field("DEBUG", env="TG_LOG_LEVEL")
    log_format: str = Field(
        "%(asctime)s:[%(filename)s:%(lineno)s->%(funcName)s()]:%(levelname)s: %(message)s",
        env="TG_LOG_FORMAT",
    )
    return_raw_scores: bool = Field(False, env="TG_RETURN_RAW_SCORES")
    normalize_scores: bool = Field(False, env="TG_NORMALISE_SCORES")
    run_with_profiler: bool = Field(False, env="TG_RUN_WITH_PROFILER")
    enable_mixed_precision: bool = Field(False, env="TG_DISABLE_MIXED_PRECISION")


class NoiseGradConfig(NamedTuple):
    """
    Hyper parameters for NoiseGrad.

    Attributes
    ----------

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
    """

    n: int = 10
    mean: float = 1.0
    std: float = 0.0055
    explain_fn: BaselineExplainFn | ExplainFn = "IntGrad"
    noise_fn: ApplyNoiseFn = "multiplicative"


class SmoothGradConfing(NamedTuple):
    """
    Hyper parameters for SmoothGrad.

    Attributes
    ----------
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
    """

    n: int = 10
    mean: float = 1.0
    std: float = 0.0055
    explain_fn: BaselineExplainFn | ExplainFn = "IntGrad"
    noise_fn: ApplyNoiseFn = "multiplicative"


class FusionGradConfig(NamedTuple):
    """
    Hyper parameters for FusionGrad.

    Attributes
    ----------
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
    """

    n: int = 10
    m: int = 10
    mean: float = 1.0
    sg_mean: float = 0.0
    std: float = 0.0055
    sg_std: float = 0.05
    explain_fn: BaselineExplainFn | ExplainFn = "IntGrad"
    noise_fn: ApplyNoiseFn = "multiplicative"


# Alias to FusionGrad.
NoiseGradPlusPlusConfig = FusionGradConfig


class LimeConfig(NamedTuple):
    alpha: float = 1.0
    num_samples: int = 1000
    mask_token: str = "[UNK]"
    distance_scale: float = 100.0
    batch_size: int = 256


class PlottingConfig(NamedTuple):
    """
    Configuration for plotting.

    Attributes
    ----------
    ignore_special_tokens:
        If true, values in config.special_tokens will not be rendered on the heatmap.
    return_raw_html:
        If true, will return html string, by default will try to render in Jupyter.
    special_tokens:
        List of tokens to ignore during rendering, if ignore_special_tokens=True, default=["[CLS]", "[SEP]", "[PAD]"].
    rgb_scale:
        Scaling factor for colors, default=1.
    rgb_range:
        255 or 1, as per RGC standard.
    color_mapping_strategy:
        - global: RGB space is spanned by all explanations. Use when you want to compare different XAI methods
            or hyperparameter configurations.
        - row-wise: create separate RGB space for each explanation.

    """

    ignore_special_tokens: bool = False
    return_raw_html: bool = False
    color_mapping_strategy: ColorMappingStrategy = "row-wise"
    special_tokens: List[str] = ["[CLS]", "[SEP]", "[PAD]"]
    rgb_scale: float | Tuple[float, float, float] = 1.0
    rgb_range: RgbRange = 255
