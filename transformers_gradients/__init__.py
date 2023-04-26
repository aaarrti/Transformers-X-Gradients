from transformers_gradients.lib_types import (
    IntGradConfig,
    SmoothGradConfing,
    NoiseGradConfig,
    NoiseGradPlusPlusConfig,
    LibConfig,
    LimeConfig,
    BaselineFn,
    Explanation,
    ExplainFn,
    ApplyNoiseFn,
    PlottingConfig,
)
from transformers_gradients.plotting import html_heatmap
from transformers_gradients.api import text_classification
from transformers_gradients.functions import normalize_sum_to_1

config = LibConfig()  # type: ignore


def update_config(**kwargs):
    import tensorflow as tf
    import numpy as np
    import logging

    global config

    values = config.dict()
    values.update(kwargs)

    config = LibConfig(**values)
    tf.random.set_seed(config.prng_seed)
    np.random.seed(config.prng_seed)

    logging.basicConfig(
        format=config.log_format, level=logging.getLevelName(config.log_level)
    )
