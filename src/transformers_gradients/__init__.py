from transformers_gradients.types import (
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
)
from transformers_gradients.plotting import visualise_explanations_as_html
from transformers_gradients.api import text_classification

config = LibConfig()


def update_config(**kwargs):
    import tensorflow as tf
    import numpy as np
    import logging

    global config

    config = LibConfig()
    tf.random.set_seed(config.prng_seed)
    np.random.seed(config.prng_seed)

    logging.basicConfig(
        format=config.log_format, level=logging.getLevelName(config.log_level)
    )


def get_config() -> LibConfig:
    global config
    return config


update_config()
