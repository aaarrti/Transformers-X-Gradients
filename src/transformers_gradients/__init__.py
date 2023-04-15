from transformers_gradients.types import (
    IntGradConfig,
    SmoothGradConfing,
    NoiseGradConfig,
    NoiseGradPlusPlusConfig,
    LibConfig,
)
from transformers_gradients.types import (
    BaselineFn,
    Explanation,
    ExplainFn,
    ApplyNoiseFn,
)


def update_config(**kwargs):
    import tensorflow as tf
    import numpy as np
    import logging

    config = LibConfig()
    tf.random.set_seed(config.prng_seed)
    np.random.seed(config.prng_seed)

    logging.basicConfig(
        format=config.log_format, level=logging.getLevelName(config.log_level)
    )


update_config()
