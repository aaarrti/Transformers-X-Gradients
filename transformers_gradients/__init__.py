import logging
from transformers_gradients.lib_types import (
    SmoothGradConfing,
    NoiseGradConfig,
    FusionGradConfig,
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
import tensorflow as tf
from keras import mixed_precision

log = logging.getLogger(__name__)

config = LibConfig()  # type: ignore


def update_config(**kwargs):
    global config
    import numpy as np

    values = config.dict()
    values.update(kwargs)

    config = LibConfig(**values)
    tf.random.set_seed(config.prng_seed)
    np.random.seed(config.prng_seed)

    logging.basicConfig(
        format=config.log_format, level=logging.getLevelName(config.log_level)
    )


gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    gpu_details = tf.config.experimental.get_device_details(gpus[0])
    supports_mixed_precision = False
    name = gpu_details.get("device_name", "Unknown GPU")
    cc = gpu_details.get("compute_capability")
    if cc:
        device_str = f"{name}, compute capability {cc[0]}.{cc[1]}"
        if cc >= (7, 0):
            supports_mixed_precision = True

    if supports_mixed_precision:
        log.info("Enabled mixed precision.")
        mixed_precision.set_global_policy("mixed_float16")
