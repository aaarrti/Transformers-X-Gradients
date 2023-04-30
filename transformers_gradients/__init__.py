import logging
import tensorflow as tf
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
    NoiseGradPlusPlusConfig,
)
from transformers_gradients.plotting import html_heatmap
from transformers_gradients.api import text_classification
from transformers_gradients.functions import normalize_sum_to_1
from transformers_gradients.utils import is_xla_compatible_platform

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


update_config()


if is_xla_compatible_platform():
    tf.config.optimizer.set_jit("autoclustering")
# tf.config.optimizer.set_experimental_options(
#    dict(
#        layout_optimizer=True,
#        constant_folding=True,
#        shape_optimization=True,
#        remapping=True,
#        arithmetic_optimization=True,
#        dependency_optimization=True,
#        loop_optimization=True,
#        function_optimization=True,
#        debug_stripper=True,
#        scoped_allocator_optimization=True,
#        # this one breaks
#        # pin_to_host_optimization=True,
#        implementation_selector=True,
#    )
# )


def enable_mixed_precision():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        supports_mixed_precision = False
        cc = gpu_details.get("compute_capability")
        if cc:
            supports_mixed_precision = cc >= (7, 0)

        if supports_mixed_precision:
            from keras import mixed_precision

            log.info("Enabled mixed precision.")
            mixed_precision.set_global_policy("mixed_float16")
            # tf.config.optimizer.set_experimental_options(
            #    dict(auto_mixed_precision=True)
            # )


enable_mixed_precision()
