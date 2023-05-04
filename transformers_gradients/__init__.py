import logging
import tensorflow as tf
from transformers_gradients.lib_types import (
    SmoothGradConfing,
    NoiseGradConfig,
    FusionGradConfig,
    LibConfig,
    LimeConfig,
    Explanation,
    PlottingConfig,
    NoiseGradPlusPlusConfig,
)
from transformers_gradients.plotting import html_heatmap
from transformers_gradients.api import text_classification
from transformers_gradients.functions import normalize_sum_to_1
from transformers_gradients.utils import (
    is_xla_compatible_platform,
    is_mixed_precision_supported_device,
)

log = logging.getLogger(__name__)
config = LibConfig()  # type: ignore


def update_config(
    *,
    prng_seed: int = 42,
    log_level: str = "DEBUG",
    log_format: str = "%(asctime)s:[%(filename)s:%(lineno)s->%(funcName)s()]:%(levelname)s: %(message)s",
    return_raw_scores: bool = False,
    normalize_scores: bool = False,
    run_with_profiler: bool = False,
    enable_mixed_precision: bool = False,
):
    global config

    config = LibConfig(
        prng_seed=prng_seed,
        log_format=log_format,
        log_level=log_level,
        return_raw_scores=return_raw_scores,
        normalize_scores=normalize_scores,
        run_with_profiler=run_with_profiler,
        enable_mixed_precision=enable_mixed_precision,
    )
    tf.random.set_seed(config.prng_seed)
    tf.experimental.numpy.random.seed(config.prng_seed)

    logging.basicConfig(
        format=config.log_format, level=logging.getLevelName(config.log_level)
    )


update_config()

if is_xla_compatible_platform():
    tf.config.optimizer.set_jit("autoclustering")


# if is_mixed_precision_supported_device() and not config.enable_mixed_precision:
#     from keras import mixed_precision
#
#     log.info("Enabled mixed precision.")
#     mixed_precision.set_global_policy("mixed_float16")

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
