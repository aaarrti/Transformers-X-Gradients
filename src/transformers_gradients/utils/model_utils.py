import logging
import tempfile

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from transformers import TFPreTrainedModel

from transformers_gradients.types import UserObject

log = logging.getLogger(__name__)


def convert_graph_to_tensor_rt(
    model: TFPreTrainedModel, fallback_to_saved_model: bool = False
) -> UserObject:
    with tempfile.TemporaryDirectory() as tmpdir:
        tf.saved_model.save(model, f"{tmpdir}/saved_model")

        try:
            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=f"{tmpdir}/saved_model", use_dynamic_shape=True
            )
            converter.convert()
            converter.save(f"{tmpdir}/tensor_rt")
            tensor_rt_func = tf.saved_model.load(f"{tmpdir}/tensor_rt")
            return tensor_rt_func
        except RuntimeError as e:
            if not fallback_to_saved_model:
                raise e
            log.error(
                f"Failed to convert model to TensoRT: {e}, falling back to TF saved model."
            )
            return tf.saved_model.load(f"{tmpdir}/saved_model")
