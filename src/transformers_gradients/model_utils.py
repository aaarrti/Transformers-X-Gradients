import logging
import tempfile

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from transformers import TFPreTrainedModel

from transformers_gradients.types import ModelConfig, UserObject

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


def build_embeddings_model(
    hf_model: TFPreTrainedModel, config: ModelConfig | None = None
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(
        shape=[None, 768], dtype=tf.float32, name="inputs_embeds"
    )
    mask_in = tf.keras.layers.Input(shape=[None], dtype=tf.int32, name="attention_mask")

    if config is None:
        model_family = hf_model.base_model_prefix
        embeddings_dim = getattr(hf_model, model_family).embeddings.dim
        num_hidden_layers = getattr(hf_model, model_family).num_hidden_layers
        config = ModelConfig(model_family, num_hidden_layers, embeddings_dim)

    distilbert_output = getattr(hf_model, config.model_family).transformer(
        inputs,
        mask_in,
        [None] * config.num_hidden_layers,
        False,
        False,
        False,
    )
    hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
    pooled_output = hidden_state[:, 0]  # (bs, dim)
    pooled_output = hf_model.pre_classifier(pooled_output)  # (bs, dim)
    pooled_output = hf_model.dropout(pooled_output, training=False)  # (bs, dim)
    logits = hf_model.classifier(pooled_output)  # (bs, dim)

    new_model = tf.keras.Model(
        inputs={"inputs_embeds": inputs, "attention_mask": mask_in}, outputs=[logits]
    )
    # Build graph
    new_model(
        {
            "inputs_embeds": tf.random.uniform([8, 10, config.embeddings_dim]),
            "attention_mask": tf.ones([8, 10], dtype=tf.int32),
        }
    )
    return new_model
