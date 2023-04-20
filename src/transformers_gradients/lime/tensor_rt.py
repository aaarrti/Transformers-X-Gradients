import tensorflow as tf
from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
)

from transformers_gradients.types import UserObject, LimeConfig, DistanceFn, KernelFn
from transformers_gradients.lime.lime import lime as base_lime
from transformers_gradients.functions import exponential_kernel


def lime(
    model: UserObject,
    x_batch: tf.TensorArray,
    y_batch: tf.Tensor,
    mask_token_id: int,
    config: LimeConfig = LimeConfig(),
    distance_fn: DistanceFn = tf.keras.losses.cosine_similarity,
    kernel: KernelFn = exponential_kernel,
) -> tf.TensorArray:
    def predict_fn(x):
        attention_mask = tf.ones(shape=tf.shape(x))
        return model.signatures(DEFAULT_SERVING_SIGNATURE_DEF_KEY)(x, attention_mask)

    return base_lime(
        predict_fn=predict_fn,
        kernel=kernel,
        distance_fn=distance_fn,
        num_samples=config.num_samples,
        mask_token_id=mask_token_id,
        x_batch=x_batch,
        y_batch=y_batch,
        distance_scale=config.distance_scale,
    )
