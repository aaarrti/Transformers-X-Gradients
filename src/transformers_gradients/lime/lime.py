from __future__ import annotations

from typing import Callable

import tensorflow as tf

from transformers_gradients.functions import ridge_regression, sample_masks, mask_tokens
from transformers_gradients.types import KernelFn, DistanceFn


def lime(
    predict_fn: Callable[[tf.Tensor], tf.Tensor],
    x_batch: tf.TensorArray,
    y_batch: tf.Tensor,
    mask_token_id: int,
    num_samples: int,
    distance_scale: float,
    distance_fn: DistanceFn,
    kernel: KernelFn,
) -> tf.TensorArray:
    """
    LIME explains classifiers by returning a feature attribution score
    for each input feature. It works as follows:

    1) Sample perturbation masks. First the number of masked features is sampled
        (uniform, at least 1), and then that number of features are randomly chosen
        to be masked out (without replacement).
    2) Get predictions from the model for those perturbations. Use these as labels.
    3) Fit a linear model to associate the input positions indicated by the binary
        mask with the resulting predicted label.

    The resulting feature importance scores are the linear model coefficients for
    the requested output class or (in case of regression) the output score.

    This is a reimplementation of the original https://github.com/marcotcr/lime
    and is tested for compatibility. This version supports applying LIME to text input.

    Returns
    -------

    """
    distance_scale = tf.constant(distance_scale)
    mask_token_id = tf.constant(mask_token_id)

    scores = tf.TensorArray(
        x_batch.dtype,
        size=num_samples,
        colocate_with_first_write_call=True,
    )

    for i, y in enumerate(y_batch):
        ids = x_batch.read(i)
        masks = sample_masks(num_samples - 1, len(ids), seed=42)
        if masks.shape[0] != num_samples - 1:
            raise ValueError("Expected num_samples + 1 masks.")

        all_true_mask = tf.ones_like(masks[0], dtype=tf.bool)
        masks = tf.concat([tf.expand_dims(all_true_mask, 0), masks], axis=0)

        perturbations = mask_tokens(ids, masks, mask_token_id)
        logits = predict_fn(perturbations)
        outputs = logits[:, y]
        distances = distance_fn(
            tf.cast(all_true_mask, dtype=tf.float32), tf.cast(masks, dtype=tf.float32)
        )  # noqa
        distances = distance_scale * distances
        distances = kernel(distances)
        score = ridge_regression(masks, outputs, sample_weight=distances)
        scores = scores.write(i, score)
    return scores
