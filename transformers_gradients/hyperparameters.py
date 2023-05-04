from __future__ import annotations

from typing import List, Tuple
from tqdm.auto import tqdm
import logging
from functools import wraps
import tensorflow as tf
from tensorflow_probability.python.distributions.normal import Normal
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)


def return_float(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return float(func(*args, **kwargs))

    return wrapper


@return_float
def recommend_noise_grad_std_dev(
    model: TFPreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    x_batch: List[str],
    y_batch: tf.Tensor,
    batch_size: int = 64,
) -> float:
    """
    For NoiseGrad it is recommended to set the relative accuracy drop to around 5%.

    References
    -------
    - Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne, 2021, NoiseGrad: enhancing explanations by introducing stochasticity to model weights, https://arxiv.org/abs/2106.10185


    Parameters
    ----------
    model:
        Model, which you want to use with NoiseGrad.
    tokenizer:
        Corresponding tokenizer.
    x_batch:
        Batch of plain-text inputs.
    y_batch:
        Batch of ground truth labels.
    batch_size:
        Batch size to use for keras.engine's Model.predict.

    Returns
    -------

    σ_NG: float
        Recommended std dev for NoiseGrad.

    """

    encoded_inputs = (tokenizer(x_batch, padding="longest", return_tensors="tf"),)
    og_predicted = tf.argmax(
        tf.nn.softmax(
            model.predict(
                encoded_inputs,
                batch_size=batch_size,
                verbose=0,
            ).logits
        ),
        axis=-1,
    )
    og_accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_batch, og_predicted)
    og_weights = model.get_weights().copy()
    max_val = get_signal_level(model)
    amplitudes = tf.linspace(0.0, max_val, 100)

    last_accuracy_drop = tf.constant(0)

    for i, std_dev in enumerate(tqdm(amplitudes)):  # noqa
        noise_distribution = Normal(loc=0, scale=std_dev)
        model.set_weights(
            tf.nest.map_structure(
                lambda w: w * noise_distribution.sample(tf.shape(w)),
                model.get_weights(),
            )
        )
        y_pred = tf.argmax(
            tf.nn.softmax(
                model.predict(
                    encoded_inputs,
                    batch_size=batch_size,
                    verbose=0,
                ).logits
            ),
            axis=-1,
        )
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_batch, y_pred)
        accuracy_drop = og_accuracy - accuracy
        log.info(f"Std dev = {std_dev}, accuracy drop = {accuracy_drop}")
        if accuracy_drop > 0.05:
            model.set_weights(og_weights)
            if tf.abs(last_accuracy_drop) - 0.05 < tf.abs(accuracy_drop - 0.05):
                return amplitudes[i - 1]
            else:
                return amplitudes[i]

    model.set_weights(og_weights)
    raise ValueError


@return_float
def recommend_smooth_grad_std_dev(
    model: TFPreTrainedModel, scale: float = 0.1
) -> float:
    """
    For SmoothGrad 10%–20%, compared to the signal level are recommended.

    References
    -------
    - Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne, 2021, NoiseGrad: enhancing explanations by introducing stochasticity to model weights, https://arxiv.org/abs/2106.10185



    Parameters
    ----------
    model:
        Model, which you want to use with NoiseGrad.
    scale:
        percentage of signal to use, default 10%.

    Returns
    -------

    σ_SG: float
        Recommended std dev for SmoothGrad.

    """
    return get_signal_level(model) * scale


def recommend_noise_grad_plus_plus_std_dev(
    model: TFPreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    x_batch: List[str],
    y_batch: tf.Tensor,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """
    For FusionGrad, it is recommended to half both σ_SG and σ_NG.

    References
    -------
    - Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne, 2021, NoiseGrad: enhancing explanations by introducing stochasticity to model weights, https://arxiv.org/abs/2106.10185

    Parameters
    ----------
    model:
        Model, which you want to use with NoiseGrad.
    tokenizer:
        Corresponding tokenizer.
    x_batch:
        Batch of plain-text inputs.
    y_batch:
        Batch of ground truth labels.
    batch_size:
        Batch size to use for keras.engine's Model.predict.

    Returns
    -------

    σ_NG: float
        Recommended std dev for NoiseGrad component.
    σ_SG: float
        Recommended std dev for SmoothGrad component.

    """
    sigma_ng = recommend_noise_grad_std_dev(
        model, tokenizer, x_batch, y_batch, batch_size
    )
    sigma_sg = recommend_smooth_grad_std_dev(model)

    return sigma_ng, sigma_sg


def get_signal_level(model: TFPreTrainedModel) -> float:
    embedding_matrix = model.get_input_embeddings().weight
    return tf.reduce_max(tf.abs(embedding_matrix))
