from functools import partial
from os import environ

import numpy as np
import pytest
import tensorflow as tf
from datasets import load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from transformers_gradients.text_classification.explanation_func import (
    gradient_norm,
    gradient_x_input,
    integrated_gradients,
    smooth_grad,
    noise_grad,
    noise_grad_plus_plus,
)
from transformers_gradients.config import (
    IntGradConfig,
    NoiseGradConfig,
    NoiseGradPlusPlusConfig,
    SmoothGradConfing,
)
from transformers_gradients.util import is_xla_compatible_platform, get_input_ids


# @pytest.fixture(scope="session", autouse=True)
# def profile():
#    options = tf.profiler.experimental.ProfilerOptions(
#        host_tracer_level=3, python_tracer_level=1, device_tracer_level=1
#    )
#    tf.profiler.experimental.start("profile_logs", options=options)
#
#    yield
#
#    tf.profiler.experimental.stop()


skip_in_ci = pytest.mark.skipif("CI" in environ, reason="OOM in GitHub action.")


@pytest.fixture(scope="session")
def sst2_model():
    return TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )


@pytest.fixture(scope="session")
def sst2_tokenizer():
    return AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )


def unk_token_baseline_func():
    unknown_token = tf.constant(np.load("tests/assets/unknown_token_embedding.npy"))

    @tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
    def unk_token_baseline(x):
        return unknown_token

    return unk_token_baseline


@pytest.fixture(scope="session")
def sst2_batch():
    dataset = load_dataset("sst2")["train"]
    x_batch = dataset["sentence"][:32]
    y_batch = dataset["label"][:32]
    return x_batch, tf.constant(y_batch)


@pytest.fixture(scope="session")
def sst2_batch_embeddings(sst2_batch, sst2_model, sst2_tokenizer):
    x_batch = sst2_batch[0]
    input_ids, predict_kwargs = get_input_ids(sst2_tokenizer, x_batch)
    x_embeddings = sst2_model.get_input_embeddings()(input_ids)
    return x_embeddings, sst2_batch[1], predict_kwargs


@pytest.mark.parametrize(
    "func",
    [
        gradient_norm,
        gradient_x_input,
        partial(
            integrated_gradients,
            config=IntGradConfig(
                batch_interpolated_inputs=False,
                baseline_fn=lambda x: tf.zeros_like(x, dtype=x.dtype),
            ),
        ),
        pytest.param(
            partial(
                integrated_gradients,
                config=IntGradConfig(baseline_fn=unk_token_baseline_func()),
            ),
            marks=skip_in_ci,
        ),
        partial(smooth_grad, config=SmoothGradConfing(n=2, explain_fn="GradNorm")),
        partial(noise_grad, config=NoiseGradConfig(n=2, explain_fn="GradNorm")),
        partial(
            noise_grad_plus_plus,
            config=NoiseGradPlusPlusConfig(n=2, m=2, explain_fn="GradNorm"),
        ),
    ],
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad_iterative",
        "IntGrad_batched",
        "SmoothGrad",
        "NoiseGrad",
        "NoiseGrad++",
    ],
)
def test_explain_on_text(func, sst2_model, sst2_batch, sst2_tokenizer):
    explanations = func(sst2_model, *sst2_batch, sst2_tokenizer)
    assert len(explanations) == 32
    for t, s in explanations:
        assert isinstance(t, list)
        assert [isinstance(i, str) for i in t]
        assert isinstance(s, tf.Tensor)
        assert not np.isnan(s).any()


@pytest.mark.parametrize(
    "func",
    [
        gradient_norm,
        gradient_x_input,
        partial(
            integrated_gradients,
            config=IntGradConfig(
                batch_interpolated_inputs=False,
                baseline_fn=lambda x: tf.zeros_like(x, dtype=x.dtype),
            ),
        ),
        pytest.param(
            partial(
                integrated_gradients,
                config=IntGradConfig(baseline_fn=unk_token_baseline_func()),
            ),
            marks=skip_in_ci,
        ),
        partial(smooth_grad, config=SmoothGradConfing(n=2, explain_fn="GradNorm")),
        partial(noise_grad, config=NoiseGradConfig(n=2, explain_fn="GradNorm")),
        partial(
            noise_grad_plus_plus,
            config=NoiseGradPlusPlusConfig(n=2, m=2, explain_fn="GradNorm"),
        ),
    ],
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad_iterative",
        "IntGrad_batched",
        "SmoothGrad",
        "NoiseGrad",
        "NoiseGrad++",
    ],
)
def test_explain_on_embeddings(func, sst2_model, sst2_batch_embeddings, sst2_tokenizer):
    explanations = func(
        sst2_model,
        sst2_batch_embeddings[0],
        sst2_batch_embeddings[1],
        sst2_tokenizer,
        **sst2_batch_embeddings[2],
    )
    assert len(explanations) == 32
    for s in explanations:
        assert isinstance(s, tf.Tensor)
        assert not np.isnan(s).any()
