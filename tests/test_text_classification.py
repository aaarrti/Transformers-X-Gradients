from functools import partial
from os import environ

import numpy as np
import pytest
import tensorflow as tf
from datasets import load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from transformers_gradients.types import (
    IntGradConfig,
    NoiseGradConfig,
    NoiseGradPlusPlusConfig,
    SmoothGradConfing,
)
from transformers_gradients.util import is_xla_compatible_platform, encode_inputs
from transformers_gradients.text_classification import tensor_rt, huggingface
from transformers_gradients.model_utils import (
    build_embeddings_model,
    convert_graph_to_tensor_rt,
)

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


@pytest.fixture(scope="session")
def sst2_batch():
    dataset = load_dataset("sst2")["train"]
    x_batch = dataset["sentence"][:32]
    y_batch = dataset["label"][:32]
    return x_batch, tf.constant(y_batch)


@pytest.fixture(scope="session")
def sst2_batch_embeddings(sst2_batch, sst2_model, sst2_tokenizer):
    x_batch = sst2_batch[0]
    input_ids, predict_kwargs = encode_inputs(sst2_tokenizer, x_batch)
    x_embeddings = sst2_model.get_input_embeddings()(input_ids)
    return x_embeddings, sst2_batch[1], predict_kwargs


@pytest.fixture(scope="session")
def sst2_saved_model(sst2_model):
    return convert_graph_to_tensor_rt(
        build_embeddings_model(sst2_model), fallback_to_saved_model=True
    )


@pytest.mark.parametrize(
    "func",
    [
        huggingface.gradient_norm,
        huggingface.gradient_x_input,
        partial(
            huggingface.integrated_gradients,
            config=IntGradConfig(
                batch_interpolated_inputs=False,
            ),
        ),
        pytest.param(
            huggingface.integrated_gradients,
            marks=skip_in_ci,
        ),
        partial(
            huggingface.smooth_grad,
            config=SmoothGradConfing(n=2),
            explain_fn="GradNorm",
        ),
        partial(
            huggingface.noise_grad, config=NoiseGradConfig(n=2), explain_fn="GradNorm"
        ),
        partial(
            huggingface.noise_grad_plus_plus,
            config=NoiseGradPlusPlusConfig(n=2, m=2),
            explain_fn="GradNorm",
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
    explanations = func(sst2_model, *sst2_batch, tokenizer=sst2_tokenizer)
    assert len(explanations) == 32
    for t, s in explanations:
        assert isinstance(t, list)
        assert [isinstance(i, str) for i in t]
        assert isinstance(s, tf.Tensor)
        assert not np.isnan(s).any()


@pytest.mark.parametrize(
    "func",
    [
        huggingface.gradient_norm,
        huggingface.gradient_x_input,
        partial(
            huggingface.integrated_gradients,
            config=IntGradConfig(
                batch_interpolated_inputs=False,
            ),
        ),
        pytest.param(
            huggingface.integrated_gradients,
            marks=skip_in_ci,
        ),
        partial(
            huggingface.smooth_grad,
            config=SmoothGradConfing(n=2),
            explain_fn="GradNorm",
        ),
        partial(
            huggingface.noise_grad, config=NoiseGradConfig(n=2), explain_fn="GradNorm"
        ),
        partial(
            huggingface.noise_grad_plus_plus,
            config=NoiseGradPlusPlusConfig(n=2, m=2),
            explain_fn="GradNorm",
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
    explanations = func(sst2_model, *sst2_batch_embeddings, tokenizer=sst2_tokenizer)
    assert len(explanations) == 32
    for s in explanations:
        assert isinstance(s, tf.Tensor)
        assert not np.isnan(s).any()


@pytest.mark.parametrize(
    "func",
    [
        tensor_rt.gradient_norm,
        tensor_rt.gradient_x_input,
        tensor_rt.integrated_gradients,
        partial(
            tensor_rt.smooth_grad, config=SmoothGradConfing(n=2), explain_fn="GradNorm"
        ),
        partial(
            tensor_rt.noise_grad, config=NoiseGradConfig(n=2), explain_fn="GradNorm"
        ),
        partial(
            tensor_rt.noise_grad_plus_plus,
            config=NoiseGradPlusPlusConfig(n=2, m=2),
            explain_fn="GradNorm",
        ),
    ],
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad",
        "SmoothGrad",
        "NoiseGrad",
        "NoiseGrad++",
    ],
)
def test_saved_model_on_embeddings(func, sst2_saved_model, sst2_batch_embeddings):
    explanations = func(
        sst2_saved_model,
        *sst2_batch_embeddings,
    )
    assert len(explanations) == 32
    for s in explanations:
        assert isinstance(s, tf.Tensor)
        assert not np.isnan(s).any()


@pytest.mark.parametrize(
    "func",
    [
        tensor_rt.gradient_norm,
        tensor_rt.gradient_x_input,
        tensor_rt.integrated_gradients,
        partial(
            tensor_rt.smooth_grad, config=SmoothGradConfing(n=2), explain_fn="GradNorm"
        ),
        partial(
            tensor_rt.noise_grad, config=NoiseGradConfig(n=2), explain_fn="GradNorm"
        ),
        partial(
            tensor_rt.noise_grad_plus_plus,
            config=NoiseGradPlusPlusConfig(n=2, m=2),
            explain_fn="GradNorm",
        ),
    ],
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad",
        "SmoothGrad",
        "NoiseGrad",
        "NoiseGrad++",
    ],
)
def test_saved_model_on_plain_text(
    func, sst2_saved_model, sst2_model, sst2_tokenizer, sst2_batch
):
    def fn(model, input_ids):
        return sst2_model.get_input_embeddings()(input_ids)

    explanations = func(
        sst2_saved_model, *sst2_batch, tokenizer=sst2_tokenizer, embeddings_lookup_fn=fn
    )
    assert len(explanations) == 32
    for t, s in explanations:
        assert isinstance(t, list)
        assert [isinstance(i, str) for i in t]
        assert isinstance(s, tf.Tensor)
        assert not np.isnan(s).any()
