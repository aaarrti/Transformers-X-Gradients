from functools import partial

import numpy as np
import pytest
import tensorflow as tf
from datasets import load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from transformers_gradients import (
    IntGradConfig,
    NoiseGradConfig,
    NoiseGradPlusPlusConfig,
    SmoothGradConfing,
    LimeConfig,
    text_classification,
)
from transformers_gradients.utils.util import encode_inputs


from tests.markers import skip_in_ci


BATCH_SIZE = 8


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
    x_batch = dataset["sentence"][:BATCH_SIZE]
    y_batch = dataset["label"][:BATCH_SIZE]
    return x_batch, tf.constant(y_batch)


@pytest.fixture(scope="session")
def sst2_batch_embeddings(sst2_batch, sst2_model, sst2_tokenizer):
    x_batch = sst2_batch[0]
    input_ids, predict_kwargs = encode_inputs(sst2_tokenizer, x_batch)
    x_embeddings = sst2_model.get_input_embeddings()(input_ids)
    return x_embeddings, sst2_batch[1], predict_kwargs


@pytest.mark.parametrize(
    "func",
    [
        text_classification.gradient_norm,
        text_classification.gradient_x_input,
        partial(
            text_classification.integrated_gradients,
            config=IntGradConfig(
                batch_interpolated_inputs=False,
            ),
        ),
        pytest.param(
            text_classification.integrated_gradients,
            marks=skip_in_ci,
        ),
        partial(
            text_classification.smooth_grad,
            config=SmoothGradConfing(n=2, explain_fn="GradNorm"),
        ),
        partial(
            text_classification.noise_grad,
            config=NoiseGradConfig(n=2, explain_fn="GradNorm"),
        ),
        partial(
            text_classification.noise_grad_plus_plus,
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
    explanations = func(sst2_model, *sst2_batch, tokenizer=sst2_tokenizer)
    assert len(explanations) == BATCH_SIZE
    for t, s in explanations:
        assert isinstance(t, list)
        assert [isinstance(i, str) for i in t]
        assert isinstance(s, tf.Tensor)
        assert not np.isnan(s).any()


@pytest.mark.parametrize(
    "func",
    [
        text_classification.gradient_norm,
        text_classification.gradient_x_input,
        partial(
            text_classification.integrated_gradients,
            config=IntGradConfig(
                batch_interpolated_inputs=False,
            ),
        ),
        pytest.param(
            text_classification.integrated_gradients,
            marks=skip_in_ci,
        ),
        partial(
            text_classification.smooth_grad,
            config=SmoothGradConfing(n=2, explain_fn="GradNorm"),
        ),
        partial(
            text_classification.noise_grad,
            config=NoiseGradConfig(n=2, explain_fn="GradNorm"),
        ),
        partial(
            text_classification.noise_grad_plus_plus,
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
    explanations = func(sst2_model, *sst2_batch_embeddings, tokenizer=sst2_tokenizer)
    assert len(explanations) == BATCH_SIZE
    for s in explanations:
        assert isinstance(s, tf.Tensor)
        assert not np.isnan(s).any()


def test_lime_huggingface_model(sst2_model, sst2_batch, sst2_tokenizer):
    explanations = text_classification.lime(
        sst2_model,
        *sst2_batch,
        tokenizer=sst2_tokenizer,
        config=LimeConfig(num_samples=10),
    )
    assert len(explanations) == BATCH_SIZE
    for t, s in explanations:
        assert isinstance(t, list)
        assert [isinstance(i, str) for i in t]
        assert isinstance(s, tf.Tensor)
        assert not np.isnan(s).any()
