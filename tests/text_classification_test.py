from functools import partial

import pytest
import tensorflow as tf
from datasets import load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from transformers_gradients import (
    NoiseGradConfig,
    FusionGradConfig,
    SmoothGradConfing,
    LimeConfig,
    text_classification,
    is_xla_compatible_platform,
)
from transformers_gradients.lib_types import ExplainFn
from transformers_gradients.utils import encode_inputs

BATCH_SIZE = 64


@pytest.fixture(scope="session")
def sst2_model():
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model._jit_compile = is_xla_compatible_platform()
    model.__call__ = tf.function(
        model.__call__, reduce_retracing=True, jit_compile=is_xla_compatible_platform()
    )
    return model


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


# -----------------------------------------------------------------------


text_classification_tests = pytest.mark.parametrize(
    "func",
    [
        text_classification.gradient_norm,
        text_classification.gradient_x_input,
        text_classification.integrated_gradients,
        # ---
        partial(
            text_classification.smooth_grad,
            config=SmoothGradConfing(n=2, explain_fn="GradNorm"),
        ),
        partial(
            text_classification.smooth_grad,
            config=SmoothGradConfing(n=2, explain_fn="GradXInput"),
        ),
        partial(text_classification.smooth_grad, config=SmoothGradConfing(n=2)),
        # ---
        partial(
            text_classification.noise_grad,
            config=NoiseGradConfig(n=2, explain_fn="GradNorm"),
        ),
        partial(
            text_classification.noise_grad,
            config=NoiseGradConfig(n=2, explain_fn="GradXInput"),
        ),
        partial(
            text_classification.noise_grad,
            config=NoiseGradConfig(n=2),
        ),
        # ---
        partial(
            text_classification.fusion_grad,
            config=FusionGradConfig(n=2, m=2, explain_fn="GradNorm"),
        ),
        partial(
            text_classification.fusion_grad,
            config=FusionGradConfig(n=2, m=2, explain_fn="GradXInput"),
        ),
        partial(text_classification.fusion_grad, config=FusionGradConfig(n=2, m=2)),
    ],
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad",
        # ---
        "SmoothGrad_GradNorm",
        "SmoothGrad_GradXInput",
        "SmoothGrad_IntGrad",
        # ---
        "NoiseGrad_GradNorm",
        "NoiseGrad_GradXInput",
        "NoiseGrad_IntGrad",
        # ---
        "FusionGrad_GradNorm",
        "FusionGrad_GradXInput",
        "FusionGrad_IntGrad",
    ],
)


@text_classification_tests
def test_plain_text(func: ExplainFn, sst2_model, sst2_batch, sst2_tokenizer):
    explanations = func(sst2_model, *sst2_batch, tokenizer=sst2_tokenizer)
    assert len(explanations) == BATCH_SIZE
    for e in explanations:
        assert isinstance(e.tokens, tuple)
        assert [isinstance(i, str) for i in e.tokens]
        assert isinstance(e.scores, tf.Tensor)
        tf.debugging.check_numerics(e.scores, "NaNs not allowed")


@text_classification_tests
def test_embeddings(func: ExplainFn, sst2_model, sst2_batch_embeddings, sst2_tokenizer):
    explanations = func(
        sst2_model,
        sst2_batch_embeddings[0],
        sst2_batch_embeddings[1],
        attention_mask=sst2_batch_embeddings[2],
        tokenizer=sst2_tokenizer,
    )
    assert len(explanations) == BATCH_SIZE
    assert isinstance(explanations, tf.Tensor)
    tf.debugging.check_numerics(explanations, "NaNs not allowed")


def test_lime(sst2_model, sst2_batch, sst2_tokenizer):
    explanations = text_classification.lime(
        sst2_model,
        *sst2_batch,
        tokenizer=sst2_tokenizer,
        config=LimeConfig(num_samples=10),
    )
    assert len(explanations) == BATCH_SIZE
    for e in explanations:
        assert isinstance(e.tokens, tuple)
        assert [isinstance(i, str) for i in e.tokens]
        assert isinstance(e.scores, tf.Tensor)
        tf.debugging.check_numerics(explanations, "NaNs not allowed")
