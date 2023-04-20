import pytest


from transformers import TFAutoModelForDocumentQuestionAnswering, AutoTokenizer
from datasets import load_dataset


@pytest.fixture(scope="session")
def squad_tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")


@pytest.fixture(scope="session")
def squad_model():
    return TFAutoModelForDocumentQuestionAnswering.from_pretrained(
        "distilbert-base-cased-distilled-squad"
    )


@pytest.fixture(scope="session")
def squad_batch():
    dataset = load_dataset("squad")


@pytest.mark.parametrize("func", [])
def test_explain_on_plain_text(func, squad_model, squad_tokenizer):
    explanations = func(
        squad_model,
    )
