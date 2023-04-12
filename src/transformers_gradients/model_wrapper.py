from __future__ import annotations

import tensorflow as tf
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase


class ModelWrapper(tf.Module):
    model: TFPreTrainedModel

    def __init__(self, model: TFPreTrainedModel):
        super().__init__()
        self.model = model

    def __call__(self, inputs_embeds: tf.Tensor, **kwargs) -> tf.Tensor:
        return self.model(
            None, inputs_embeds=inputs_embeds, training=False, **kwargs
        ).logits

    def embedding_lookup(self, input_ids: tf.Tensor) -> tf.Tensor:
        return self.model.get_input_embeddings()(input_ids)


class TokenizerWrapper(tf.Module):
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def convert_ids_to_tokens(self, input_ids: list[int]) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(list(input_ids))

    def batch_encode(self, x_batch: list[str]) -> dict[str, tf.Tensor]:
        return self.tokenizer(x_batch, padding="longest", return_tensors="tf").data
