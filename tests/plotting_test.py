import pickle

import pytest
import tensorflow as tf  # noqa
import numpy as np
from transformers_gradients.lib_types import PlottingConfig
from transformers_gradients.plotting import html_heatmap, map_to_rgb


@pytest.fixture
def a_batch():
    with open("tests/data/a_batch.pickle", "rb") as f:
        data = pickle.load(f)

    return data


def test_heatmap(a_batch):
    html = html_heatmap(a_batch, config=PlottingConfig(return_raw_html=True))
    # print(html, file=open("x.html", "w+"))


@pytest.mark.parametrize(
    "scores, expected",
    [
        (
            [25.0, 20.0, 10.0, 5.0, 0.0],
            [
                (255, 0, 0),
                (255, 51, 51),
                (255, 153, 153),
                (255, 204, 204),
                (255, 255, 255),
            ],
        ),
        (
            [25.0, -10.0, -5.0, 5.0],
            [
                (255, 0, 0),
                (0, 0, 255),
                (36.0, 36.0, 255),
                (255.0, 146.0, 146.0),
            ],
        ),
    ],
    ids=["positive", "negative"],
)
def test_color_mapper(scores, expected):
    scores = tf.constant(scores)
    result = map_to_rgb(scores, config=PlottingConfig())
    assert isinstance(result, list)
    assert isinstance(result[0], tuple)
    result = np.round(result)
    assert np.allclose(result, expected)
