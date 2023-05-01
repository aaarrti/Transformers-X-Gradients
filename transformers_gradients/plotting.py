from __future__ import annotations

import logging
from functools import lru_cache, partial
from itertools import starmap
from typing import List, Tuple, Mapping, TYPE_CHECKING

import tensorflow as tf
import tensorflow_probability as tfp
from jinja2 import FileSystemLoader, Environment
from pydantic import BaseModel

from transformers_gradients.lib_types import Explanation, PlottingConfig
from transformers_gradients.utils import value_or_default, mapping_to_config

log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from IPython.core.display import HTML


class TokenSaliency(BaseModel):
    item: str
    color: Tuple[float, float, float]

    @staticmethod
    def build(item, color):
        return TokenSaliency(item=item, color=color)


class HeatmapRow(BaseModel):
    label: str
    explanation: List[TokenSaliency]

    @staticmethod
    def build(label, explanation):
        return HeatmapRow(label=label, explanation=explanation)


def post_process_colors(
    colors: tf.Tensor, rgb_range: float
) -> List[Tuple[float, float, float]]:
    if rgb_range == 255:
        colors = tf.cast(colors, tf.int32)
    colors = colors.numpy().tolist()
    colors = list(map(tuple, colors))
    return colors


def map_to_rgb(
    scores: tf.Tensor,
    *,
    config: PlottingConfig,
    max_score: float | None = None,
    min_score: float | None = None,
) -> List[Tuple[float, float, float]]:
    """
    - Highest score get red (255,0,0).
    - Lowest score gets blue (0,0,255).
    - Positive scores are linearly interpolated between red and white (255, 255, 255).
    - Negative scores are linearly interpolated between blue and white (255, 255, 255).
    """
    rgb_range = float(config.rgb_range)
    min_score = value_or_default(min_score, lambda: tf.reduce_min(scores))
    max_score = value_or_default(max_score, lambda: tf.reduce_max(scores))

    positive_colors = (
        tfp.math.interp_regular_1d_grid(
            x=tf.abs(scores),
            x_ref_min=min_score,
            x_ref_max=max_score,
            y_ref=[
                tf.constant([rgb_range, rgb_range, rgb_range]),
                tf.constant([rgb_range, 0.0, 0.0]),
            ],
            axis=0,
        )
        * config.rgb_scale
    )

    positive_colors = post_process_colors(positive_colors, rgb_range)

    if tf.reduce_min(scores) > 0:
        return positive_colors

    negative_colors = (
        tfp.math.interp_regular_1d_grid(
            x=-1 * tf.abs(scores),
            x_ref_min=min_score,
            x_ref_max=max_score,
            y_ref=[
                tf.constant([0.0, 0.0, rgb_range]),
                tf.constant([rgb_range, rgb_range, rgb_range]),
            ],
            axis=0,
        )
        * config.rgb_scale
    )

    negative_colors = post_process_colors(negative_colors, rgb_range)

    colors = []
    for i, s in enumerate(scores):
        if s >= 0:
            colors.append(positive_colors[i])
        else:
            colors.append(negative_colors[i])

    return colors


@lru_cache(maxsize=None)
def load_template():
    template_folder = "/".join(__file__.split("/")[:-1]) + "/templates/"
    templateLoader = FileSystemLoader(searchpath=template_folder)
    templateEnv = Environment(loader=templateLoader)
    template = templateEnv.get_template("heatmap.html")
    return template


def html_heatmap(
    explanations: List[Explanation],
    *,
    labels: List[str] | None = None,
    config: PlottingConfig | Mapping[str, ...] | None = None,
) -> str | HTML:
    """
    Creates a heatmap visualisation from list of explanations. This method should be preferred for longer
    examples. It is rendered correctly in VSCode, PyCharm, Colab, however not in GitHub or JupyterLab.

    Parameters
    ----------
    explanations:
        List of tuples (tokens, salience) containing batch of explanations.
    labels:
        Optional, list of labels to display on top of each explanation.
    config:
        plotting config.

    Returns
    -------

    html:
        string containing raw html to visualise explanations.

    """

    labels = value_or_default(
        labels, lambda: [f"{i}. sample" for i in range(len(explanations))]
    )
    config = mapping_to_config(config, PlottingConfig)
    config = value_or_default(config, lambda: PlottingConfig())

    scores_only = tf.stack([i.scores for i in explanations])

    if config.color_mapping_strategy == "global":
        color_mapper = partial(
            map_to_rgb,
            max_score=tf.reduce_max(scores_only),
            min_score=tf.reduce_min(scores_only),
        )
    else:
        color_mapper = map_to_rgb

    color_mapper = partial(color_mapper, config=config)

    def mask_token_if_ignored(t: str) -> str:
        if not config.ignore_special_tokens:
            return t
        return "" if t in config.special_tokens else t

    def to_token_saliencies(explanation: Explanation) -> List[TokenSaliency]:
        tokens = map(mask_token_if_ignored, explanation.tokens)
        colors = color_mapper(explanation.scores)
        return list(
            starmap(
                TokenSaliency.build,
                zip(tokens, colors),
            )
        )

    explanations_batch = list(
        starmap(HeatmapRow.build, zip(labels, map(to_token_saliencies, explanations)))
    )
    template = load_template()
    heatmap = template.render(explanations_batch=explanations_batch)

    if config.return_raw_html:
        return heatmap

    try:
        from IPython.core.display import HTML

        return HTML(heatmap)
    except ModuleNotFoundError:
        log.warning("Not running in Jupyter.")
        return heatmap
