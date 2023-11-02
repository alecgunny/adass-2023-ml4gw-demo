from typing import Optional

import numpy as np
import pandas as pd
from bokeh.io import curdoc, output_notebook, show
from bokeh.models import LinearAxis, Range1d
from bokeh.palettes import Bright7 as palette
from bokeh.plotting import figure
from bokeh.themes import Theme
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

theme = {
    "attrs": {
        "figure": {
            "background_fill_color": "#fafafa",
            "background_fill_alpha": 0.8,
            "height": 300,
            "width": 700,
        },
        "Grid": {"grid_line_color": "#aaaaaa", "grid_line_width": 0.8},
    },
    "line_defaults": {
        "line_width": 1.25,
        "line_alpha": 0.6,
        "line_color": palette[0],
    },
}

curdoc().theme = Theme(json=theme)
output_notebook()


def _latexify(label):
    return rf"$$\text{{{label}}}$$"


def make_figure(**kwargs):
    default_kwargs = dict(tools="")
    default_kwargs.update(kwargs)
    for key in ["x", "y", "title"]:
        if key in ["x", "y"]:
            key = f"{key}_axis_label"

        try:
            value = default_kwargs[key]
        except KeyError:
            continue
        if not value.startswith("$$"):
            default_kwargs[key] = _latexify(value)

    p = figure(**default_kwargs)
    if not default_kwargs["tools"]:
        p.toolbar_location = None

    title = default_kwargs.get("title")
    if title:
        p.title.text_font_style = "normal"
    return p


def hide_axis(p, axis):
    axis = getattr(p, axis + "axis")
    axis.major_tick_line_color = None
    axis.minor_tick_line_color = None

    # can't set this to 0 otherwise log-axis
    # plots freak out and won't render
    axis.major_label_text_font_size = "1pt"
    axis.major_label_text_color = None


def plot_lines(
    x: Optional[np.ndarray] = None,
    fig_kwargs: dict = {},
    line_kwargs: dict = {},
    legend_location: Optional[str] = None,
    **y: np.ndarray,
):
    p = make_figure(**fig_kwargs)
    for color, (label, arr) in zip(palette, y.items()):
        if x is not None:
            _x = x
        elif isinstance(arr, TimeSeries):
            _x = arr.times
            axlabel = f"Time from GPS epoch {_x[0].value:0.1f} [s]"
            p.xaxis.axis_label = _latexify(axlabel)
            _x = _x - _x[0]
        elif isinstance(arr, FrequencySeries):
            _x = arr.frequencies
        else:
            raise ValueError(
                "Must specify x values if timeseries "
                "do not have an xindex associated with them."
            )

        p.line(_x, y=arr, line_color=color, legend_label=label, **line_kwargs)
    if legend_location is not None:
        p.legend.location = legend_location
    show(p)


def plot_timeseries(
    x: Optional[np.ndarray] = None,
    fig_kwargs: dict = {},
    line_kwargs: dict = {},
    legend_location: Optional[str] = None,
    **y: np.ndarray,
):
    if "x_axis_label" not in fig_kwargs:
        fig_kwargs["x_axis_label"] = "Time [s]"
    if "y_axis_label" not in fig_kwargs:
        fig_kwargs["y_axis_label"] = "Strain [unitless]"
    plot_lines(x, fig_kwargs, line_kwargs, legend_location, **y)


def plot_spectral(
    x: Optional[np.ndarray] = None,
    fig_kwargs: dict = {},
    line_kwargs: dict = {},
    legend_location: Optional[str] = None,
    **y: np.ndarray,
):
    if "x_axis_label" not in fig_kwargs:
        fig_kwargs["x_axis_label"] = "Frequency [Hz]"
    if "y_axis_label" not in fig_kwargs:
        fig_kwargs["y_axis_label"] = r"$$\text{Power [Hz}^{-1}\text{]}$$"
    fig_kwargs["y_axis_type"] = "log"
    fig_kwargs["x_axis_type"] = "log"
    plot_lines(x, fig_kwargs, line_kwargs, legend_location, **y)


def plot_run(name, version=0):
    df = pd.read_csv(f"logs/{name}/version_{version}/metrics.csv")
    p = make_figure(x_axis_label="Step", y_axis_label="Train Loss")

    mask = ~pd.isnull(df["train_loss"])
    p.line(
        df["step"][mask],
        df["train_loss"][mask],
        line_color=palette[2],
        line_alpha=0.8,
        line_width=1.5,
        legend_label="Train Loss",
    )

    max_val = df["valid_auroc"].max()
    min_val = df["valid_auroc"].min()
    pad = 0.05 * (max_val - min_val)
    max_val += pad
    min_val -= pad
    p.extra_y_ranges = {"auroc": Range1d(min_val, max_val)}
    ax = LinearAxis(axis_label="Valid AUROC", y_range_name="auroc")
    p.add_layout(ax, "right")

    mask = ~pd.isnull(df["valid_auroc"])
    p.line(
        df["step"][mask],
        df["valid_auroc"][mask],
        line_color=palette[3],
        line_alpha=0.8,
        line_width=1.5,
        y_range_name="auroc",
        legend_label="Valid AUROC",
    )
    p.legend.location = "bottom_right"
    show(p)
