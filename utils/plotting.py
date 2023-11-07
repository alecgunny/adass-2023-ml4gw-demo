from typing import Optional

import numpy as np
import pandas as pd
from bokeh.io import curdoc, output_notebook
from bokeh.io import show as _show
from bokeh.layouts import grid, row
from bokeh.models import HoverTool, LinearAxis, Range1d
from bokeh.palettes import Vibrant7 as palette
from bokeh.plotting import figure
from bokeh.themes import Theme
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

from utils.evaluate import evaluate

theme = {
    "attrs": {
        "figure": {
            "background_fill_color": "#fafafa",
            "background_fill_alpha": 0.8,
            "height": 300,
            "width": 800,
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
    fig_kwargs: Optional[dict] = None,
    line_kwargs: Optional[dict] = None,
    legend_location: Optional[str] = None,
    show: bool = True,
    **y: np.ndarray,
):
    fig_kwargs = fig_kwargs or {}
    line_kwargs = line_kwargs or {}
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

    p.legend.location = legend_location or "top_left"
    if not show:
        return p
    _show(p)


def plot_timeseries(
    x: Optional[np.ndarray] = None,
    fig_kwargs: Optional[dict] = None,
    line_kwargs: Optional[dict] = None,
    legend_location: Optional[str] = None,
    **y: np.ndarray,
):
    fig_kwargs = fig_kwargs or {}
    line_kwargs = line_kwargs or {}
    if "x_axis_label" not in fig_kwargs:
        fig_kwargs["x_axis_label"] = "Time [s]"
    if "y_axis_label" not in fig_kwargs:
        fig_kwargs["y_axis_label"] = "Strain [unitless]"
    return plot_lines(x, fig_kwargs, line_kwargs, legend_location, **y)


def plot_side_by_side(
    y1: dict[str, np.ndarray],
    y2: dict[str, np.ndarray],
    x: Optional[np.ndarray] = None,
    fig_kwargs: Optional[dict] = None,
    line_kwargs: Optional[dict] = None,
    legend_location: Optional[str] = None,
    titles: Optional[list[str]] = None,
):
    fig_kwargs = fig_kwargs or {}
    line_kwargs = line_kwargs or {}
    if "x_axis_label" not in fig_kwargs:
        fig_kwargs["x_axis_label"] = "Time [s]"
    if "y_axis_label" not in fig_kwargs:
        fig_kwargs["y_axis_label"] = "Strain [unitless]"
    if "width" not in fig_kwargs:
        fig_kwargs["width"] = 450

    if titles is not None:
        fig_kwargs["title"] = titles[0]

    p1 = plot_lines(
        x, fig_kwargs, line_kwargs, legend_location, show=False, **y1
    )
    fig_kwargs.pop("y_axis_label")
    fig_kwargs["y_range"] = p1.y_range
    if titles is not None:
        fig_kwargs["title"] = titles[1]
    fig_kwargs["width"] -= 75

    p2 = plot_lines(
        x, fig_kwargs, line_kwargs, legend_location, show=False, **y2
    )
    hide_axis(p2, "y")
    _show(row(p1, p2))


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
    return plot_lines(x, fig_kwargs, line_kwargs, legend_location, **y)


def _get_bounds(values, pad):
    max_val = values.max()
    min_val = values.min()
    pad = pad * (max_val - min_val)
    max_val += pad
    min_val -= pad
    return min_val, max_val


def plot_run(name, version=0):
    df = pd.read_csv(f"logs/{name}/version_{version}/metrics.csv")
    y_range = _get_bounds(df["train_loss"], 0.05)
    p = make_figure(
        x_axis_label="Step", y_axis_label="Train Loss", y_range=y_range
    )

    mask = ~pd.isnull(df["train_loss"])
    r = p.line(
        "step",
        "train_loss",
        line_color=palette[2],
        line_alpha=0.8,
        line_width=2.0,
        legend_label="Train Loss",
        source=df[mask],
    )
    tool = HoverTool(
        renderers=[r],
        tooltips=[("Step", "@step"), ("Train Loss", "@train_loss")],
    )
    p.add_tools(tool)

    y_range = _get_bounds(df["valid_auroc"], 0.05)
    p.extra_y_ranges = {"auroc": Range1d(*y_range)}
    ax = LinearAxis(axis_label=_latexify("Valid AUROC"), y_range_name="auroc")
    p.add_layout(ax, "right")

    mask = ~pd.isnull(df["valid_auroc"])
    r = p.line(
        "step",
        "valid_auroc",
        line_color=palette[3],
        line_alpha=0.8,
        line_width=2.0,
        y_range_name="auroc",
        legend_label="Valid AUROC",
        source=df[mask],
    )
    tool = HoverTool(
        renderers=[r],
        tooltips=[("Step", "@step"), ("Valid AUROC", "@valid_auroc")],
    )
    p.add_tools(tool)

    p.legend.location = "right"
    _show(p)


def make_grid(combos):
    if len(combos) != 4:
        raise ValueError(
            "Only support 2x2 grids, can't plot {} combos".format(len(combos))
        )

    plots = []
    for i, combo in enumerate(combos):
        kwargs = dict(
            title=r"$$\text{{Log Normal }}m_1={}, m_2={}$$".format(*combo),
            x_axis_type="log",
        )

        kwargs["width"] = 350
        if not i % 2:
            # plots on the left need space for y-axis label
            kwargs["width"] += 30
            kwargs["y_axis_label"] = (
                r"$$\text{Sensitive Volume [Gpc}" r"^{3}\text{]}$$"
            )

        kwargs["height"] = 220
        if i > 1:
            # lower plots need space for x-axis label
            kwargs["height"] += 30
            kwargs["x_axis_label"] = (
                r"$$\text{False Alarm Rate " r"[weeks}^{-1}\text{]}$$"
            )

        # share x range between all plots
        if plots:
            kwargs["x_range"] = plots[0].x_range

        p = make_figure(**kwargs)
        p.outline_line_color = "#ffffff"

        # don't show x axis on upper plots
        if i < 2:
            hide_axis(p, "x")
        plots.append(p)
    return plots


def plot_evaluation(**results):
    plots = []
    for color, (label, result) in zip(palette[2:], results.items()):
        fars, svs = evaluate(*result, max_far_per_week=200)
        combos = sorted(svs)
        if not plots:
            plots = make_grid(combos)

        for plot, combo in zip(plots, combos):
            kwargs = {}
            if combo == combos[0]:
                kwargs["legend_label"] = label

            plot.line(
                x=fars,
                y=svs[combo],
                line_color=color,
                line_width=1.5,
                line_alpha=0.7,
                **kwargs,
            )
            if combo == combos[0]:
                plot.legend.location = "top_left"
    _show(grid(plots, ncols=2))
