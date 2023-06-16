import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go

from preprocessing.quantile_format import split_prediction_interval_symmetrically

from typing import List


def plot_quantiles(y_pred: pd.DataFrame,
                   y_real: pd.Series = None,
                   quantiles: List[float] = None,
                   y_axis_label: str = None):
    """Plot each quantile with matplotlib"""
    if quantiles is None:
        quantiles = y_pred.columns.to_series().sort_values()
    color_palette = plt.get_cmap("BuPu")(np.linspace(0, 1, len(quantiles)))
    fig, ax = plt.subplots()
    if y_real is not None:
        y_real.plot(color="red", ax=ax, label="Real Value")
    y_pred.plot(color=color_palette, ax=ax, legend=False)
    plt.colorbar(mappable=mpl.cm.ScalarMappable(mpl.colors.Normalize(0.01, 0.99), cmap="BuPu"),
                 ax=ax,
                 ticks=[i / 10 for i in range(1, 10)],
                 label="Quantile level")
    if y_axis_label:
        ax.set_ylabel(y_axis_label)
    plt.show()


def plot_prediction_intervals(y_pred: pd.DataFrame,
                              y_real: pd.Series = None,
                              quantiles: List[float] = None,
                              plot_median: bool = False,
                              y_axis_label: str = None):
    """Plot symmetric prediction intervals with matplotlib"""
    if quantiles is None:
        quantiles = y_pred.columns.to_series().sort_values()
    else:
        quantiles = pd.Series(quantiles).sort_values()
    lower_bounds, upper_bounds = split_prediction_interval_symmetrically(quantiles[quantiles < 0.5],
                                                                         quantiles[quantiles > 0.5])

    color_palette = plt.get_cmap("BuPu")(np.linspace(0, 1, len(lower_bounds)))
    fig, ax = plt.subplots()
    if y_real is not None:
        y_real.plot(color="red", ax=ax, label="Real Value")
    if plot_median:
        y_pred[0.5].plot(color="yellow", ax=ax, label="Median", linestyle="--")
    for c, l, u in zip(color_palette, lower_bounds, upper_bounds.sort_values(ascending=False)):
        ax.fill_between(
            y_pred.index,
            y_pred[l],
            y_pred[u],
            facecolor=c,
            interpolate=True
        )
    plt.colorbar(mappable=mpl.cm.ScalarMappable(mpl.colors.Normalize(1, 99), cmap="BuPu_r"),
                 ax=ax,
                 ticks=[i * 10 for i in range(1, 10)],
                 label="Prediction Interval [%]")
    if y_axis_label:
        ax.set_ylabel(y_axis_label)
    plt.legend()
    plt.show()


def plot_quantiles_interactive(y_pred: pd.DataFrame,
                               y_real: pd.Series = None,
                               quantiles: List[float] = None,
                               y_axis_label: str = None):
    """Plot each quantile with plotly"""
    if quantiles is None:
        quantiles = y_pred.columns.to_series().sort_values()
    fig = px.line(y_pred,
                  color_discrete_sequence=px.colors.sample_colorscale('bupu', np.linspace(0, 1, len(quantiles))),
                  labels={'variable': 'Quantile'})
    if y_real is not None:
        fig.add_trace(go.Scatter(x=y_real.index, y=y_real.values, mode='lines', line_color='red', name='Real value'))
    if y_axis_label:
        fig.update_layout(yaxis_title=y_axis_label)
    fig.show()


def plot_prediction_intervals_interactive(y_pred: pd.DataFrame,
                                          y_real: pd.Series = None,
                                          quantiles: List[float] = None,
                                          plot_median: bool = False,
                                          y_axis_label: str = None):
    """Plot symmetric prediction intervals with plotly"""
    if quantiles is None:
        quantiles = y_pred.columns.to_series().sort_values()
    else:
        quantiles = pd.Series(quantiles).sort_values()
    lower_bounds, upper_bounds = split_prediction_interval_symmetrically(quantiles[quantiles < 0.5],
                                                                         quantiles[quantiles > 0.5])

    color_palette = px.colors.sample_colorscale('bupu', np.linspace(0, 1, len(lower_bounds)))
    fig = go.Figure()
    for c, l, u in zip(color_palette, lower_bounds, upper_bounds.sort_values(ascending=False)):
        pi = int(100 * round((u - l), 2))
        fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred[u].values, fill=None,
                                 mode='lines', line_color=c, name=f'{pi}%', legendgroup=pi))
        fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred[l].values, fill='tonexty',
                                 mode='lines', line_color=c, name=l, legendgroup=pi, showlegend=False))
    if plot_median:
        fig.add_trace(
            go.Scatter(x=y_pred.index, y=y_pred[0.5].values, mode='lines', line={'color': 'yellow', 'dash': 'dash'},
                       name='0.5q'))
    if y_real is not None:
        fig.add_trace(go.Scatter(x=y_real.index, y=y_real.values, mode='lines', line_color='red', name='Real value'))
    fig.update_layout(legend_title_text='Prediction interval')
    if y_axis_label:
        fig.update_layout(yaxis_title=y_axis_label)
    fig.show()
