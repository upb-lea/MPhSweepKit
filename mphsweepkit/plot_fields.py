from typing import Any

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_field(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    value_col: str = "Magnetic flux density norm",
    kind: str = "scatter",  # "scatter" or "tricontourf"
    cmap: str = "viridis",
    point_size: int = 8,
    levels: int = 30,
    figsize: tuple = (7, 5),
):
    """
    Plot COMSOL point data from a DataFrame.

    :param df: DataFrame containing x, y, and value columns.
    :param x_col: Column name for x-axis values.
    :param y_col: Column name for y-axis values.
    :param value_col: Column name for color values.
    :param kind: Type of plot ('scatter' or 'tricontourf').
    :param cmap: Colormap for the plot.
    :param point_size: Size of points in scatter plot.
    :param levels: Number of contour levels.
    :param figsize: Figure size for the plot.
    :param kind: Type of plot ('scatter' or 'tricontourf').
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    z = df[value_col].to_numpy()

    if kind == "scatter":
        mappable = ax.scatter(x, y, c=z, s=point_size, cmap=cmap)
    elif kind == "tricontourf":
        mappable = ax.tricontourf(x, y, z, levels=levels, cmap=cmap)
    else:
        raise ValueError("kind must be 'scatter' or 'tricontourf'")

    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(value_col)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{value_col} over ({x_col}, {y_col})")
    ax.set_aspect("equal")  # useful for geometric fields
    plt.tight_layout()
    plt.show()

