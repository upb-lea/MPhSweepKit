from typing import Any

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PlotField:
    """
    Class for plotting COMSOL point data from a DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the PlotField class with a DataFrame.

        :param df: DataFrame containing x, y, and value columns.
        """
        self.df = df

    def plot_field(
        self,
        x_col: str = "x",
        y_col: str = "y",
        value_col: str = "Magnetic flux density norm",
        kind: str = "scatter",  # "scatter" or "tricontourf"
        cmap: str = "viridis",
        point_size: int = 8,
        levels: int = 30,
        figsize: tuple = (7, 5)
        ):
        """
        Plot COMSOL point data from a DataFrame.

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
        x = self.df[x_col].to_numpy()
        y = self.df[y_col].to_numpy()
        z = self.df[value_col].to_numpy()

        fig, ax = plt.subplots(figsize=figsize)

        if kind == "scatter":
            mappable = ax.scatter(x, y, c=z, s=point_size, cmap=cmap)
        else:
            mappable = ax.tricontourf(x, y, z, levels=levels, cmap=cmap)

        ax.set_aspect("equal", adjustable="box")

        fig.colorbar(mappable, ax=ax)
        plt.tight_layout()
        plt.show()
