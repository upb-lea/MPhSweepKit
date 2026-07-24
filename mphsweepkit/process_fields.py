import matplotlib.pyplot as plt
from collections.abc import Sequence
from matplotlib.colors import Normalize

import numpy as np
import pandas as pd


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
        value_col: str,
        ax: plt.Axes | None = None,
        x_col: str = "x",
        y_col: str = "y",
        x_unit: str = "m",
        y_unit: str = "m",
        cmap: str = "viridis",
        levels: int | Sequence[float] = 30,
        colorbar: bool = True,
        norm: Normalize | None = None,
    ):
        """
        Plot an irregularly sampled field as filled contours.

        :param value_col: DataFrame column containing the plotted field values.
        :param ax: Axes on which to draw. A new figure and axes are created if omitted.
        :param x_col: DataFrame column containing x coordinates.
        :param y_col: DataFrame column containing y coordinates.
        :param x_unit: Unit for the x-axis.
        :param y_unit: Unit for the y-axis.
        :param cmap: Matplotlib colormap name.
        :param levels: Number of contour levels or an explicit sequence of level boundaries.
        :param colorbar: Whether to add a colorbar.

        :returns: contour
        """
        if ax is None:
            _, ax = plt.subplots()

        x = self.df[x_col].to_numpy()
        y = self.df[y_col].to_numpy()
        fct_xy = self.df[value_col].to_numpy()

        contour = ax.tricontourf(x, y, fct_xy, levels=levels, cmap=cmap, norm=norm)
        ax.set(xlabel=f"{x_col} [{x_unit}]", ylabel=f"{y_col} [{y_unit}]", aspect="equal")
        ax.figure.colorbar(contour, ax=ax) if colorbar else None

        return contour


    def plot_fields(
        self,
        list_of_value_cols: Sequence[str],
        list_of_labels: Sequence[str],
        field_name: str,
        unit: str,
        figsize: tuple[int, int] = (4, 4),
        x_col: str = "x",
        y_col: str = "y",
        cmap: str = "viridis",
        levels: int = 50,
        share_y: bool = True
    ):
        """
        Plot multiple fields side by side with a shared color scale.

        :param list_of_value_cols: DataFrame columns containing the fields to plot.
        :param list_of_labels: Labels for each subplot.
        :param unit: The unit of the field values.
        :param figsize: Figure size.
        :param x_col: DataFrame column containing x coordinates.
        :param y_col: DataFrame column containing y coordinates.
        :param cmap: Matplotlib colormap name.
        :param levels: Number of contour levels.
        :param share_y: Whether to share the y-axis across subplots.
        :param field_name: The name of the field.
        :param unit: The unit of the field values.

        :returns: fig, axes, cbar
        """
        values = self.df[list(list_of_value_cols)].to_numpy()
        norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
        shared_levels = np.linspace(norm.vmin, norm.vmax, levels)

        fig, axes = plt.subplots(
            1,
            len(list_of_value_cols),
            figsize=figsize,
            constrained_layout=True,
            sharey=share_y
        )
        axes = np.atleast_1d(axes)

        for ax, value_col, label in zip(axes, list_of_value_cols, list_of_labels):
            contour = self.plot_field(
                ax=ax,
                x_col=x_col,
                y_col=y_col,
                value_col=value_col,
                cmap=cmap,
                levels=shared_levels,
                norm=norm,
                colorbar=False,
            )

            ax.set_title(label)

            # no y label for all but the first subplot
            if ax != axes[0]:
                ax.set_ylabel("")

        cbar = fig.colorbar(contour, ax=axes, location="right")
        cbar.ax.set_title(f"{field_name}\n[{unit}]")

        return fig, axes, cbar