import matplotlib.pyplot as plt
from collections.abc import Sequence
from matplotlib.colors import Normalize

import numpy as np
import pandas as pd


def plot_field(
    df: pd.DataFrame,
    value_col: str,
    ax: plt.Axes | None = None,
    x_col: str = "x",
    y_col: str = "y",
    length_unit: str = "m",
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
    :param length_unit: Unit for the length dimensions.
    :param cmap: Matplotlib colormap name.
    :param levels: Number of contour levels or an explicit sequence of level boundaries.
    :param colorbar: Whether to add a colorbar.

    :returns: contour
    """
    if ax is None:
        _, ax = plt.subplots()

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    fct_xy = df[value_col].to_numpy()

    contour = ax.tricontourf(x, y, fct_xy, levels=levels, cmap=cmap, norm=norm)
    ax.set(xlabel=f"{x_col} [{length_unit}]", 
           ylabel=f"{y_col} [{length_unit}]", 
           aspect="equal")
    ax.figure.colorbar(contour, ax=ax) if colorbar else None

    return contour


def plot_row_of_fields(
    df: pd.DataFrame,
    list_of_col_names: Sequence[str],
    list_of_labels: Sequence[str],
    field_name: str = "",
    field_symbol: str = "",
    field_unit: str = "",
    figsize: tuple[int, int] = (4, 4),
    x_col: str = "x",
    y_col: str = "y",
    length_unit: str = "m",
    cmap: str = "viridis",
    levels: int = 50,
    share_y: bool = True
):
    """
    Plot multiple fields side by side with a shared color scale.

    :param list_of_value_cols: DataFrame columns containing the fields to plot.
    :param list_of_labels: Labels for each subplot.
    :param field_name: The name of the field. To make it invisible use an empty string "".
    :param field_symbol: The symbol of the field. To make it invisible use an empty string "".
    :param field_unit: The unit of the field values. To make it invisible use an empty string "".
    :param figsize: Figure size.
    :param x_col: DataFrame column containing x coordinates.
    :param y_col: DataFrame column containing y coordinates.
    :param length_unit: Unit for the length dimensions. To make it invisible use an empty string "".
    :param cmap: Matplotlib colormap name.
    :param levels: Number of contour levels.
    :param share_y: Whether to share the y-axis across subplots.

    :returns: fig, axes, cbar
    """
    values = df[list(list_of_col_names)].to_numpy()
    norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    shared_levels = np.linspace(start=norm.vmin, stop=norm.vmax, num=levels)

    fig, axes = plt.subplots(
        1,
        len(list_of_col_names),
        figsize=figsize,
        constrained_layout=True,
        sharey=share_y
    )
    axes = np.atleast_1d(axes)

    for ax, value_col, label in zip(axes, list_of_col_names, list_of_labels):
        contour = plot_field(
            df=df,
            ax=ax,
            x_col=x_col,
            y_col=y_col,
            length_unit=length_unit,
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
    cbar.ax.set_title(f"{field_name}\n{field_symbol} [{field_unit}]", ha="left", x=0)

    return fig, axes, cbar


def plot_grid_of_fields(
    list_of_dfs: Sequence[pd.DataFrame],
    list_of_row_labels: Sequence[str],
    list_of_value_cols: Sequence[Sequence[str]],
    list_of_col_labels: Sequence[str],
    field_name: str = "",
    field_symbol: str = "",
    field_unit: str = "",
    length_unit: str = "m",
    x_col: str = "x",
    y_col: str = "y",
    cmap: str = "viridis",
    levels: int = 50,
    figsize: tuple = (10, 8),
    share_x: bool = True,
    share_y=True,
    normalize_mode="global",  # "global" or "per_row"
    title: str | None = None,
):
    """
    Plot a 2D grid of field contours:
      - rows: different geometries
      - cols: different parameter cases (e.g., frequencies, materials)

    :param list_of_dfs: One field dataframe per row.
    :param list_of_row_labels: Row labels (same length as list_of_dfs).
    :param list_of_value_cols: 
        If list[str]: same value columns for every row.
        If list[list[str]]: row-specific columns.
    :param list_of_col_labels: Column titles.
    :param field_name: The name of the field. To make it invisible use an empty string "".
    :param field_symbol: The symbol of the field. To make it invisible use an empty string "".
    :param field_unit: The unit of the field values. To make it invisible use an empty string "".
    :param length_unit: Unit for the length dimensions. To make it invisible use an empty string "".
    :param x_col: DataFrame column containing x coordinates.
    :param y_col: DataFrame column containing y coordinates.
    :param cmap: The colormap to use for the contours.
    :param levels: The number of contour levels.
    :param figsize: The size of the figure.
    :param share_x: Whether to share the x-axis across subplots.
    :param share_y: Whether to share the y-axis across subplots.
    :param normalize_mode: "global" -> one color scale for all subplots
        "per_row" -> one color scale per row
    :param title: Optional title shown above the full subplot grid.
    """
    n_rows = len(list_of_dfs)
    if n_rows == 0:
        raise ValueError("list_of_dfs is empty.")

    # allow shared or row-specific column selection
    if len(list_of_value_cols) > 0 and isinstance(list_of_value_cols[0], (list, tuple)):
        row_value_cols = [list(cols) for cols in list_of_value_cols]
    else:
        row_value_cols = [list(list_of_value_cols) for _ in range(n_rows)]

    n_cols = len(row_value_cols[0])
    if n_cols == 0:
        raise ValueError("No value columns provided.")

    if any(len(cols) != n_cols for cols in row_value_cols):
        raise ValueError("All rows must use the same number of columns in list_of_value_cols.")

    if len(list_of_col_labels) != n_cols:
        raise ValueError("list_of_col_labels length must equal number of columns.")
    if len(list_of_row_labels) != n_rows:
        raise ValueError("list_of_row_labels length must equal number of rows.")

    # validate column availability
    for i, (df_i, cols_i) in enumerate(zip(list_of_dfs, row_value_cols)):
        missing = [c for c in cols_i if c not in df_i.columns]
        if missing:
            raise KeyError(f"Row {i} is missing field columns: {missing}")

    # compute normalizations
    if normalize_mode == "global":
        all_vals = np.concatenate(
            [df_i[cols_i].to_numpy().ravel() for df_i, cols_i in zip(list_of_dfs, row_value_cols)]
        )
        norm_global = Normalize(vmin=np.nanmin(all_vals), vmax=np.nanmax(all_vals))
        row_norms = [norm_global] * n_rows
    elif normalize_mode == "per_row":
        row_norms = []
        for df_i, cols_i in zip(list_of_dfs, row_value_cols):
            vals = df_i[cols_i].to_numpy()
            row_norms.append(Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals)))
    else:
        raise ValueError("normalize_mode must be 'global' or 'per_row'.")

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True,
        sharex=share_x,
        sharey=share_y,
        squeeze=False
    )

    if title is not None:
        fig.suptitle(title)

    contour_last = None
    row_contours = []

    for r, (df_i, cols_i, row_label, norm_i) in enumerate(
        zip(list_of_dfs, row_value_cols, list_of_row_labels, row_norms)
    ):
        shared_levels = np.linspace(norm_i.vmin, norm_i.vmax, num=levels)

        for c, (value_col, col_label) in enumerate(zip(cols_i, list_of_col_labels)):
            ax = axes[r, c]
            contour_last = plot_field(
                df=df_i,
                value_col=value_col,
                ax=ax,
                x_col=x_col,
                y_col=y_col,
                length_unit=length_unit,
                cmap=cmap,
                levels=shared_levels,
                colorbar=False,
                norm=norm_i,
            )

            # column titles only on first row
            if r == 0:
                ax.set_title(col_label)

            # row label on first column
            if c == 0:
                ax.annotate(
                    row_label,
                    xy=(-0.2, 0.5),
                    xycoords="axes fraction",
                    xytext=(-ax.yaxis.labelpad - 28, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=10,
                    fontweight="bold",
                )
            else:
                ax.set_ylabel("")

        row_contours.append(contour_last)

    # keep x-label only on bottom row if x-axes are shared
    if share_x and n_rows > 1:
        for r in range(n_rows - 1):
            for ax in axes[r, :]:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel("")

    # colorbar(s)
    if normalize_mode == "global":
        cbar = fig.colorbar(contour_last, ax=axes, location="right")
        cbar.ax.set_title(f"{field_name}\n{field_symbol} [{field_unit}]", ha="left", x=0)
        cbars = [cbar]
    else:
        cbars = []
        for r in range(n_rows):
            cbar_r = fig.colorbar(row_contours[r], ax=axes[r, :], location="right")
            cbar_r.ax.set_title(f"{field_name}\n{field_symbol} [{field_unit}]", ha="left", x=0)
            cbars.append(cbar_r)

    return fig, axes, cbars