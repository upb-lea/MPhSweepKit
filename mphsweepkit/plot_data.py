from __future__ import annotations

from typing import Any
from pathlib import Path
import json

import matplotlib.patheffects as pe
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


class DataPlot:
    """Plot helper for scalar metrics stored in result-data tables.

    The class is designed to load data exported by
    ``CascadedSweepModel.save_result_data``.
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        input_meta: dict[str, Any] | None = None,
        output_meta: dict[str, Any] | None = None,
        folder: str | Path | None = None,
    ):
        self.folder = Path(folder) if folder is not None else None

        self.input_df = input_df
        self.output_df = output_df

        input_meta = input_meta or {}
        output_meta = output_meta or {}

        self.input_unit_map: dict[str, str | None] = input_meta.get("input_unit_map", {})
        self.input_sweep_map: dict[str, str | None] = input_meta.get("input_sweep_map", {})
        self.output_unit_map: dict[str, str | None] = output_meta.get("output_unit_map", {})
        self.output_label_map: dict[str, str | None] = output_meta.get("output_label_map", {})

        self.meta: dict[str, Any] = {
            "input_unit_map": self.input_unit_map,
            "input_sweep_map": self.input_sweep_map,
            "output_unit_map": self.output_unit_map,
            "output_label_map": self.output_label_map,
        }

        self.combined_df = self._build_combined_df()

    @classmethod
    def from_result_folder(
        cls,
        folder: str | Path = "result_data",
        *,
        index_col: int | str | None = 0,
    ) -> "DataPlot":
        """Load result tables and metadata from a result-data folder."""
        root = Path(folder)

        input_path = root / "input_data.csv"
        output_path = root / "output_data.csv"
        input_meta_path = root / "input_meta.json"
        output_meta_path = root / "output_meta.json"

        if not root.exists():
            raise FileNotFoundError(f"Result folder does not exist: {root}")

        if not input_path.exists() and not output_path.exists():
            raise FileNotFoundError(
                f"Neither input_data.csv nor output_data.csv found in: {root}"
            )

        input_df = cls._load_csv(input_path, index_col=index_col)
        output_df = cls._load_csv(output_path, index_col=index_col)
        input_meta = cls._load_json(input_meta_path)
        output_meta = cls._load_json(output_meta_path)

        return cls(
            input_df=input_df,
            output_df=output_df,
            input_meta=input_meta,
            output_meta=output_meta,
            folder=root,
        )

    @staticmethod
    def _load_csv(path: Path, index_col: int | str | None = 0) -> pd.DataFrame:
        """Load CSV or return empty dataframe if missing."""
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, index_col=index_col)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Load JSON or return empty mapping if missing."""
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def _build_combined_df(self) -> pd.DataFrame:
        """Join input and output tables on index."""
        if self.input_df.empty and self.output_df.empty:
            return pd.DataFrame()
        if self.input_df.empty:
            return self.output_df.copy()
        if self.output_df.empty:
            return self.input_df.copy()
        return self.input_df.join(self.output_df, how="left")

    def input_columns(self) -> list[str]:
        """Return available input-column names."""
        return list(self.input_df.columns)

    def output_columns(self) -> list[str]:
        """Return available output-column names."""
        return list(self.output_df.columns)

    def available_columns(self) -> list[str]:
        """Return all available combined-column names."""
        return list(self.combined_df.columns)

    def assert_columns_exist(self, columns: list[str]) -> None:
        """Raise KeyError when one or more columns are not available."""
        available = set(self.combined_df.columns)
        missing = [col for col in columns if col not in available]
        if missing:
            raise KeyError(f"Missing columns: {missing}. Available: {sorted(available)}")

    def get_unit(self, column: str) -> str | None:
        """Return unit for a column, if available."""
        if column in self.output_unit_map:
            return self.output_unit_map.get(column)
        return self.input_unit_map.get(column)

    def get_label(self, column: str) -> str:
        """Return display label for a column, falling back to the column name."""
        label = self.output_label_map.get(column)
        if label:
            return label
        return column

    def format_axis_label(self, column: str) -> str:
        """Return a Matplotlib-friendly axis label with unit if present."""
        label = self.get_label(column)
        unit = self.get_unit(column)
        return f"{label} [{unit}]" if unit else label

    def y_over_x_with_color_and_style(
        self,
        ax: Axes,
        y_col: str = "p_loss",
        x_col: str = "freq",
        color_col: str = "w",
        style_col: str = "b_mean",
    ) -> None:
        """Create a datasheet-style log-log plot for one result over frequency."""
        cols = [y_col, x_col, color_col, style_col]
        self.assert_columns_exist(cols)

        color_values = sorted(self.df[color_col].dropna().unique())
        cmap = plt.get_cmap("viridis", len(color_values))
        value_to_color = {v: cmap(i) for i, v in enumerate(color_values)}

        style_values = sorted(self.df[style_col].dropna().unique())
        linestyles = ["-", "--", "-.", ":"]
        value_to_ls = {v: linestyles[i % len(linestyles)] for i, v in enumerate(style_values)}

        label_points: dict[Any, tuple[float, float]] = {}

        grouped = self.df.sort_values(x_col).groupby([color_col, style_col], sort=True)
        for (color_val, style_val), g in grouped:
            x = g[x_col].to_numpy()
            y = g[y_col].to_numpy()

            ax.plot(
                x,
                y,
                marker="o",
                ms=4,
                lw=1.8,
                alpha=0.95,
                color=value_to_color[color_val],
                linestyle=value_to_ls[style_val],
                label=f"{color_col}={color_val:.3g}",
            )

            i = int(np.argmax(x))
            xp, yp = float(x[i]), float(y[i])
            if style_val not in label_points or (
                xp >= label_points[style_val][0] and yp > label_points[style_val][1]
            ):
                label_points[style_val] = (xp, yp)

        freq_unit = self.input_unit_map.get(x_col)
        style_unit = self.input_unit_map.get(style_col)
        metric_unit = self.output_unit_map.get(y_col)
        metric_label = self.get_label(y_col)

        x_label = self.format_axis_label(x_col) if freq_unit else x_col
        resolved_y_label = (
            self.format_axis_label(y_col)
            if (metric_label or metric_unit)
            else y_col
        )
        style_suffix = "" if not style_unit else f" {style_unit}"

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(resolved_y_label)
        ax.set_title(f"{metric_label} over frequency")
        ax.grid(True, which="both", alpha=0.3)

        color_handles = [
            Line2D(
                [0],
                [0],
                color=value_to_color[color_val],
                linestyle="-",
                lw=2,
                marker="o",
                ms=4,
                label=f"{color_val:.3g}",
            )
            for color_val in color_values
        ]
        color_legend = ax.legend(
            handles=color_handles,
            title=color_col,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        ax.add_artist(color_legend)

        style_handles = [
            Line2D(
                [0],
                [0],
                color="k",
                linestyle=value_to_ls[style_val],
                lw=2,
                label=f"{style_val:.3g}{style_suffix}",
            )
            for style_val in style_values
        ]
        ax.legend(
            handles=style_handles,
            title=style_col,
            bbox_to_anchor=(1.02, 0.45),
            loc="upper left",
        )

        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax * 1.25)

        plt.tight_layout()
