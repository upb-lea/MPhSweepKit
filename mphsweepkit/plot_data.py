from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path
import json

import matplotlib.patheffects as pe
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .meta_data import METADATA_ROW_NAMES


@dataclass(slots=True)
class PlotSettings:
    """Configurable plot settings for grouped line plots."""

    x_scale: str = "log"
    y_scale: str = "log"
    show_grid: bool = True
    grid_which: str = "both"
    grid_alpha: float = 0.3
    marker: str = "o"
    marker_size: float = 4
    line_width: float = 1.8
    color_map_name: str = "viridis"
    line_styles: tuple[str, ...] = ("-", "--", "-.", ":")
    show_color_legend: bool = True
    show_style_legend: bool = True
    color_legend_anchor: tuple[float, float] = (1.02, 1)
    style_legend_anchor: tuple[float, float] = (1.02, 0.45)
    legend_loc: str = "upper left"
    use_tight_layout: bool = True


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

    def _resolve_column_key(self, column: str) -> Any:
        """Resolve a user-facing column name to an actual dataframe column key.

        Supports both flat columns and MultiIndex columns where the first level
        stores the semantic column name.
        """
        if column in self.combined_df.columns:
            return column

        if isinstance(self.combined_df.columns, pd.MultiIndex):
            hits = [c for c in self.combined_df.columns if len(c) > 0 and c[0] == column]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                # Prefer output-like columns with no group (4th level None)
                for c in hits:
                    if len(c) >= 4 and c[3] is None:
                        return c
                return hits[0]

        raise KeyError(f"Missing column: {column}. Available: {list(self.combined_df.columns)}")

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

    @staticmethod
    def _fmt_legend_value(value: Any) -> str:
        """Format value for legend labels, preferring compact numeric formatting."""
        try:
            return f"{float(value):.3g}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _sort_key(value: Any) -> tuple[int, float | str]:
        """Sort numerically when possible, otherwise lexicographically."""
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))

    @staticmethod
    def _read_meta_value(frame: pd.DataFrame, key: Any, row_name: str) -> str | None:
        """Read metadata value from a specific dataframe row and column key."""
        try:
            value = frame.loc[row_name, key]
        except Exception:
            return None
        if pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    def _legend_title_and_unit(self, column: str, key: Any) -> tuple[str, str | None]:
        """Return legend title and unit for a semantic column name and resolved key."""
        unit = self._read_meta_value(self.combined_df, key, "unit") or self.get_unit(column)
        title = f"{self.get_label(column)} [{unit}]" if unit else self.get_label(column)
        return title, unit

    def _prepare_plot_df(self, x_key: Any, y_key: Any, color_key: Any, style_key: Any) -> pd.DataFrame:
        """Return cleaned dataframe used for plotting grouped lines."""
        df = self.combined_df.copy()
        df = df.loc[~df.index.astype(str).str.lower().isin(METADATA_ROW_NAMES)]
        df[x_key] = pd.to_numeric(df[x_key], errors="coerce")
        df[y_key] = pd.to_numeric(df[y_key], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[x_key, y_key, color_key, style_key])
        return df

    def _build_style_maps(self, df: pd.DataFrame, color_key: Any, style_key: Any) -> tuple[list[Any], list[Any], dict[Any, Any], dict[Any, str]]:
        """Create ordered color/style values and corresponding matplotlib maps."""
        color_values = sorted(df[color_key].unique(), key=self._sort_key)
        style_values = sorted(df[style_key].unique(), key=self._sort_key)

        settings = getattr(self, "_active_plot_settings", PlotSettings())

        cmap = plt.get_cmap(settings.color_map_name, len(color_values))
        color_map = {value: cmap(i) for i, value in enumerate(color_values)}

        linestyles = list(settings.line_styles) if settings.line_styles else ["-"]
        style_map = {value: linestyles[i % len(linestyles)] for i, value in enumerate(style_values)}
        return color_values, style_values, color_map, style_map

    def _add_legends(
        self,
        ax: Axes,
        color_values: list[Any],
        style_values: list[Any],
        color_map: dict[Any, Any],
        style_map: dict[Any, str],
        color_title: str,
        style_title: str,
        color_unit: str | None,
        style_unit: str | None,
        settings: PlotSettings,
    ) -> None:
        """Add separate color and style legends to the axis."""
        color_suffix = f" {color_unit}" if color_unit else ""
        style_suffix = f" {style_unit}" if style_unit else ""

        if settings.show_color_legend:
            color_handles = [
                Line2D(
                    [0],
                    [0],
                    color=color_map[value],
                    marker=settings.marker,
                    lw=2,
                    ms=settings.marker_size,
                    label=f"{self._fmt_legend_value(value)}{color_suffix}",
                )
                for value in color_values
            ]
            color_legend = ax.legend(
                handles=color_handles,
                title=color_title,
                bbox_to_anchor=settings.color_legend_anchor,
                loc=settings.legend_loc,
            )
            ax.add_artist(color_legend)

        if settings.show_style_legend:
            style_handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle=style_map[value],
                    lw=2,
                    label=f"{self._fmt_legend_value(value)}{style_suffix}",
                )
                for value in style_values
            ]
            ax.legend(
                handles=style_handles,
                title=style_title,
                bbox_to_anchor=settings.style_legend_anchor,
                loc=settings.legend_loc,
            )

    def y_over_x_with_color_and_style(
        self,
        ax: Axes,
        y_col: str = "p_loss",
        x_col: str = "freq",
        color_col: str = "w",
        style_col: str = "b_mean",
        settings: PlotSettings | None = None,
    ) -> None:
        """Plot y(x) grouped by color/style columns with simple legends."""
        settings = settings or PlotSettings()
        self._active_plot_settings = settings

        x_key = self._resolve_column_key(x_col)
        y_key = self._resolve_column_key(y_col)
        color_key = self._resolve_column_key(color_col)
        style_key = self._resolve_column_key(style_col)

        color_title, color_unit = self._legend_title_and_unit(color_col, color_key)
        style_title, style_unit = self._legend_title_and_unit(style_col, style_key)

        df = self._prepare_plot_df(x_key, y_key, color_key, style_key)
        if df.empty:
            return

        color_values, style_values, color_map, style_map = self._build_style_maps(df, color_key, style_key)

        for (cval, sval), g in df.groupby([color_key, style_key], sort=True):
            g = g.sort_values(x_key)
            ax.plot(
                g[x_key].to_numpy(float),
                g[y_key].to_numpy(float),
                marker=settings.marker,
                ms=settings.marker_size,
                lw=settings.line_width,
                color=color_map[cval],
                linestyle=style_map[sval],
            )

        ax.set_xscale(settings.x_scale)
        ax.set_yscale(settings.y_scale)
        ax.set_xlabel(self.format_axis_label(x_col))
        ax.set_ylabel(self.format_axis_label(y_col))
        ax.set_title(f"{self.get_label(y_col)} over {self.get_label(x_col)}")
        if settings.show_grid:
            ax.grid(True, which=settings.grid_which, alpha=settings.grid_alpha)

        self._add_legends(
            ax=ax,
            color_values=color_values,
            style_values=style_values,
            color_map=color_map,
            style_map=style_map,
            color_title=color_title,
            style_title=style_title,
            color_unit=color_unit,
            style_unit=style_unit,
            settings=settings,
        )

        if settings.use_tight_layout:
            plt.tight_layout()