from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast
from pathlib import Path

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
    grid_which: Literal["major", "minor", "both"] = "both"
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
    legend_loc: Literal[
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    ] = "upper left"
    use_tight_layout: bool = True


@dataclass(slots=True)
class PlotTextOverrides:
    """Optional label/unit overrides for axes and legends."""

    x_label: str | None = None
    x_unit: str | None = None
    y_label: str | None = None
    y_unit: str | None = None
    color_label: str | None = None
    color_unit: str | None = None
    style_label: str | None = None
    style_unit: str | None = None


class DataPlot:
    """Plot helper for scalar metrics stored in result-data tables.

    The class is designed to load data exported by
    ``CascadedSweepModel.save_result_data``.
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        folder: str | Path | None = None,
    ):
        self.folder = Path(folder) if folder is not None else None

        self.input_df = input_df
        self.output_df = output_df

        self.combined_df = self._build_combined_df()

    @classmethod
    def from_result_folder(
        cls,
        folder: str | Path,
        index_col: int | str | None = 0,
    ) -> "DataPlot":
        """Load result tables from a result-data folder."""
        root = Path(folder)

        input_path = root / "input_data.csv"
        output_path = root / "output_data.csv"

        if not root.exists():
            raise FileNotFoundError(f"Result folder does not exist: {root}")

        if not input_path.exists() and not output_path.exists():
            raise FileNotFoundError(
                f"Neither input_data.csv nor output_data.csv found in: {root}"
            )

        input_df = cls._load_csv(input_path, index_col=index_col)
        output_df = cls._load_csv(output_path, index_col=index_col)

        return cls(
            input_df=input_df,
            output_df=output_df,
            folder=root,
        )

    @staticmethod
    def _load_csv(path: Path, index_col: int | str | None = 0) -> pd.DataFrame:
        """Load CSV or return empty dataframe if missing."""
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, index_col=index_col)

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

    def filter_rows(
        self,
        filters: dict[str, list[Any] | tuple[Any, ...] | set[Any]],
        *,
        numeric_match: bool = True,
    ) -> "DataPlot":
        """Return a filtered ``DataPlot`` using AND-combined column filters.

        Parameters
        ----------
        filters:
            Mapping from semantic column name (e.g. ``"freq"``) to accepted
            values. Values are matched with OR within each column.
        numeric_match:
            When ``True``, attempts numeric matching via ``pd.to_numeric`` in
            addition to exact/object matching. This makes filters robust against
            string-vs-numeric storage differences.
        """
        if not filters:
            return DataPlot(
                input_df=self.input_df.copy(),
                output_df=self.output_df.copy(),
                folder=self.folder,
            )

        if self.combined_df.empty:
            return DataPlot(
                input_df=self.input_df.copy(),
                output_df=self.output_df.copy(),
                folder=self.folder,
            )

        plot_df = self.combined_df.copy()
        plot_df = plot_df.loc[~plot_df.index.astype(str).str.lower().isin(METADATA_ROW_NAMES)]

        if plot_df.empty:
            return DataPlot(
                input_df=self.input_df.iloc[0:0].copy(),
                output_df=self.output_df.iloc[0:0].copy(),
                folder=self.folder,
            )

        mask = pd.Series(True, index=plot_df.index)

        for column, accepted_values in filters.items():
            key = self._resolve_column_key(column)
            values = list(accepted_values)

            col_mask = plot_df[key].isin(values)

            if numeric_match:
                col_numeric = pd.to_numeric(plot_df[key], errors="coerce")
                vals_numeric = pd.to_numeric(pd.Series(values), errors="coerce")
                vals_numeric = vals_numeric.dropna().tolist()
                if vals_numeric:
                    col_mask = col_mask | col_numeric.isin(vals_numeric)

            mask = mask & col_mask

        selected_index = plot_df.index[mask]

        filtered_input = self.input_df.loc[self.input_df.index.intersection(selected_index)].copy()
        filtered_output = self.output_df.loc[self.output_df.index.intersection(selected_index)].copy()

        return DataPlot(
            input_df=filtered_input,
            output_df=filtered_output,
            folder=self.folder,
        )

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

    def _resolve_meta_column_key(self, column: str) -> Any | None:
        """Resolve column to key for metadata-row lookup; return None when missing."""
        try:
            return self._resolve_column_key(column)
        except KeyError:
            return None

    def _get_meta_text(self, column: str, row_name: str) -> str | None:
        """Read metadata-row text for a semantic column name."""
        key = self._resolve_meta_column_key(column)
        if key is None:
            return None
        return self._read_meta_value(self.combined_df, key, row_name)

    def _label_and_unit(self, column: str) -> tuple[str, str | None]:
        """Return display label and unit from metadata rows."""
        label = self._get_meta_text(column, "label") or column
        unit = self._get_meta_text(column, "unit")
        return label, unit

    def get_unit(self, column: str) -> str | None:
        """Return unit for a column from metadata rows, if available."""
        _, unit = self._label_and_unit(column)
        return unit

    def get_label(self, column: str) -> str:
        """Return display label from metadata rows, falling back to column name."""
        label, _ = self._label_and_unit(column)
        return label

    def format_axis_label(self, column: str) -> str:
        """Return a Matplotlib-friendly axis label with unit if present."""
        label, unit = self._label_and_unit(column)
        return label + " [" + unit + "]" if unit else label

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
        label, unit = self._label_and_unit(column)
        title = label
        return title, unit

    @staticmethod
    def _format_label_with_unit(label: str, unit: str | None) -> str:
        """Return display label with optional unit suffix."""
        return label + " [" + unit + "]" if unit else label

    @staticmethod
    def _apply_text_override(
        base_label: str,
        base_unit: str | None,
        override_label: str | None,
        override_unit: str | None,
    ) -> tuple[str, str | None]:
        """Apply optional label/unit overrides and return effective values."""
        return (
            override_label or base_label,
            base_unit if override_unit is None else override_unit,
        )

    def _prepare_plot_df(self, x_key: Any, y_key: Any, color_key: Any, style_key: Any) -> pd.DataFrame:
        """Return cleaned dataframe used for plotting grouped lines."""
        df = self.combined_df.copy()
        df = df.loc[~df.index.astype(str).str.lower().isin(METADATA_ROW_NAMES)]
        df[x_key] = pd.to_numeric(df[x_key], errors="coerce")
        df[y_key] = pd.to_numeric(df[y_key], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[x_key, y_key, color_key, style_key])
        return df

    def _build_style_maps(
        self,
        df: pd.DataFrame,
        color_key: Any,
        style_key: Any,
        settings: PlotSettings,
    ) -> tuple[list[Any], list[Any], dict[Any, Any], dict[Any, str]]:
        """Create ordered color/style values and corresponding matplotlib maps."""
        color_values = sorted(df[color_key].unique(), key=self._sort_key)
        style_values = sorted(df[style_key].unique(), key=self._sort_key)

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
                    linestyle=cast(Any, style_map[value]),
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
        filters: dict[str, list[Any] | tuple[Any, ...] | set[Any]] | None = None,
        settings: PlotSettings | None = None,
        text_overrides: PlotTextOverrides | None = None,
    ) -> None:
        """Plot y(x) grouped by color/style columns with simple legends."""
        settings = settings or PlotSettings()
        text_overrides = text_overrides or PlotTextOverrides()

        x_key = self._resolve_column_key(x_col)
        y_key = self._resolve_column_key(y_col)
        color_key = self._resolve_column_key(color_col)
        style_key = self._resolve_column_key(style_col)

        color_title, color_unit = self._legend_title_and_unit(color_col, color_key)
        style_title, style_unit = self._legend_title_and_unit(style_col, style_key)

        x_label, x_unit = self._label_and_unit(x_col)
        y_label, y_unit = self._label_and_unit(y_col)

        x_label, x_unit = self._apply_text_override(
            x_label,
            x_unit,
            text_overrides.x_label,
            text_overrides.x_unit,
        )
        y_label, y_unit = self._apply_text_override(
            y_label,
            y_unit,
            text_overrides.y_label,
            text_overrides.y_unit,
        )

        color_title, color_unit = self._apply_text_override(
            color_title,
            color_unit,
            text_overrides.color_label,
            text_overrides.color_unit,
        )
        style_title, style_unit = self._apply_text_override(
            style_title,
            style_unit,
            text_overrides.style_label,
            text_overrides.style_unit,
        )

        # If filters are provided, plot from a temporary filtered view of this DataPlot.
        source = self.filter_rows(filters) if filters else self
        df = source._prepare_plot_df(x_key, y_key, color_key, style_key)
        if df.empty:
            return

        color_values, style_values, color_map, style_map = self._build_style_maps(
            df,
            color_key,
            style_key,
            settings,
        )

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
        ax.set_xlabel(self._format_label_with_unit(x_label, x_unit))
        ax.set_ylabel(self._format_label_with_unit(y_label, y_unit))
        ax.set_title(f"{y_label} over {x_label}")
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