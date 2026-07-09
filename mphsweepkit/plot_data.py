from __future__ import annotations

from typing import Any
from pathlib import Path
import json

import matplotlib.patheffects as pe
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
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

    @property
    def df(self) -> pd.DataFrame:
        """Backward-compatible alias for the combined dataframe."""
        return self.combined_df

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
