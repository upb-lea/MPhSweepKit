"""Shared constants for metadata handling and plotting."""

from __future__ import annotations

# Row names that may appear as metadata rows in exported result tables.
METADATA_ROW_NAMES: tuple[str, str, str] = ("name", "unit", "group")
ALLOWED_SWEEP_TYPES: set[str] = {"BatchSweep", "MaterialSweep", "Parametric", "Frequency", "CoilCurrentCalculation"}
