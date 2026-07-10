"""Shared constants for metadata handling and plotting."""

from __future__ import annotations

# Row names that may appear as metadata rows in exported result tables.
METADATA_ROW_NAMES: frozenset[str] = frozenset({"name","unit", "group"})
    