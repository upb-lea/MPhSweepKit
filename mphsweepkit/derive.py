"""This module provides functions to derive new columns from existing data columns in DataFrame with Multiindex."""
import numpy as np
import pandas as pd

from .meta_data import METADATA_ROW_NAMES


def _as_complex_series(df, source_col):
    return pd.to_numeric(df[source_col], errors="coerce").astype(complex)


def _data_mask(df):
    return ~df.index.astype(str).str.lower().isin(METADATA_ROW_NAMES)


def _init_derived_column(df, col):
    # object dtype allows both numeric data + metadata strings
    df[col] = pd.Series(np.nan, index=df.index, dtype="object")


def _write_data_values(df, col, values):
    mask = _data_mask(df)
    df.loc[mask, col] = pd.Series(values, index=df.index).loc[mask].to_numpy()


def _set_meta(df, col, unit, group="Derived"):
    df.loc["name", col] = col
    df.loc["unit", col] = unit
    df.loc["group", col] = group


def _add_derived(df, source_col, new_col, unit, transform, group="Derived"):
    z = _as_complex_series(df, source_col)
    values = transform(z)

    _init_derived_column(df, new_col)
    _write_data_values(df, new_col, values)
    _set_meta(df, new_col, unit, group=group)
    return df


def add_abs(df, source_col, new_col, unit, group="Derived"):
    return _add_derived(df, source_col, new_col, unit, np.abs, group=group)


def add_phase_rad(df, source_col, new_col, unit="rad", group="Derived"):
    return _add_derived(df, source_col, new_col, unit, np.angle, group=group)


def add_phase_deg(df, source_col, new_col, unit="deg", group="Derived"):
    return _add_derived(
        df, source_col, new_col, unit, lambda z: np.degrees(np.angle(z)), group=group
    )


def add_real(df, source_col, new_col, unit, group="Derived"):
    return _add_derived(df, source_col, new_col, unit, np.real, group=group)


def add_imag(df, source_col, new_col, unit, group="Derived"):
    return _add_derived(df, source_col, new_col, unit, np.imag, group=group)