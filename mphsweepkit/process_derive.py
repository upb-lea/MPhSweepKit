"""This module provides functions to derive new columns from existing data columns in DataFrame with Multiindex."""
import numpy as np
import pandas as pd

# Import of meta data
from .meta_names import METADATA_ROW_NAMES


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


# General single-column derived data functions
def from_single_column(df, source_col, new_col, unit, transform, group="Derived"):
    """
    transform gets a complex Series and returns a numeric Series.
    Example transform: lambda z: np.real(z) + np.imag(z)
    """
    return _add_derived(df, source_col, new_col, unit, transform, group=group)


# General multi-column derived data functions
def from_multiple_columns(df, source_cols, new_col, unit, func, group="Derived"):
    """
    func gets a list of complex Series in the same order as source_cols.
    Example func: lambda cols: np.real(cols[0]) + np.real(cols[1])
    """
    numeric_cols = [pd.to_numeric(df[c], errors="coerce") for c in source_cols]

    has_complex_input = any(
        np.any(np.abs(np.imag(col.to_numpy(dtype=complex, copy=False))) > 0)
        for col in numeric_cols
    )

    if has_complex_input:
        cols = [col.astype(complex) for col in numeric_cols]
    else:
        cols = [col.astype(float) for col in numeric_cols]

    values = func(cols)

    _init_derived_column(df, new_col)
    _write_data_values(df, new_col, values)
    _set_meta(df, new_col, unit, group=group)
    return df


# Exemplary single-column derived data functions
def abs(df, source_col, new_col, unit, group="Derived"):
    return _add_derived(df, source_col, new_col, unit, np.abs, group=group)


def phase_rad(df, source_col, new_col, unit="rad", group="Derived"):
    return _add_derived(df, source_col, new_col, unit, np.angle, group=group)


def phase_deg(df, source_col, new_col, unit="deg", group="Derived", change_sign=False):
    sign = -1 if change_sign else 1
    return _add_derived(
        df, source_col, new_col, unit, lambda z: sign * np.degrees(np.angle(z)), group=group
    )


def real(df, source_col, new_col, unit, group="Derived", change_sign=False):
    sign = -1 if change_sign else 1
    return _add_derived(df, source_col, new_col, unit, lambda z: sign * np.real(z), group=group)



def imag(df, source_col, new_col, unit, group="Derived", change_sign=False):
    sign = -1 if change_sign else 1
    return _add_derived(df, source_col, new_col, unit, lambda z: sign * np.imag(z), group=group)
