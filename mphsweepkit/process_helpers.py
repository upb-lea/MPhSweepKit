import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any


def load_post_processing_exprs(
    json_path: str | Path, print_info: bool = False
) -> dict[str, dict[str, str]]:
    """
    Load post-processing expressions from a JSON file.

    Parameters
    ----------
    json_path : str | Path
        Path to the JSON file.
    print_info : bool, optional
        If True, print info about the resolved file path and a formatted
        preview of the loaded JSON content, by default False.

    Returns
    -------
    dict[str, dict[str, str]]
        Dictionary like:
        {
          "p_loss": {"expression": "...", "unit": "...", "label": "..."},
          ...
        }

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If the JSON structure is invalid.
    """
    path = Path(json_path)
    if print_info:
        print(f"Loading post-processing expressions from: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if print_info:
        print("Loaded post-processing expressions:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object/dict.")

    required_keys = {"expression", "unit", "label"}
    for name, entry in data.items():
        if not isinstance(name, str):
            raise ValueError("All top-level keys must be strings.")
        if not isinstance(entry, dict):
            raise ValueError(f"Entry '{name}' must be an object/dict.")
        missing = required_keys - set(entry.keys())
        if missing:
            raise ValueError(f"Entry '{name}' is missing keys: {sorted(missing)}")
        for key in required_keys:
            if not isinstance(entry[key], str):
                raise ValueError(f"Entry '{name}' key '{key}' must be a string.")

    return data



def read_fields_on_geometry(subfolder, description, geometry_idx, target_length_unit="m", switch_xy: bool = False) -> tuple[pd.DataFrame, str]:
    """
    Read field data from a text file exported from COMSOL.
    
    :param subfolder: Subfolder where the field data file is located.
    :param description: Description of the field data file.
    :param geometry_idx: Index of the geometry for which the field data is read.
    :param target_length_unit: The unit for the length dimensions.
    :param switch_xy: If True and dimension is 2D/3D, swap x and y coordinate columns.
    :returns: A tuple containing the DataFrame with field data and the length unit.
    """
    dimension = None
    expressions = None
    header = None
    read_length_unit = target_length_unit
    coordinate_header = None

    filename = f"field_data/{subfolder}/geometry_{geometry_idx}_{description}.txt"
    with open(filename) as f:
        for line in f:
            if line.startswith("% Dimension:"):
                dimension = int(line.split(":", 1)[1])
            elif line.startswith("% Expressions:"):
                expressions = int(line.split(":", 1)[1])
            elif line.startswith("% Length unit:"):
                read_length_unit = line.split(":", 1)[1].strip()
            elif line.startswith("%"):
                tokens = line[1:].split()
                if len(tokens) >= 2:
                    first_two = tuple(tok.lower() for tok in tokens[:2])
                    # Cartesian 2D (x y ...), axisymmetric 2D (r z ...), and 1D/3D variants
                    if first_two in {("x", "y"), ("r", "z")} or tokens[0].lower() in {"x", "r"}:
                        header = tokens
                        coordinate_header = list(first_two)

    if type(dimension) is int and type(expressions) is int and type(header) is list:
        # print(f"type dimension={type(dimension)}")
        # print(f"type expressions={type(expressions)}")
        # print(f"type header={type(header)}")

        coordinates = {
            1: ["x"],
            2: ["x", "y"],
            3: ["x", "y", "z"],
        }.get(dimension)

        if coordinates is None:
            raise ValueError("Dimension must be 1, 2, or 3")

        # Detect and normalize axisymmetric coordinates (r, z) -> (x, y)
        if dimension == 2 and coordinate_header == ["r", "z"]:
            coordinates = ["r", "z"]

        words = header[dimension:]
        expression_names = []
        start = 0

        for i, word in enumerate(words):
            if word == "@" and i + 1 < len(words):
                expression_names.append(" ".join(words[start:i + 2]))
                start = i + 2

        if len(expression_names) != expressions:
            raise ValueError("Could not parse all expression names")

        df = pd.read_csv(filename, sep=r"\s+", comment="%", header=None)
        df.columns = coordinates + expression_names

        # Normalize axisymmetric naming for downstream Cartesian plotting helpers
        if dimension == 2 and {"r", "z"}.issubset(df.columns):
            df = df.rename(columns={"r": "x", "z": "y"})

        # Convert the length units of the DataFrame columns
        df = convert_length_unit(df, read_length_unit, target_length_unit)

        if switch_xy and {"x", "y"}.issubset(df.columns):
            df[["x", "y"]] = df[["y", "x"]].to_numpy()

        return df, target_length_unit

    else:
        raise ValueError(f"Could not parse file header for dimension {dimension}, expressions {expressions}, and header {header}")


def convert_length_unit(df: pd.DataFrame, length_unit: str, target_unit: str) -> pd.DataFrame:
    """
    Convert the length units of the DataFrame columns.

    :param df: DataFrame containing the field data.
    :param length_unit: Current length unit of the DataFrame.
    :param target_unit: Target length unit to convert to.
    :returns: DataFrame with converted length units.
    """
    conversion_factors = {
        ("m", "mm"): 1000.0,
        ("mm", "m"): 0.001,
        ("m", "cm"): 100.0,
        ("cm", "m"): 0.01,
        ("mm", "cm"): 0.1,
        ("cm", "mm"): 10.0,
    }

    if (length_unit, target_unit) not in conversion_factors:
        raise ValueError(f"Conversion from {length_unit} to {target_unit} is not supported.")

    factor = conversion_factors[(length_unit, target_unit)]
    for col in df.columns:
        if col in ["x", "y", "z", "r"]:
            df[col] *= factor

    return df


def _normalize_filter_scalar(value: Any) -> float | str:
    """
    Normalize a filter value for robust numeric/string comparison.

    :param value: Raw filter value.
    :returns: `float` if conversion is possible, otherwise stripped `str`.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value).strip()


def _filter_df_by_conditions(df: pd.DataFrame, conditions: dict[str, Any]) -> pd.DataFrame:
    """
    Filter a DataFrame with mixed-type tolerant comparisons.

    Scalar condition values are matched numerically when possible, otherwise as
    stripped strings. Iterable condition values are treated as OR-conditions and
    can mix numeric and string values.

    :param df: DataFrame to filter.
    :param conditions: Mapping of column names to scalar or iterable filter values.
    :returns: Filtered DataFrame.
    :raises KeyError: If a condition column does not exist in the DataFrame.
    """
    missing = [col for col in conditions if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    filtered_df = df.copy()

    for col, raw_val in conditions.items():
        col_as_num = pd.to_numeric(filtered_df[col], errors="coerce")

        if isinstance(raw_val, (list, tuple, set, np.ndarray, pd.Series)):
            values = list(raw_val)
            num_values: list[float] = []
            str_values: list[str] = []

            for v in values:
                normalized = _normalize_filter_scalar(v)
                if isinstance(normalized, (int, float, np.floating)):
                    num_values.append(float(normalized))
                else:
                    str_values.append(str(normalized))

            value_mask = pd.Series(False, index=filtered_df.index)

            if col_as_num.notna().any() and num_values:
                value_mask |= col_as_num.isin(num_values)

            if str_values:
                col_as_str = filtered_df[col].astype(str).str.strip()
                value_mask |= col_as_str.isin(str_values)

            filtered_df = filtered_df[value_mask]
        else:
            normalized = _normalize_filter_scalar(raw_val)
            if col_as_num.notna().any() and isinstance(normalized, (int, float, np.floating)):
                filtered_df = filtered_df[col_as_num == float(normalized)]
            else:
                filtered_df = filtered_df[filtered_df[col].astype(str).str.strip() == str(normalized)]

    return filtered_df

def get_geometry_idx_from_filter(df_input: pd.DataFrame, filter_conditions: dict) -> int:
    """
    Get the geometry index from the input dataframe based on the provided filter conditions.

    :param df_input: Input dataframe containing geometry information.
    :param filter_conditions: Dictionary containing filter conditions where keys are column names and values are the
    :return: The geometry index that matches the filter conditions.
    """
    filtered_df = _filter_df_by_conditions(df_input, filter_conditions)

    # check if the filtered dataframe is not empty
    if filtered_df.empty:
        raise ValueError("No matching geometry index found for the given filter conditions.")

    # check if there is more than one matching geometry index
    if len(set(filtered_df["geometry_idx"])) > 1:
        raise ValueError("Multiple matching geometry indices found for the given filter conditions. Please refine your filter.")

    # return the geometry index of the first (and only) matching row
    return filtered_df["geometry_idx"].iloc[0]


def get_internal_idx_from_filters(
    combined_df: pd.DataFrame,
    geometry_idx: int,
    extra_filters: dict,
    *,
    geometry_col: str = "geometry_idx",
    internal_idx_col: str = "internal_idx",
) -> list[int]:
    """
    Return `internal_idx` value(s) for a selected geometry and additional filters.

    :param combined_df: DataFrame containing at least geometry and internal index columns.
    :param geometry_idx: Selected geometry index (for example from `get_geometry_idx_from_filter`).
    :param extra_filters: Dictionary of additional filter conditions (e.g. material, excitation, frequency).
        Keys are column names, values are either scalar values (exact match) or iterables (membership match).
    :param geometry_col: Column name in `combined_df` that stores geometry indices.
    :param internal_idx_col: Column name in `combined_df` that stores internal indices.
    :returns: All matching `internal_idx` values as `list[int]`.
    :raises KeyError: If one or more required columns are missing in `combined_df`.
    :raises ValueError: If no match is found.
    """
    required_cols = {geometry_col, internal_idx_col, *extra_filters.keys()}
    missing = [c for c in required_cols if c not in combined_df.columns]
    if missing:
        raise KeyError(f"Missing columns in combined_df: {missing}")

    all_conditions = {geometry_col: geometry_idx, **extra_filters}
    filtered_df = _filter_df_by_conditions(combined_df, all_conditions)

    matches = filtered_df[internal_idx_col].dropna().astype(int).tolist()

    if len(matches) == 0:
        raise ValueError(
            f"No match for geometry_idx={geometry_idx} and filters={extra_filters}"
        )
    return matches


def scale_field_columns(
    df_fields: pd.DataFrame,
    scale_factor: float,
    *,
    coordinate_names: tuple[str, ...] = ("x", "y", "z", "r"),
) -> pd.DataFrame:
    """
    Scale all non-coordinate field value columns in-place.

    :param df_fields: Field dataframe to modify.
    :param scale_factor: Multiplicative scale factor.
    :param coordinate_names: Coordinate column names excluded from scaling.
    :returns: The modified dataframe (same object).
    """
    if scale_factor == 1:
        return df_fields

    coord_set = {name.strip().lower() for name in coordinate_names}
    for col in df_fields.columns:
        if str(col).strip().lower() in coord_set:
            continue
        df_fields[col] = df_fields[col] * scale_factor

    return df_fields

def _count_coordinate_columns(df: pd.DataFrame, coord_candidates: set[str] | None = None) -> int:
    """
    Return the number of coordinate-related columns in a dataframe.

    :param df : DataFrame containing coordinate and field columns.
    :param coord_candidates : Coordinate names to match case-insensitively. Defaults to {"x","y","z","r"}.
    :returns: Count of coordinate columns.
    """
    if coord_candidates is None:
        coord_candidates = {"x", "y", "z", "r"}

    return sum(
        1
        for c in df.columns
        if str(c).strip().lower() in coord_candidates
    )



def get_list_of_col_names(df: pd.DataFrame, list_internal_idx: list[int]) -> list[str]:
    """
    Get the list of value columns names corresponding to the selected internal indices.

    The function assumes that the first N columns of the DataFrame are coordinate 
    columns (e.g., "x", "y", "z", "r"). These coordinate columns are skipped when 
    determining the value columns.

    :param df: DataFrame containing the field data.
    :param list_internal_idx: List of internal indices for which to retrieve the column names.
    :returns: List of column names corresponding to the selected internal indices.
    """
    field_start_idx = _count_coordinate_columns(df)
    if field_start_idx == 0:
        raise ValueError("No coordinate columns detected in field dataframe.")
    
    return [df.columns[field_start_idx + idx] for idx in list_internal_idx]
