import itertools
from typing import Any
import numpy as np
import pandas as pd

# Import of meta data
from .meta_names import ALLOWED_SWEEP_TYPES

# Import from helper module
from mphsweepkit.sweep_get_set import get_parametric_sweep_data, get_material_sweep_data, get_frequency_sweep_data


def get_cascaded_dataset(
    list_of_sweep_data: list[dict],
    list_of_sweep_types: list[str | None],
    list_of_sweep_names: list[str],
) -> tuple[pd.DataFrame, dict[str, str | None], dict[str, str | None]]:
    """
    Build a cascaded dataset for an arbitrary number of sweep loops.

    :param list_of_sweep_data: List of sweep property dictionaries, one per sweep level.
    :param list_of_sweep_types: List of sweep types corresponding to each sweep level.
    :param list_of_sweep_names: List of sweep names corresponding to each sweep level.
    :return:
        - DataFrame with all cascaded combinations
        - units_map: {parameter_name: unit}
        - sweep_map: {parameter_name: sweep_name}
    """
    _validate_inputs(list_of_sweep_data, list_of_sweep_types, list_of_sweep_names)

    units_map: dict[str, str | None] = {}
    sweep_map: dict[str, str | None] = {}
    expanded_levels: list[list[dict[str, Any]]] = []

    for raw_data, sweep_type, sweep_name in zip(
        list_of_sweep_data, list_of_sweep_types, list_of_sweep_names
    ):
        if sweep_type == "CoilCurrentCalculation":
            # The CoilCurrentCalculation sweep type does not add any additional parameters.
            continue
        elif sweep_type == "Frequency":
            sweep_data = get_frequency_sweep_data(raw_data)
            level_records, freq_unit = _expand_frequency_sweep(sweep_data)
            expanded_levels.append(level_records)
            units_map.setdefault("freq", _normalize_unit(freq_unit))
            sweep_map.setdefault("freq", sweep_name)

        elif sweep_type == "MaterialSweep":
            sweep_data = get_material_sweep_data(raw_data)
            _record_units_non_frequency(sweep_data, units_map)
            expanded_levels.append(_expand_non_frequency_sweep(sweep_data))
            sweep_map.update({pname: sweep_name for pname in sweep_data.get("pname", [])})

        else:  # BatchSweep, Parametric
            sweep_data = get_parametric_sweep_data(raw_data)
            _record_units_non_frequency(sweep_data, units_map)
            expanded_levels.append(_expand_non_frequency_sweep(sweep_data))
            sweep_map.update({pname: sweep_name for pname in sweep_data.get("pname", [])})

    records = _combine_expanded_levels(expanded_levels)
    return pd.DataFrame(records), units_map, sweep_map


# ============================================================
# Helpers
# ============================================================
def _validate_inputs(
    list_of_sweep_data: list[dict],
    list_of_sweep_types: list[str | None],
    list_of_sweep_names: list[str],
) -> None:
    if len(list_of_sweep_data) != len(list_of_sweep_types):
        raise ValueError("list_of_sweep_data and list_of_sweep_types must have the same length.")
    if len(list_of_sweep_data) != len(list_of_sweep_names):
        raise ValueError("list_of_sweep_data and list_of_sweep_names must have the same length.")

    invalid = [t for t in list_of_sweep_types if t not in ALLOWED_SWEEP_TYPES]
    if invalid:
        raise ValueError(f"Invalid sweep types: {invalid}. Allowed: {sorted(ALLOWED_SWEEP_TYPES)}")


def _normalize_unit(unit: Any) -> str | None:
    if unit is None:
        return None
    if isinstance(unit, str):
        return unit
    return ""


def _record_units_non_frequency(
    sweep_data: dict[str, Any],
    units_map: dict[str, str | None],
) -> None:
    pnames = sweep_data.get("pname", [])
    punits = sweep_data.get("punit", None)

    if punits is None:
        for name in pnames:
            units_map.setdefault(name, "")
        return

    if isinstance(punits, str) or not hasattr(punits, "__len__"):
        unit_val = _normalize_unit(punits)
        for name in pnames:
            units_map.setdefault(name, unit_val if unit_val is not None else "")
        return

    for i, name in enumerate(pnames):
        unit_val = punits[i] if i < len(punits) else ""
        units_map.setdefault(name, _normalize_unit(unit_val))


def _expand_non_frequency_sweep(sweep_data: dict[str, Any]) -> list[dict[str, Any]]:
    parameter_names = sweep_data["pname"]
    parameter_values = np.asarray(sweep_data["plistarr"], dtype=object)
    sweep_type = sweep_data["sweeptype"]

    if sweep_type == "sparse":
        return [dict(zip(parameter_names, values)) for values in zip(*parameter_values)]

    value_lists = [np.ravel(row).tolist() for row in parameter_values]
    return [
        dict(zip(parameter_names, combination))
        for combination in itertools.product(*value_lists)
    ]


def _expand_frequency_sweep(sweep_data: dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
    freq_values = np.ravel(sweep_data["plist"]).tolist()
    records = [{"freq": f} for f in freq_values]
    freq_unit = sweep_data.get("punit", "")
    return records, freq_unit


def _combine_expanded_levels(
    expanded_levels: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for combo in itertools.product(*expanded_levels):
        merged: dict[str, Any] = {}
        for level_record in combo:
            merged.update(level_record)
        records.append(merged)

    return records