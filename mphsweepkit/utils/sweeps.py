import numpy as np
from numpy.typing import NDArray
import pandas as pd
import itertools
import mph


def get_sweep_data(sweep_params: dict) -> dict:
    """
    Explore the sweep data and return relevant information.

    :param sweep_params: Dictionary containing sweep parameters
    :return: A dictionary containing the sweep data.
    """
    return {
        "pname": sweep_params["pname"],
        "punit": sweep_params["punit"],
        "plistarr": sweep_params["plistarr"],
        "sweeptype": sweep_params["sweeptype"],
        "plistarr_shape": sweep_params["plistarr"].shape,
    }


def set_parametric_sweep(
    sweep_obj: mph.Node,
    param_names: list,
    param_units: list,
    param_values: NDArray[np.float64],
    sweep_type: str = "sparse",
):
    """
    Set the parameter names and values for a given sweep object.

    IMPORTANT: Use all floats for the parameter values to avoid issues with COMSOL's handling of data types.

    :param sweep_obj: The sweep object to set parameters for.
    :param param_names: List of parameter names.
    :param param_units: List of parameter units corresponding to the parameter names.
    :param param_values: List of parameter values corresponding to the parameter names.
    :param sweep_type: Type of sweep ('sparse' or 'filled').
    """
    sweep_obj.property("pname", param_names)
    sweep_obj.property("punit", param_units)
    sweep_obj.property("plistarr", param_values)
    sweep_obj.property("sweeptype", sweep_type)


def get_material_data(sweep_materials: dict) -> dict:
    """
    Return the material properties defined in a material sweep block.

    :param sweep_materials: Dictionary containing material sweep properties.
    :return: A dictionary containing the material data.
    """
    return {
        "pname": sweep_materials["pname"],
        "plistarr": sweep_materials["plistarr"],
        "sweeptype": sweep_materials["sweeptype"]
    }

def set_material_sweep(
    sweep_obj: mph.Node,
    material_names: list,
    material_list: NDArray[np.float64],
    sweep_type: str = "sparse"):
    """
    Set the material names and sweep type for a given sweep object.

    :param sweep_obj: The sweep object to set parameters for.
    :param material_names: List of material names.
    :param material_list: List of material values.
    :param sweep_type: Type of sweep ('sparse' or 'filled').
    """
    sweep_obj.property("pname", material_names)
    sweep_obj.property("plistarr", material_list)
    sweep_obj.property("sweeptype", sweep_type)


def get_frequency_data(frequency_properties: dict) -> dict:
    """
    Return the frequencies defined in a frequency-domain sweep block.

    :param frequency_properties: Dictionary containing frequency sweep properties.
    :return: A dictionary containing the frequency data.
    """
    return {
        "frequencies": np.ravel(frequency_properties["plist"]).tolist(),
        "unit": frequency_properties["punit"],
    }

def set_frequency_sweep(
    sweep_obj: mph.Node,
    frequency_values: NDArray[np.float64],
    frequency_unit: str = "Hz"
):
    """
    Set the frequency values and unit for a given frequency sweep object.

    :param sweep_obj: The sweep object to set parameters for.
    :param frequency_values: List of frequency values.
    :param frequency_unit: Unit of the frequency values (default is "Hz").
    """
    sweep_obj.property("plist", frequency_values)
    sweep_obj.property("punit", frequency_unit)


def get_cascaded_dataset(
    geometry_sweep_data: dict, material_sweep_data: dict, excitation_sweep_data: dict, frequency_sweep_data: dict
) -> pd.DataFrame:
    """
    Build the full cascaded dataset for geometry -> material -> excitation -> frequency.

    Sparse sweeps are expanded case-by-case, filled sweeps are expanded as a
    cartesian product, and the resulting records are combined in cascade order.

    :param geometry_sweep_data: Geometry sweep data returned by get_sweep_data.
    :param material_sweep_data: Material sweep data returned by get_sweep_data.
    :param excitation_sweep_data: Excitation sweep data returned by get_sweep_data.
    :param frequency_sweep_data: Frequency data returned by get_frequency_data.
    :return: DataFrame containing all parameter combinations.
    """

    def _expand_sweep(sweep_data):
        parameter_names = sweep_data["pname"]
        parameter_values = np.asarray(sweep_data["plistarr"], dtype=object)
        sweep_type = sweep_data["sweeptype"]

        if sweep_type == "sparse":
            return [
                dict(zip(parameter_names, values)) for values in zip(*parameter_values)
            ]

        value_lists = [np.ravel(row).tolist() for row in parameter_values]
        return [
            dict(zip(parameter_names, combination))
            for combination in itertools.product(*value_lists)
        ]

    geometry_records = _expand_sweep(geometry_sweep_data)
    material_records = _expand_sweep(material_sweep_data)  # a material sweep can be treated as a parameter sweep
    excitation_records = _expand_sweep(excitation_sweep_data)

    frequency_values = np.ravel(frequency_sweep_data["frequencies"])
    records = [
        {**geometry_record, **material_record, **excitation_record, "freq": frequency_value}
        for geometry_record in geometry_records
        for material_record in material_records
        for excitation_record in excitation_records
        for frequency_value in frequency_values
    ]

    return pd.DataFrame(records)
