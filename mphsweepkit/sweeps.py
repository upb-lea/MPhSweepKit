from typing import Any
import numpy as np
from numpy.typing import NDArray
import mph


# -------------------------
# Getter functions
# -------------------------
def get_parametric_sweep_data(sweep_properties: dict[str, Any]) -> dict[str, Any]:
    """
    Extract relevant data from a COMSOL parametric sweep node.

    Expected COMSOL keys:
    - pname
    - punit
    - plistarr
    - sweeptype
    """
    return {
        "pname": sweep_properties["pname"],
        "punit": sweep_properties.get("punit", None),
        "plistarr": sweep_properties["plistarr"],
        "sweeptype": sweep_properties["sweeptype"],
    }


def get_material_sweep_data(sweep_properties: dict[str, Any]) -> dict[str, Any]:
    """
    Extract relevant data from a COMSOL material sweep node.

    Expected COMSOL keys:
    - pname
    - plistarr
    - sweeptype
    Optional:
    - punit
    """
    return {
        "pname": sweep_properties["pname"],
        "punit": sweep_properties.get("punit", None),
        "plistarr": sweep_properties["plistarr"],
        "sweeptype": sweep_properties["sweeptype"],
    }


def get_frequency_sweep_data(sweep_properties: dict[str, Any]) -> dict[str, Any]:
    """
    Extract relevant data from a COMSOL frequency sweep node.

    Expected COMSOL keys:
    - plist
    Optional:
    - punit
    """
    return {
        "plist": sweep_properties["plist"],
        "punit": sweep_properties.get("punit", ""),
    }


# -------------------------
# Setter functions
# -------------------------
def set_parametric_sweep(
    sweep_node: mph.Node,
    param_names: list[str],
    param_units: list[str],
    param_values: NDArray[np.float64],
    sweep_type: str = "sparse",
) -> None:
    """
    Set a COMSOL parametric sweep.

    :param sweep_node: MPh node for the parametric sweep feature.
    :param param_names: Parameter names (same order as rows in param_values).
    :param param_units: Units aligned with param_names.
    :param param_values: 2D values array aligned with param_names.
    :param sweep_type: COMSOL sweep type, usually 'sparse' or 'filled'.
    """
    if len(param_names) != len(param_units):
        raise ValueError("param_names and param_units must have the same length.")

    sweep_node.property("pname", param_names)
    sweep_node.property("punit", param_units)
    sweep_node.property("plistarr", np.asarray(param_values, dtype=float))
    sweep_node.property("sweeptype", sweep_type)


def set_material_sweep(
    sweep_node: mph.Node,
    material_names: list[str],
    material_values: NDArray[np.float64],
    sweep_type: str = "sparse",
    material_units: list[str] | None = None,
) -> None:
    """
    Set a COMSOL material sweep.

    :param sweep_node: MPh node for the material sweep feature.
    :param material_names: Material parameter names.
    :param material_values: 2D values array aligned with material_names.
    :param sweep_type: COMSOL sweep type, usually 'sparse' or 'filled'.
    :param material_units: Optional units aligned with material_names.
    """
    if material_units is not None and len(material_units) != len(material_names):
        raise ValueError("material_units must have the same length as material_names.")

    sweep_node.property("pname", material_names)
    if material_units is not None:
        sweep_node.property("punit", material_units)
    sweep_node.property("plistarr", np.asarray(material_values, dtype=float))
    sweep_node.property("sweeptype", sweep_type)


def set_frequency_sweep(
    sweep_node: mph.Node,
    frequency_values: NDArray[np.float64],
    frequency_unit: str = "Hz",
) -> None:
    """
    Set a COMSOL frequency sweep.

    :param sweep_node: MPh node for the frequency sweep feature.
    :param frequency_values: 1D/2D frequency values.
    :param frequency_unit: Frequency unit, e.g. 'Hz', 'kHz', 'MHz'.
    """
    sweep_node.property("plist", np.asarray(frequency_values, dtype=float))
    sweep_node.property("punit", frequency_unit)