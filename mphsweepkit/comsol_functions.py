import itertools
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
import mph


def get_parameters(model: mph.Model) -> dict:
    """
    Get all global parameters and their values.

    :param model: The COMSOL model object.
    :return: A dictionary containing parameter names and their corresponding values."""
    parameters = {}
    for name in model.parameters().keys():
        value = model.parameters().get(name)
        parameters[name] = value
    return parameters


def print_parameters(parameters: dict):
    """
    Print the parameters in a readable format.

    :param parameters: A dictionary containing parameter names and their corresponding values."""
    for name, value in parameters.items():
        print(f"{name}: {value}")


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


def get_descriptions(model: mph.Model, parameter_names: list) -> dict:
    """
    Get the COMSOL descriptions for a list of parameter names.

    :param model: COMSOL model used to resolve the parameter descriptions.
    :param parameter_names: List of parameter names to describe.
    :return: A dictionary mapping parameter names to their descriptions.
    """
    return {name: model.description(name) for name in parameter_names}


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


def get_cascaded_dataset(
    geometry_sweep_data: dict, excitation_sweep_data: dict, frequency_sweep_data: dict
) -> pd.DataFrame:
    """
    Build the full cascaded dataset for geometry -> excitation -> frequency.

    Sparse sweeps are expanded case-by-case, filled sweeps are expanded as a
    cartesian product, and the resulting records are combined in cascade order.

    :param geometry_sweep_data: Geometry sweep data returned by get_sweep_data.
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
    excitation_records = _expand_sweep(excitation_sweep_data)

    frequency_values = np.ravel(frequency_sweep_data["frequencies"])
    records = [
        {**geometry_record, **excitation_record, "freq": frequency_value}
        for geometry_record in geometry_records
        for excitation_record in excitation_records
        for frequency_value in frequency_values
    ]

    return pd.DataFrame(records)


def read_comsol_txt_to_df(filepath: str) -> pd.DataFrame:
    """
    Read a COMSOL-style .txt export into a pandas DataFrame.

    :param filepath: Path to the COMSOL .txt file.
    :return: DataFrame containing the data from the file.
    """
    header_cols = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("%"):
                content = line.lstrip("%").strip()
                if content.startswith("x"):
                    # split on 2+ spaces so multi-word final column name stays intact
                    header_cols = (
                        pd.Series(content).str.split(r"\s{2,}", regex=True).iloc[0]
                    )
                    break

    if header_cols is None:
        raise ValueError("Could not find header line starting with '% x'.")

    df = pd.read_csv(
        filepath,
        sep=r"\s+",  # <- modern replacement for delim_whitespace=True
        comment="%",
        header=None,
        names=header_cols,
        engine="python",  # robust with regex separators
    )

    return df


def plot_comsol_df(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    value_col: str = "Magnetic flux density norm",
    kind: str = "scatter",  # "scatter" or "tricontourf"
    cmap: str = "viridis",
    point_size: int = 8,
    levels: int = 30,
    figsize: tuple = (7, 5),
):
    """
    Plot COMSOL point data from a DataFrame.

    :param df: DataFrame containing x, y, and value columns.
    :param x_col: Column name for x-axis values.
    :param y_col: Column name for y-axis values.
    :param value_col: Column name for color values.
    :param kind: Type of plot ('scatter' or 'tricontourf').
    :param cmap: Colormap for the plot.
    :param point_size: Size of points in scatter plot.
    :param levels: Number of contour levels.
    :param figsize: Figure size for the plot.
    :param kind: Type of plot ('scatter' or 'tricontourf').
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    z = df[value_col].to_numpy()

    if kind == "scatter":
        mappable = ax.scatter(x, y, c=z, s=point_size, cmap=cmap)
    elif kind == "tricontourf":
        mappable = ax.tricontourf(x, y, z, levels=levels, cmap=cmap)
    else:
        raise ValueError("kind must be 'scatter' or 'tricontourf'")

    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(value_col)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{value_col} over ({x_col}, {y_col})")
    ax.set_aspect("equal")  # useful for geometric fields
    plt.tight_layout()
    plt.show()


post_processing_exprs = {
    "p_loss": {
        "expression": "intopFlux(mf.Qh)/A_c",
        "unit": "W /m^3",
        "label": r"$p_\mathrm{loss}$",
    },
    "p_mag": {
        "expression": "intopFlux(mf.Qml)/A_c",
        "unit": "W /m^3",
        "label": r"$p_\mathrm{mag}$",
    },
    "p_el": {
        "expression": "intopFlux(mf.Qrh)/A_c",
        "unit": "W /m^3",
        "label": r"$p_\mathrm{el}$",
    },
}
