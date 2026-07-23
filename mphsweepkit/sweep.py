"""Provides the class to perform a cascaded sweep on a COMSOL model accessed via the MPh API."""

from typing import Optional, TypedDict, NotRequired, Any, Literal
from pathlib import Path
import pandas as pd
import numpy as np
import mph

# Import of meta data
from .meta_names import METADATA_ROW_NAMES, ALLOWED_SWEEP_TYPES

# Imports from the sweep_... modules
from mphsweepkit.sweep_cascade import get_cascaded_dataset
from mphsweepkit.sweep_get_set import set_material_sweep, set_parametric_sweep
from mphsweepkit.process_helpers import load_post_processing_exprs


class PostProcessSpec(TypedDict):
    expression: str
    unit: NotRequired[str]


class CascadedSweepModel:
    """A class to work on a COMSOL model with cascaded sweeps."""

    def __init__(self, model: Any, study_name: str, show_param_names: bool = False):
        self.model = model
        self.study_name = study_name
        self.show_param_names = show_param_names

        # Resolve study node
        self.node_studies = self.model / "studies" / self.study_name
        self.node_datasets = self.model / "datasets"
        self.node_exports = self.model / "exports"

        # Collect cascaded sweep structure
        self.sweep_loop_nodes = self.node_studies.children()
        self.sweep_loop_levels = [node.name() for node in self.sweep_loop_nodes]
        self.sweep_loop_types = [node.type() for node in self.sweep_loop_nodes]
        self.sweep_loop_lengths = [
            self._get_loop_length(node) for node in self.sweep_loop_nodes
        ]
        self.print_initialization_summary(show_param_names=self.show_param_names)

        # Set...
        # ... default dataset name for the solution.
        self.solution_dataset_name = "Cascaded Sweep Solution"
        # ... directories for global data (input/output tables)
        self.dir_global_data = Path("global_data")
        self.dir_global_data.mkdir(parents=True, exist_ok=True)
        # ... directories for field data
        self.dir_field_data = Path("field_data")
        self.dir_field_data.mkdir(parents=True, exist_ok=True)
        # ... directories for batch directory
        self.dir_batch_data = Path("batch_data")
        self.dir_batch_data.mkdir(parents=True, exist_ok=True)
        # ... directories for meta data
        self.dir_meta_data = Path("meta_data")
        self.dir_meta_data.mkdir(parents=True, exist_ok=True)

        # From the COMSOL-model, update global ...
        # ... input data (sweep parameters)
        # ... output data (post-processing results -> initially empty)
        self.input_data: Optional[pd.DataFrame] = None
        self.output_data: pd.DataFrame = pd.DataFrame()
        self.meta_data: pd.DataFrame = pd.DataFrame()
        self._update_data_from_model()

    @property
    def combined_data(self) -> pd.DataFrame:
        """Combined input/output view for convenience."""
        if self.input_data is None:
            return pd.DataFrame()
        return self.input_data.join(self.output_data, how="left")

    @staticmethod
    def _get_loop_length(node):
        if node.type() == "Frequency":
            return len(node.property("plist"))
        elif node.type() == "CoilCurrentCalculation":
            # The CoilCurrentCalculation sweep type does not add any additional loops.
            return 1
        else:
            arr = node.property("plistarr")
            if node.property("sweeptype") == "sparse":
                return arr.shape[1]
            else:
                looplength = 1
                for row in arr:
                    looplength *= len(row)
                return looplength

    @staticmethod
    def _build_multiindex_columns(
        names: list[str],
        unit_map: dict[str, str | None] | None = None,
        group_map: dict[str, str | None] | None = None,
    ) -> pd.MultiIndex:
        """Create metadata-rich 3-level MultiIndex columns."""
        unit_map = unit_map or {}
        group_map = group_map or {}
        tuples = [
            (
                name,
                unit_map.get(name),
                group_map.get(name),
            )
            for name in names
        ]
        return pd.MultiIndex.from_tuples(tuples, names=list(METADATA_ROW_NAMES))

    def print_initialization_summary(self, show_param_names: bool = False):
        """Print a message about the initialized cascaded sweep model.

        :param show_param_names: If True, append sweep parameter names from the
            COMSOL node property ``pname`` (when available).
        """
        print("Initialized CascadedSweepModel")
        print(f"Study name: {self.study_name}")
        print("Sweep Structure:")
        spaces = "  "
        for node, name, sweep_type in zip(
            self.sweep_loop_nodes, self.sweep_loop_levels, self.sweep_loop_types
        ):
            spaces += "  "
            line = f"{spaces}- {name} ({sweep_type})"
            if show_param_names:
                props = node.properties()
                pname = props.get("pname")
                if pname:
                    if isinstance(pname, str):
                        pname_list = [pname]
                    else:
                        pname_list = [str(item) for item in pname]
                    shown = ", ".join(repr(item) for item in pname_list)
                    line += f" -> {shown}"
            print(line)

    def _reset_outputs_to_inputs_index(self):
        """Reset output_data so it has exactly the same index as input_data.

        Reason:
            Post-processing results are assigned row-wise to the sweep parameter rows.
            Whenever input_data changes (e.g., new sweep settings, different number/order
            of cases), old output_data may no longer align and could silently attach
            results to the wrong parameter set. Reinitializing output_data with the
            current input_data index guarantees safe, deterministic alignment.
        """
        if self.input_data is None:
            self.output_data = pd.DataFrame()
            return
        self.output_data = pd.DataFrame(
            index=self.input_data.index,
            columns=pd.MultiIndex.from_tuples([], names=list(METADATA_ROW_NAMES)),
        )

    def _update_data_from_model(self):
        """
        Load sweep data from the COMSOL model and (re-)build the cascaded input dataset as pd.DataFrame.

        This method is called automatically when a CascadedSweepModel is initialized or when sweep data is updated.
        """
        list_of_sweep_properties = [node.properties() for node in self.sweep_loop_nodes]

        input_data, unit_map, sweep_map = get_cascaded_dataset(
            allowed_sweep_types=ALLOWED_SWEEP_TYPES,
            list_of_sweep_data=list_of_sweep_properties,
            list_of_sweep_types=self.sweep_loop_types,
            list_of_sweep_names=self.sweep_loop_levels,
        )

        # Create a DataFrame with the cascaded sweep data and metadata in the columns.
        self.input_data = input_data
        self.input_data.columns = self._build_multiindex_columns(
            names=[str(col) for col in self.input_data.columns],
            unit_map=unit_map,
            group_map=sweep_map,
        )

        # Print loop lengths
        self.sweep_loop_lengths = [
            self._get_loop_length(node) for node in self.sweep_loop_nodes
        ]
        print(f"Loop names: {self.sweep_loop_levels}")
        print(f"Loop lengths: {self.sweep_loop_lengths}")

        # TODO: write sweep info to self.meta_data DataFrame

        # Input table changed -> reset outputs to aligned empty frame
        self._reset_outputs_to_inputs_index()

        print("--------------------------------")
        print("Data updated from MPh-model.")
        print(f"Input data shape: {self.input_data.shape}")
        print(f"Reset output data to shape of the input data: {self.output_data.shape}")
        print(f"Combined shape: {self.combined_data.shape}")

    def set_material_sweep(
        self,
        sweep_name: str,
        material_names: list[str],
        material_values: list[list[float]] | np.ndarray,
        sweep_type: str = "filled",
    ):
        """Set the data for a specific material sweep."""
        if sweep_name not in self.sweep_loop_levels:
            raise ValueError(f"Sweep name '{sweep_name}' not found in the model.")
        sweep_index = self.sweep_loop_levels.index(sweep_name)

        set_material_sweep(
            sweep_node=self.sweep_loop_nodes[sweep_index],
            material_names=material_names,
            material_values=np.asarray(material_values, dtype=np.float64),
            sweep_type=sweep_type,
        )

        self._update_data_from_model()

    def set_parametric_sweep(
        self,
        sweep_name: str,
        param_names: list[str],
        param_units: list[str],
        param_values: list[list[float] | np.ndarray],
        sweep_type: str = "sparse",
    ):
        """Set the data for a specific parametric sweep."""
        if sweep_name not in self.sweep_loop_levels:
            raise ValueError(f"Sweep name '{sweep_name}' not found in the model.")
        sweep_index = self.sweep_loop_levels.index(sweep_name)

        set_parametric_sweep(
            sweep_node=self.sweep_loop_nodes[sweep_index],
            param_names=param_names,
            param_units=param_units,
            param_values=np.asarray(param_values, dtype=np.float64),
            sweep_type=sweep_type,
        )

        self._update_data_from_model()

    # Simulation
    def simulate(self):
        """Run the cascaded sweep simulation."""

        if self.sweep_loop_types[0] == "BatchSweep":
            absolute_path_to_batch_dir = Path(self.dir_batch_data).absolute()
            self.sweep_loop_nodes[0].property(
                "batchdir", str(absolute_path_to_batch_dir)
            )
            print(
                f"Batch directory for the geometric sweep set to: {absolute_path_to_batch_dir}"
            )

        try:
            self.model.build()
        except Exception:
            pass
        self.model.solve(self.study_name)

        # Give the computed dataset a defined name
        # TODO: instead of renaming the last dataset, catch the actual dataset name
        #       from the model and rename it to the desired name
        # TODO: Check if the dataset already exists and handle it (e.g., rename or overwrite)
        self._rename_last_dataset(self.solution_dataset_name)

    def _rename_last_dataset(self, new_name: str):
        """Rename the last dataset in the model."""
        last_dataset = self.model / "datasets" / self.model.datasets()[-1]
        last_dataset.rename(new_name)

    # Post processing of global data
    def post_process_globals(self, post_processing_exprs: dict[str, dict[str, str]]):
        """Perform post-processing and write results to output_data only."""
        if self.input_data is None:
            raise ValueError(
                "No input data available. Please set up the sweeps and run the simulation first."
            )

        for column_name, spec in post_processing_exprs.items():
            values = self.model.evaluate(
                expression=spec["expression"],
                unit=spec["unit"],
                dataset=self.solution_dataset_name,
            )
            column_key = (column_name, spec.get("unit"), "post-processing")
            self.output_data.loc[:, column_key] = values

    def save_global_data(self):
        """Save input/output dataframes to disk.

        Folder structure:
            <folder>/
                input_data.csv
                output_data.csv
        """
        target_dir = Path(self.dir_global_data)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save tabular data
        input_df = self.input_data if self.input_data is not None else pd.DataFrame()
        input_df.to_csv(target_dir / "input_data.csv", index=True)
        self.output_data.to_csv(target_dir / "output_data.csv", index=True)

        print(f"Saved result data to: {target_dir.resolve()}")

    # Working with selections and datasets
    def print_available_selections(self):
        """
        Print the available selections in the model.
        """
        print("Available selections:")
        for tag in self.model.java.selection().tags():
            selection = self.model.java.selection(tag)
            print("  ", tag, selection.label())

    def _get_selection_tag(
        self, selection_name, selection_type: Literal["dom", "bnd", "pt"] = "dom"
    ) -> str:
        """
        Get the tag of the selection with the given name.
        """
        return next(
            str(tag)
            for tag in self.model.java.selection().tags()
            if self.model.java.selection(str(tag)).label() == selection_name
            and str(tag).endswith("_" + selection_type)
        )

    def _duplicate_solution_dataset(self, duplicate_name):
        """
        Duplicate a dataset and rename it.
        """
        # Get the node of the solution dataset
        solution_dataset = self.model / "datasets" / self.solution_dataset_name

        # Get the node of the newly created dataset
        new_dataset = self.node_datasets.create(
            solution_dataset.type(), name=duplicate_name
        )

        # Get the solution from the original dataset and set it to the new dataset
        new_dataset.property("solution", value=solution_dataset.property("solution"))

        return new_dataset

    def create_dataset_selection(
        self,
        selection_name: str,
        selection_type: Literal["dom", "bnd", "pt"] = "dom",
        selection_dataset_name: str | None = None,
    ):
        """
        Create a dataset from a selection.
        """
        # Handle default dataset name if not provided
        if selection_dataset_name is None:
            selection_dataset_name = f"{selection_name}_{selection_type}"

        # Check if the selection dataset already exists
        if selection_dataset_name in self.model.datasets():
            print(f"Dataset '{selection_dataset_name}' already exists.")
        else:
            # Get the selection tag from name and type
            selection_tag = self._get_selection_tag(selection_name, selection_type)

            # Duplicate the solution dataset to create a new dataset for the selection
            new_dataset = self._duplicate_solution_dataset(selection_dataset_name)

            # Set the selection for the dataset to the selected tag through the Java API
            new_dataset.java.selection().named(selection_tag)

            print(
                f"Creating dataset '{selection_dataset_name}' from selection '{selection_name}' of type '{selection_type}'."
            )

    # Loop-Level
    def _get_looplevel_array(self):
        """Create loop-level arrays."""
        return [list(range(1, length + 1)) for length in self.sweep_loop_lengths]

    @staticmethod
    def _pack_looplevels_per_geometry(
        geometry_levels: list[int],
        material_levels: list[int],
        excitation_levels: list[int],
        frequency_levels: list[int],
    ) -> list[list[list[int]]]:
        """
        Pack loop levels for each geometry configuration.

            returns:
                [
                    [frequency_levels, excitation_levels, material_levels_for_geometry_0],
                    [frequency_levels, excitation_levels, material_levels_for_geometry_1],
                    [frequency_levels, excitation_levels, material_levels_for_geometry_2],
                    ...
                ]

        :param geometry_levels: List of geometry levels.
        :param material_levels: List of material levels.
        :param excitation_levels: List of excitation levels.
        :param frequency_levels: List of frequency levels.
        :return: A list of loop levels for each geometry configuration.
        """

        material_count = len(material_levels)

        return [
            [
                list(frequency_levels),
                list(excitation_levels),
                [
                    geometry_index * material_count + material_index + 1
                    for material_index in range(material_count)
                ],
            ]
            for geometry_index, _ in enumerate(geometry_levels)
        ]

    def get_looplevels_per_geometry(self) -> list[list[list[int]]]:
        """
        Get the loop levels for each geometry configuration.

            returns:
                [
                    [frequency_levels, excitation_levels, material_levels_for_geometry_0],
                    [frequency_levels, excitation_levels, material_levels_for_geometry_1],
                    ...
                ]

        :return: A list of loop levels for each geometry configuration.
        """
        looplevel_array = self._get_looplevel_array()
        print(f"Loop levels from COMSOL: {looplevel_array}")

        if self.sweep_loop_types == [
            "BatchSweep",
            "MaterialSweep",
            "Parametric",
            "CoilCurrentCalculation",
            "Frequency",
        ]:
            # The 'CoilCurrentCalculation' sweep type does not add any additional loops,
            # so we can ignore it in the loop level packing.
            return self._pack_looplevels_per_geometry(
                looplevel_array[0],
                looplevel_array[1],
                looplevel_array[2],
                looplevel_array[4],
            )

        elif self.sweep_loop_types == [
            "BatchSweep",
            "MaterialSweep",
            "Parametric",
            "Frequency",
        ]:
            return self._pack_looplevels_per_geometry(
                looplevel_array[0],
                looplevel_array[1],
                looplevel_array[2],
                looplevel_array[3],
            )

        else:
            raise ValueError(
                "Sweep configuration is not valid. Expected ['BatchSweep', 'MaterialSweep', 'Parametric', 'CoilCurrentCalculation' (optional), 'Frequency']."
            )

    # Post processing of field data
    def export_fields_on_geometry(
        self,
        dataset_name: str,
        expressions: list[str],
        descriptions: list[str],
        units: list[str],
        looplevel: list[list[int]],
        looplevelinput: list[str],
        geometry_no: int,
        sub_folder: str | None = None,
    ):
        """
        Export a datasheet with expressions.

        It would also be possible to use a single export node to export all expressions at once. However, to maintain a clear and flexible structure of the resulting csv files, by convention every field quantity is exported via a single export node and into a single file. 

        :param dataset_name: Name of the dataset to export.
        :param expressions: List of expressions to export.
        :param descriptions: List of descriptions for the expressions.
        :param units: List of units for the expressions.
        :param looplevel: List of lists specifying the loop levels of the parameter cascode.
        :param looplevelinput: List of strings specifying the loop level input type for each expression.
        """
        # Check if the dataset exists
        if dataset_name not in self.model.datasets():
            raise ValueError(f"Dataset '{dataset_name}' does not exist.")

        if sub_folder is None:
            sub_folder = dataset_name

        if not Path(self.dir_field_data).joinpath(sub_folder).exists():
            Path(self.dir_field_data).joinpath(sub_folder).mkdir(
                parents=True, exist_ok=True
            )
            print(f"Created sub-folder for field data: {sub_folder}")

        # Get the dataset node
        selected_dataset_node = self.model / "datasets" / dataset_name
        print(f"\nDataset to be exported:           '{selected_dataset_node.name()}'")

        # Check and create the export nodes for each expression
        print("    Check status of export nodes:")
        for expression, description, unit in zip(expressions, descriptions, units):

            # Create an export node for the dataset
            export_name = f"{description} on {dataset_name}"
            if export_name in [node.name() for node in self.node_exports.children()]:
                print(f"        Overwrite export node:    '{export_name}'")
            else:
                print(f"  Create export node:             '{export_name}'")
                self.node_exports.create("Data", name=export_name)

        # Export the data for each expression
        for expression, description, unit in zip(expressions, descriptions, units):

            # Create an export node for the dataset
            export_name = f"{description} on {dataset_name}"

            # Get the export node
            export_node = self.node_exports / export_name

            # print(export_node.properties())  # uncomment for debugging

            # Set the export node properties
            export_node.property("exporttype", "text")
            export_node.property("ifexists", "overwrite")
            export_node.property("data", selected_dataset_node.tag())
            export_node.property("expr", value=[expression])
            export_node.property("looplevel", value=looplevel)
            export_node.property("looplevelinput", value=looplevelinput)
            export_node.property("descr", value=[description])
            export_node.property("unit", value=[unit])

            # Path to the output file for the export
            file_name = f"geometry_{geometry_no}_{description}.txt"
            path2file = str(
                self.dir_field_data
                .joinpath(sub_folder)
                .joinpath(file_name)
                .absolute()
            )

            self.model.export(export_node, file=path2file)
            print(f"    Exported field data to:       '{sub_folder}/{file_name}'")


    def export_fields_on_all_geometries(
        self,
        dataset_name: str,
        post_processing_exprs: dict[str, dict[str, str]],
        sub_folder: str | None = None,
    ):
        """Export field data with expressions."""
        looplevels_per_geometry = self.get_looplevels_per_geometry()

        # Convert the post-processing expressions into lists for the COMSOL export
        expressions = [entry["expression"] for entry in post_processing_exprs.values()]
        labels = [entry["label"] for entry in post_processing_exprs.values()]
        units = [entry["unit"] for entry in post_processing_exprs.values()]

        # Iterate over the loop levels and export the results for each geometry configuration
        for geometry_no, looplevel_arr in enumerate(looplevels_per_geometry):
            self.export_fields_on_geometry(
                dataset_name=dataset_name,
                expressions=expressions,
                descriptions=labels,
                units=units,
                looplevel=looplevel_arr,
                looplevelinput=["manual"] * len(looplevel_arr),
                geometry_no=geometry_no,
                sub_folder=sub_folder,
            )

