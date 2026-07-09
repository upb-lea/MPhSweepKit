"""Provides the class to perform a cascaded sweep on a COMSOL model accessed via the MPh API."""
from __future__ import annotations

from typing import Optional, TypedDict, NotRequired
from pathlib import Path
import json
import pandas as pd
import mph

from mphsweepkit.cascaded_sweep import get_cascaded_dataset
from mphsweepkit.directories import set_batch_directory
from mphsweepkit.sweep import set_material_sweep


class PostProcessSpec(TypedDict):
    expression: str
    unit: NotRequired[str]
    label: NotRequired[str]


class CascadedSweepModel:
    """A class to work on a COMSOL model with cascaded sweeps."""

    def __init__(self, model: mph.Model, study_name: str):
        self.model = model
        self.study_name = study_name

        # Resolve study node
        self.study_node = self.model / "studies" / self.study_name

        # Collect study children
        self.sweep_nodes = self.study_node.children()
        self.sweep_names = [node.name() for node in self.sweep_nodes]
        self.sweep_types = [node.type() for node in self.sweep_nodes]
        self.print_initialization_summary()

        # Model Datasets
        self.model_datasets = self.model.datasets()

        # Input containers and metadata maps
        self.input_data: Optional[pd.DataFrame] = None
        self.input_unit_map: dict[str, str | None] = {}
        self.input_sweep_map: dict[str, str | None] = {}

        # Output containers and metadata maps
        self.output_data: pd.DataFrame = pd.DataFrame()
        self.output_unit_map: dict[str, str | None] = {}
        self.output_label_map: dict[str, str | None] = {}

        self.update_data_from_model()

    @property
    def combined_data(self) -> pd.DataFrame:
        """Combined input/output view for convenience."""
        if self.input_data is None:
            return pd.DataFrame()
        return self.input_data.join(self.output_data, how="left")


    def print_initialization_summary(self):
        """Prints a message about the initialized cascaded sweep model."""
        print("Initialized CascadedSweepModel")
        print(f"Study name: {self.study_name}")
        print("Sweep Structure:")
        spaces = "  "
        for name, sweep_type in zip(self.sweep_names, self.sweep_types):
            spaces += "  "
            print(f"{spaces}- {name} ({sweep_type})")

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
            self.output_unit_map = {}
            return
        self.output_data = pd.DataFrame(index=self.input_data.index)
        self.output_unit_map = {}

    def update_data_from_model(self):
        """Load sweep data from the COMSOL model and build cascaded input dataset."""
        list_of_sweep_data = [node.properties() for node in self.sweep_nodes]

        input_data, unit_map, sweep_map = get_cascaded_dataset(
            list_of_sweep_data=list_of_sweep_data,
            list_of_sweep_types=self.sweep_types,
            list_of_sweep_names=self.sweep_names,
        )

        self.input_data = input_data
        self.input_unit_map = unit_map
        self.input_sweep_map = sweep_map

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
        material_values: list[list[float]],
        sweep_type: str = "filled"
    ):
        """Set the data for a specific sweep."""
        if sweep_name not in self.sweep_names:
            raise ValueError(f"Sweep name '{sweep_name}' not found in the model.")
        sweep_index = self.sweep_names.index(sweep_name)

        set_material_sweep(
            sweep_node=self.sweep_nodes[sweep_index],
            material_names=material_names,
            material_values=material_values,
            sweep_type=sweep_type
        )

        self.update_data_from_model()

    def simulate(self, batch_dir: str = "batch_data"):
        """Run the cascaded sweep simulation."""
        if self.sweep_types[0] == "BatchSweep":
            set_batch_directory(self.sweep_nodes[0], directory_name=batch_dir)

        try:
            self.model.build()   
        except Exception:
            pass
        self.model.solve(self.study_name)
    
    def post_process(self, post_processing_exprs: dict[str, dict[str, str]]):
        """Perform post-processing and write results to output_data only."""
        if self.input_data is None:
            raise ValueError("Input data is not loaded. Please run 'update_data_from_model' first.")

        for column_name, spec in post_processing_exprs.items():
            values = self.model.evaluate(
                expression=spec["expression"],
                unit=spec["unit"],
                dataset=self.model_datasets[-1],
            )
            self.output_data[column_name] = values

            # keep metadata maps in sync with your expression dict
            self.output_unit_map[column_name] = spec.get("unit")
            self.output_label_map[column_name] = spec.get("label")

    def clear_output_data(self):
        """Drop all computed outputs but keep input_data."""
        self._reset_outputs_to_inputs_index()

    def save_result_data(self, folder: str = "result_data"):
        """Save input/output dataframes and metadata maps to disk.

        Folder structure:
            <folder>/
                input_data.csv
                output_data.csv
                input_meta.json
                output_meta.json
        """
        target_dir = Path(folder)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save tabular data
        input_df = self.input_data if self.input_data is not None else pd.DataFrame()
        input_df.to_csv(target_dir / "input_data.csv", index=False)
        self.output_data.to_csv(target_dir / "output_data.csv", index=False)

        # Save metadata maps
        input_meta = {
            "input_unit_map": self.input_unit_map,
            "input_sweep_map": self.input_sweep_map,
        }
        output_meta = {
            "output_unit_map": self.output_unit_map,
            "output_label_map": self.output_label_map,
        }

        with open(target_dir / "input_meta.json", "w", encoding="utf-8") as f:
            json.dump(input_meta, f, indent=2)

        with open(target_dir / "output_meta.json", "w", encoding="utf-8") as f:
            json.dump(output_meta, f, indent=2)
