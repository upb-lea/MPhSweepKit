"""Provides the class to perform a cascaded sweep on a COMSOL model accessed via the MPh API."""
from __future__ import annotations

from typing import Optional
from xml.parsers.expat import model
import pandas as pd
import mph

from mphsweepkit.cascaded_sweep import get_cascaded_dataset
from mphsweepkit.directories import set_batch_directory
from mphsweepkit.sweep import set_material_sweep


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
        self.init_message()

        # Data containers (filled by load_data_from_model)
        self.data: Optional[pd.DataFrame] = None
        self.data_unit_map: dict[str, str | None] = {}
        self.data_sweep_map: dict[str, str | None] = {}

        self.update_data_from_model()

    def init_message(self):
        """Prints a message about the initialized cascaded sweep model."""
        print(f"Initialized CascadedSweepModel")
        print(f"Study name: {self.study_name}")
        print(f"Sweep Structure:")
        spaces = "  "
        for name, sweep_type in zip(self.sweep_names, self.sweep_types):
            spaces += "  "
            print(f"{spaces}- {name} ({sweep_type})")

    def update_data_from_model(self):
        """Load sweep data from the COMSOL model and build cascaded dataset."""
        list_of_sweep_data = [node.properties() for node in self.sweep_nodes]

        self.data, self.data_unit_map, self.data_sweep_map = get_cascaded_dataset(
            list_of_sweep_data=list_of_sweep_data,
            list_of_sweep_types=self.sweep_types,
            list_of_sweep_names=self.sweep_names,
        )

        print(f"--------------------------------")
        print(f"Data updated from MPh-model.")
        print(f"Data Shape: {self.data.shape}")
        
    def set_material_sweep(self, 
                           sweep_name: str, 
                           material_names: list[str],
                           material_values: list[list[float]],
                           sweep_type: str = "filled"):
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
