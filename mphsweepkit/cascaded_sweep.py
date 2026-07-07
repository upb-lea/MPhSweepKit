"""Provides the class to perform a cascaded sweep on a COMSOL model accessed via the MPh API."""
from __future__ import annotations

from typing import Optional
from xml.parsers.expat import model
import pandas as pd
import mph

from mphsweepkit.sweep_expansion import get_cascaded_dataset
from mphsweepkit.directories import set_batch_directory


class CascadedSweep:
    """A class to perform a cascaded sweep on a COMSOL model."""

    def __init__(self, model: mph.Model, study_name: str):
        self.model = model
        self.study_name = study_name

        # Resolve study node
        self.study_node = self.model / "studies" / self.study_name

        # Collect study children
        self.sweep_nodes = self.study_node.children()
        self.sweep_names = [node.name() for node in self.sweep_nodes]
        self.sweep_types = [node.type() for node in self.sweep_nodes]

        # Data containers (filled by load_data_from_model)
        self.data: Optional[pd.DataFrame] = None
        self.data_unit_map: dict[str, str | None] = {}
        self.data_sweep_map: dict[str, str | None] = {}

    def load_data_from_model(self):
        """Load sweep data from the COMSOL model and build cascaded dataset."""
        list_of_sweep_data = [node.properties() for node in self.sweep_nodes]

        self.data, self.data_unit_map, self.data_sweep_map = get_cascaded_dataset(
            list_of_sweep_data=list_of_sweep_data,
            list_of_sweep_types=self.sweep_types,
            list_of_sweep_names=self.sweep_names,
        )


    def simulate(self, batch_dir: str = "batch_data"):
        """Run the cascaded sweep simulation."""

        if self.sweep_types[0] == "BatchSweep":
            set_batch_directory(self.sweep_nodes[0], directory_name=batch_dir)

        try:
            self.model.build()   
        except Exception:
            pass
        self.model.solve(self.study_name)

