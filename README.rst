MPhSweepKit
===============================================

MPhSweepTools is a Python-based wrapper around the COMSOL API provided by the `MPh project <https://github.com/MPh-py/MPh>`_, tailored for large-scale parameter sweeps of magnetic components in power electronics.

Installation
-------------------

::

    pip install -e .

Core API
-------------------

- ``CascadedSweepModel(model, study_name, show_param_names=False)``
- ``set_material_sweep(...)``
- ``set_parametric_sweep(...)``
- ``simulate(batch_dir="batch_data")``
- ``post_process_data(post_processing_exprs)``
- ``save_global_data()``  -> writes ``global_data/input_data.csv`` and ``global_data/output_data.csv``
- ``create_dataset_selection(...)``
- ``export_dataset_with_expressions(...)``
- ``get_comsol_looplevels()``

Scalar workflow (scripts 1-3)
-------------------

From ``examples/studies/infinite_ferrite_cross_sections``:

1. Build and configure sweeps
2. ``simulate(...)``
3. ``post_process_data(load_post_processing_exprs("global_post_processing_expressions.json"))``
4. ``save_global_data()``
5. ``DataPlot.from_result_folder("global_data")``

Field workflow (scripts 4-5)
-------------------

From ``examples/studies/infinite_ferrite_cross_sections``:

1. ``create_dataset_selection(...)``
2. Load ``fields_post_processing_expressions.json`` via ``load_post_processing_exprs(...)``
3. Loop ``for looplevel in csm.get_comsol_looplevels():`` and call ``export_dataset_with_expressions(...)``
4. Read with ``read_comsol_dataset("field_data/geometry_1.txt")``
5. Plot with ``PlotField(df).plot_field(...)``
