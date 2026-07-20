MPhSweepKit
===============================================

MPhSweepTools is a Python-based wrapper around the COMSOL API provided by the `MPh project <https://github.com/MPh-py/MPh>`_, tailored for large-scale parameter sweeps of magnetic components in power electronics.

Installation
-------------------

Clone the repository and install the package locally using pip:

::

    pip install -e .

Currently a workaround in the MPh package is required.
Replace in the "node.py"

::

    ...
    elif datatype == 'DoubleRowMatrix':
        value = java.getDoubleMatrix(name)
        if len(value) == 0:
            rows = []
        elif len(value) == 1:
            rows = [array(value[0])]
        elif len(value) == 2:
            rows = [array(value[0]), array(value[1])]
        else:
            error = 'Cannot convert double-row matrix with more than two rows.'
            log.error(error)
            raise TypeError(error)
        return array(rows, dtype=object)
    ...
  
by

::

    ...
    elif datatype == 'DoubleRowMatrix':
        value = java.getDoubleMatrix(name)
        if len(value) == 0:
            rows = []
        else:
            rows = [array(row) for row in value]
        return array(rows, dtype=object)
    ...

.


Documentation
-------------------

MPhSweepKit provides utilities to configure, run, and evaluate cascaded parameter
sweeps on COMSOL models through the MPh API.

Core modules
^^^^^^^^^^^^^^^^^^^

- ``mphsweepkit.model``
  Contains ``CascadedSweepModel`` as the main high-level interface for sweep workflows.

  Main responsibilities:

  - Read sweep definitions from a COMSOL study and build a cascaded input table.
  - Update material sweep definitions via ``set_material_sweep(...)``.
    - Update parametric sweeps via ``set_parametric_sweep(...)``.
  - Run simulations via ``simulate(...)`` (including batch directory handling).
    - Evaluate scalar post-processing expressions via ``post_process_data(...)``.
    - Export field data via ``create_dataset_selection(...)`` and
        ``export_dataset_with_expressions(...)``.
    - Persist scalar input/output data via ``save_global_data()``.

    Result export structure created by ``save_global_data()``:

  - ``input_data.csv``
  - ``output_data.csv``
    (stored in ``global_data/`` by default)

- ``mphsweepkit.plot_data``
  Contains ``DataPlot`` for loading and exploring exported scalar result data.

  Main responsibilities:

  - Load result files using ``DataPlot.from_result_folder(...)``.
    - Expose ``input_df``, ``output_df``, and ``combined_df``.
  - Provide metadata helpers for labels and units:

     - ``get_label(column)``
     - ``get_unit(column)``
     - ``format_axis_label(column)``

  - Provide column discovery and validation helpers:

     - ``input_columns()``
     - ``output_columns()``
     - ``available_columns()``
     - ``assert_columns_exist(columns)``

    - Provide grouped plotting helper:

         - ``y_over_x_with_color_and_style(...)`` with ``PlotSettings`` and
             ``PlotTextOverrides``

Supporting modules
^^^^^^^^^^^^^^^^^^^

- ``mphsweepkit.sweep_cascade``: creation of cascaded sweep datasets.
- ``mphsweepkit.sweep_helpers``: sweep node configuration helpers.
- ``mphsweepkit.directories``: batch/result directory handling.
- ``mphsweepkit.postprocessing``: post-processing support functions.
- ``mphsweepkit.plot_fields``: field-plot related helpers.

Typical workflow
^^^^^^^^^^^^^^^^^^^

Scalar-data workflow (scripts 1-3 in
``examples/studies/infinite_ferrite_cross_sections``)

1. Build a model wrapper:

    ``csm = CascadedSweepModel(model, study_name="...")``

2. Configure sweeps (optional):

    ``csm.set_material_sweep(...)``
    ``csm.set_parametric_sweep(...)``

3. Run simulation:

    ``csm.simulate(batch_dir="batch_data")``

4. Evaluate scalar post-processing expressions:

    ``post_processing_exprs = load_post_processing_exprs("global_post_processing_expressions.json")``
    ``csm.post_process_data(post_processing_exprs)``

5. Save scalar sweep data:

    ``csm.save_global_data()``

6. Load exported results for plotting/analysis:

    ``dp = DataPlot.from_result_folder("global_data")``

Field-data workflow (scripts 4-5 in
``examples/studies/infinite_ferrite_cross_sections``)

1. Create a selection dataset from solved data:

    ``csm.create_dataset_selection(selection_name="Core", selection_type="dom", selection_dataset_name="Solution Core")``

2. Load field post-processing expressions:

    ``field_exprs = load_post_processing_exprs("fields_post_processing_expressions.json")``

3. Export one field file per geometry configuration:

    ``for looplevel in csm.get_comsol_looplevels():``
    ``    csm.export_dataset_with_expressions(..., looplevel=looplevel, looplevelinput=["manual"] * len(looplevel))``

4. Read exported field data and plot:

    ``df = read_comsol_dataset("field_data/geometry_1.txt")``
    ``PlotField(df).plot_field(...)``
