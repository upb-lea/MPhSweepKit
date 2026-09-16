MPhSweepKit
===============================================

MPhSweepTools is a Python-based wrapper around the COMSOL API provided by the `MPh project <https://github.com/MPh-py/MPh>`_, tailored for large-scale parameter sweeps of magnetic components in power electronics.

Installation
-------------------


Clone the repository and install the package locally using pip:

::

    pip install -e .



Core API
-------------------

- ``CascadedSweepModel(model, study_name)``
- ``set_material_sweep(...)``
- ``set_parametric_sweep(...)``
- ``simulate()``
- ``post_process_data(post_processing_exprs)``
- ``save_global_data()``  -> writes ``.../input_data.csv`` and ``.../output_data.csv``
- ``create_dataset_selection(...)``
- ``export_dataset_with_expressions(...)`` -> exports field data directly calculated on a (sub-)dataset selection, e.g. a geometry or a surface
- ``get_comsol_looplevels()`` -> walks through the COMSOL model tree of a cascaded sweep

Typical workflow 
-------------------

See ``examples/studies/infinite_ferrite_cross_sections`` scripts 1-5.

