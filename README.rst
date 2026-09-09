MPhSweepKit
===============================================

MPhSweepTools is a Python-based wrapper around the COMSOL API provided by the `MPh project <https://github.com/MPh-py/MPh>`_, tailored for large-scale parameter sweeps of magnetic components in power electronics.

Installation
-------------------


Clone the repository and install the package locally using pip:

::

    pip install -e .


Currently a workaround in the MPh package is required. Replace in the "node.py":

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

