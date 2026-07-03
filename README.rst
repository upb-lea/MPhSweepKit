MPhSweepKit
===============================================

MPhSweepTools is a Python-based wrapper around the COMSOL API provided by the MPh project, tailored for large-scale parameter sweeps of magnetic components in power electronics.


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
