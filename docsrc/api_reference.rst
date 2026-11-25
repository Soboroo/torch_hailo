API Reference
=============

Public package
--------------

Import the product surface from :mod:`torch_hailo`:

.. code-block:: python

   from torch_hailo import (
       AccuracyReport,
       HailoModel,
       HailoQuantizer,
       compile,
       convert_pt2e,
       get_hailo_quantization_config,
       lower_to_hailo,
       prepare_pt2e,
   )

.. automodule:: torch_hailo
   :no-members:

Compile
-------

.. automodule:: torch_hailo.compile
   :members:
   :no-undoc-members:

Lowering and runtime
--------------------

.. automodule:: torch_hailo.lower
   :members:
   :no-undoc-members:
   :exclude-members: _DeviceSession

Quantization
------------

.. automodule:: torch_hailo.quantizer
   :members:
   :no-undoc-members:

.. automodule:: torch_hailo.qconfig
   :members:
   :no-undoc-members:

.. automodule:: torch_hailo.calibration
   :members:
   :no-undoc-members:

Version helpers
---------------

.. automodule:: torch_hailo.version
   :members:
   :no-undoc-members:
