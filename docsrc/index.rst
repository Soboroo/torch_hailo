torch-hailo
===========

**torch-hailo** compiles PyTorch models for Hailo NPUs using the
`PyTorch 2 Export PTQ <https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_ptq.html>`_
flow. Call :func:`torch_hailo.compile` to export, calibrate, convert, and lower
a model to a HEF. The ``convert_pt2e`` graph remains the CPU accuracy reference.
The Hailo Dataflow Compiler does not import external Q/DQ scales, so lowering
sends the batch-norm-folded FP32 graph and captured calibration inputs to the DFC.

**New here?** Start with :doc:`getting_started`, then read :doc:`user_guide`.

**Looking up a symbol?** See :doc:`api_reference`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   user_guide
   api_reference

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
