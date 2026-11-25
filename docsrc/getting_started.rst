Getting Started
===============

Prerequisites
-------------

- Python 3.10+
- x86_64 Linux for HEF compilation
- Hailo AI Software Suite (Dataflow Compiler 5.5 and HailoRT) from the
  `Hailo Developer Zone <https://hailo.ai/developer-zone/>`_.
  ``hailo_sdk_client`` is not on PyPI.
- Default hardware target is ``hailo10h``.
- Compilation and :meth:`~torch_hailo.HailoModel.emulate` work without a Hailo
  device. :meth:`~torch_hailo.HailoModel.forward` needs HailoRT and a visible device.

This repository does not redistribute Hailo installers, wheels, or drivers.
Install the suite yourself, then put ``torch-hailo`` into the suite venv.

Install
-------

The suite creates ``hailo_venv``. Activate it, install a torchao build that
matches DFC 5.5 (PyTorch 2.9.1), then install this package.

.. code-block:: bash

   source /path/to/hailo_ai_sw_suite/hailo_venv/bin/activate
   pip install torchao==0.17.0
   pip install -e /path/to/torch_hailo
   python -m unittest tests.test_pt2e_flow

Quick start
-----------

Batch size must be 1 with a static shape. Run calibration under
``torch.no_grad()``, not ``torch.inference_mode()``.

.. code-block:: python

   import torch
   from torch import nn
   from torch_hailo import HailoModel, compile


   class Tiny(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv = nn.Conv2d(3, 8, 3, padding=1)
           self.relu = nn.ReLU()

       def forward(self, x):
           return self.relu(self.conv(x))


   torch.manual_seed(0)
   model = Tiny().eval()
   example_inputs = (torch.randn(1, 3, 32, 32),)
   calibration = (torch.randn(1, 3, 32, 32) for _ in range(64))

   hailo_model = compile(model, example_inputs, calibration, hw_arch="hailo10h")
   report = hailo_model.compare_to_reference(*example_inputs)
   print(report.max_abs_error, report.mean_abs_error)

   hailo_model.save("tiny_hailo")
   loaded = HailoModel.load("tiny_hailo")

On a device, keep the runtime session open for repeated calls:

.. code-block:: python

   with hailo_model.session():
       out = hailo_model(*example_inputs)

Build this documentation
------------------------

.. code-block:: bash

   pip install -r docsrc/requirements.txt
   sphinx-build -b html docsrc docsrc/_build/html
