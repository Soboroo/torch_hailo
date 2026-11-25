User Guide
==========

What :func:`~torch_hailo.compile` does
--------------------------------------

:func:`torch_hailo.compile` is the one-call product path. It:

1. Exports the model with ``torch.export``.
2. Runs PT2E prepare / calibrate / convert (static int8).
3. Attaches the converted graph as ``reference_model``.
4. Lowers the batch-norm-folded FP32 graph plus calibration inputs through the
   Hailo Dataflow Compiler to a HEF (and keeps the HAR for emulation).

The DFC does not consume external Q/DQ scales. PT2E is the accuracy reference;
DFC still runs its own calibration on the folded FP32 graph.

Supported scope
---------------

| Item | Status |
| --- | --- |
| Graph | Full graph to one HEF |
| Quantization | Static int8 PTQ only |
| Dynamic shapes | Not supported |
| Batch size | Fixed to 1 |
| Default ``hw_arch`` | ``hailo10h`` |

Compilation
-----------

Pass a ready example input tuple and a calibration iterable of the same shape.
Optional keyword arguments:

- ``hw_arch`` — Hailo architecture string (default ``hailo10h``).
- ``model_script`` — optional ALLS / model-script text loaded by the DFC.
- ``quantizer`` — configured :class:`~torch_hailo.HailoQuantizer` (defaults to a
  global Hailo int8 config).
- ``compiler`` — inject a custom compiler (tests use a fake backend).
- ``allow_unsupported_ops`` — skip the pre-check that rejects ops such as
  ``layer_norm`` and ``embedding``.

.. code-block:: python

   hailo_model = compile(
       model,
       example_inputs,
       calibration,
       hw_arch="hailo10h",
       model_script="performance_param(compiler_optimization_level=max)",
   )

``hw_arch="hailo8"`` and ``allow_hailo8()``
-------------------------------------------

``allow_hailo8()`` is a Hailo model-script (``.alls``) command, not a Python
API. In Dataflow Compiler 5.x, ``hailo8`` is a legacy architecture:
``ClientRunner(hw_arch="hailo8")`` alone is not enough; the script must opt in
with ``allow_hailo8()``.

When you pass ``hw_arch="hailo8"`` to :func:`~torch_hailo.compile` or
:func:`~torch_hailo.lower_to_hailo`, this package inserts that command into the
model script automatically. You do not need a separate function argument.

.. code-block:: python

   hailo_model = compile(model, example_inputs, calibration, hw_arch="hailo8")

Use ``model_script=`` only for other ``.alls`` commands (normalization, NMS,
performance params, and so on). If your script already contains
``allow_hailo8()``, it is not duplicated.

Inference
---------

:class:`~torch_hailo.HailoModel` wraps HEF bytes (and optional HAR).

- :meth:`~torch_hailo.HailoModel.forward` — device inference via HailoRT.
  Prefer :meth:`~torch_hailo.HailoModel.session` so the VDevice / InferModel
  stay open across calls.
- :meth:`~torch_hailo.HailoModel.emulate` — SDK quantized emulator from the HAR
  (no device). Requires a HAR produced at compile time.
- :meth:`~torch_hailo.HailoModel.compare_to_reference` — max / mean absolute
  error against ``reference_model``.

Persistence
-----------

:meth:`~torch_hailo.HailoModel.save` writes ``{path}.hef`` and, when present,
``{path}.har`` (plus a small JSON sidecar). :meth:`~torch_hailo.HailoModel.load`
restores HEF/HAR. The CPU ``reference_model`` is not serialized; re-attach it
after load if you still need accuracy checks.

Advanced APIs
-------------

For step-by-step control:

1. ``torch.export.export(...).module()``
2. :func:`torch_hailo.prepare_pt2e` (re-exported from torchao)
3. Run calibration under ``torch.no_grad()``
4. :func:`torch_hailo.convert_pt2e` (re-exported from torchao)
5. :func:`torch_hailo.lower_to_hailo`

Quantizer helpers live under :mod:`torch_hailo.quantizer`
(:class:`~torch_hailo.HailoQuantizer`,
:func:`~torch_hailo.get_hailo_quantization_config`).

Version check
-------------

:func:`torch_hailo.version.warn_if_untested_versions` emits warnings when the
installed DFC / PyTorch / torchao major.minor pins differ from the tested set
(DFC 5.5, torch 2.9, torchao 0.17). It runs automatically inside the DFC
compiler path.

Limits and pitfalls
-------------------

- Calibration must use ``torch.no_grad()``. ``torch.inference_mode()`` breaks
  observers.
- Keep ONNX / DFC opset at 17 or below for DFC 5.5.
- Rank-4 inputs are NCHW in PyTorch; DFC calibration is converted to NHWC.
- ``layer_norm``, ``group_norm``, ``scaled_dot_product_attention``,
  ``embedding``, and ``native_multi_head_attention`` are rejected before
  compile unless ``allow_unsupported_ops=True``.
- Without a real DFC, unit tests inject a fake compiler so CI can run without
  Hailo packages.
