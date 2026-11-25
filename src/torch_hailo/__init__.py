"""Build a Hailo NPU model with the PyTorch 2 Export PTQ flow.

Example::

    import torch
    from torch_hailo import compile

    model = Tiny().eval()
    example_inputs = (torch.randn(1, 3, 32, 32),)
    calibration = (torch.randn(1, 3, 32, 32) for _ in range(64))
    hailo_model = compile(model, example_inputs, calibration, hw_arch="hailo10h")
    report = hailo_model.compare_to_reference(*example_inputs)
    hailo_model.save("tiny_hailo")

``prepare_pt2e`` and ``convert_pt2e`` are re-exported from torchao.
The ``convert_pt2e`` result is the quantized reference model used to check
accuracy on CPU. The Hailo Dataflow Compiler does not import external Q/DQ
scales, so ``lower_to_hailo`` / ``compile`` send the batch-norm-folded FP32
graph and the inputs captured during calibration to the DFC.
"""

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from torch_hailo.compile import compile
from torch_hailo.lower import AccuracyReport, HailoModel, lower_to_hailo
from torch_hailo.qconfig import get_hailo_quantization_config
from torch_hailo.quantizer import HailoQuantizer

__all__ = [
    "AccuracyReport",
    "HailoModel",
    "HailoQuantizer",
    "compile",
    "convert_pt2e",
    "get_hailo_quantization_config",
    "lower_to_hailo",
    "prepare_pt2e",
]
