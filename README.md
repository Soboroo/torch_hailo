# torch-hailo

Compile a PyTorch model to a Hailo HEF with the [PyTorch 2 Export PTQ](https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_ptq.html) flow.

`compile()` runs `torch.export` → `prepare_pt2e` → calibration → `convert_pt2e` → Dataflow Compiler. The `convert_pt2e` graph is the CPU accuracy reference. The Hailo Dataflow Compiler does not import external Q/DQ scales, so lowering sends the batch-norm-folded FP32 graph and the captured calibration inputs to the DFC.

## Documentation

Usage guides and the API reference live under [`docsrc/`](docsrc/), in the same style as [torch_tensorrt/docsrc](https://github.com/pytorch/TensorRT/tree/main/docsrc). Build the HTML site with:

```bash
pip install -r docsrc/requirements.txt
sphinx-build -b html docsrc docsrc/_build/html
```

Then open `docsrc/_build/html/index.html`.

This repository does not redistribute Hailo software. Installers (`*.run`), Dataflow Compiler wheels, HailoRT, drivers (`*.deb`), and an extracted `hailo_ai_sw_suite/` are excluded from Git and source distributions. Install the suite yourself from the [Hailo Developer Zone](https://hailo.ai/developer-zone/).

## Prerequisites

- Python 3.10+
- x86_64 Linux compile host for HEF builds
- Hailo AI Software Suite (Dataflow Compiler 5.5 and HailoRT) from the [Hailo Developer Zone](https://hailo.ai/developer-zone/). `hailo_sdk_client` is not on PyPI
- Default target is `hailo10h`
- Compilation and `emulate()` work without a Hailo device. `forward()` needs HailoRT and a visible device

## Install

Install the Hailo suite so it creates `hailo_venv`, then install this package into that environment. DFC 5.5 pins PyTorch 2.9.1, so use `torchao==0.17.0`.

```bash
source /path/to/hailo_ai_sw_suite/hailo_venv/bin/activate
pip install torchao==0.17.0
pip install -e /path/to/torch_hailo
python -m unittest tests.test_pt2e_flow
```

## Example

Batch size must be 1 with a static shape. Calibrate under `torch.no_grad()`, not `torch.inference_mode()`. With fewer than 1024 calibration samples the DFC may drop the optimization level.

```python
import torch
from torch import nn
from torch_hailo import compile


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

from torch_hailo import HailoModel

hailo_model.save("tiny_hailo")
loaded = HailoModel.load("tiny_hailo")  # HEF + HAR only; no CPU reference
```

On a device:

```python
with hailo_model.session():
    out = hailo_model(*example_inputs)
```

Activations are int8 per-tensor affine and weights are int8 symmetric. Pass a configured `HailoQuantizer` to `compile(..., quantizer=...)` to override the default. Use `set_module_name`, `set_module_type`, or `set_object_type` for selective quantization.

### `hw_arch="hailo8"` and `allow_hailo8()`

`allow_hailo8()` is a Hailo model-script (`.alls`) command, not a `torch_hailo` Python API. In Dataflow Compiler 5.x, `hailo8` is a legacy architecture: `ClientRunner(hw_arch="hailo8")` alone is not enough, and the script must opt in with `allow_hailo8()`.

When you pass `hw_arch="hailo8"` to `compile()` or `lower_to_hailo()`, this package inserts that command into the model script automatically. You do not need a separate function argument.

```python
hailo_model = compile(model, example_inputs, calibration, hw_arch="hailo8")
```

Use `model_script=` only for other `.alls` commands (normalization, NMS, performance params, and so on). If your script already contains `allow_hailo8()`, it is not duplicated.

## Emulator

`HailoModel.emulate()` loads the attached HAR and runs `InferenceContext.SDK_QUANTIZED` on the host. `compare_to_reference()` measures absolute error against the in-process `convert_pt2e` graph. Loaded artifacts keep HEF/HAR only.

## Limits

- Batch size 1 and the same static shape used for `torch.export`
- Rank-4 inputs are NCHW in PyTorch; DFC calibration is converted to NHWC
- `layer_norm`, `group_norm`, `scaled_dot_product_attention`, `embedding`, and `native_multi_head_attention` are rejected before compile
