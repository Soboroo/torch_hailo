"""Check that the Hailo quantizer follows the PT2E tutorial order."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest

import numpy as np
import onnx
import torch
from torch import nn

_HAS_DFC = importlib.util.find_spec("hailo_sdk_client") is not None

from torch_hailo import (
    HailoModel,
    HailoQuantizer,
    compile,
    convert_pt2e,
    get_hailo_quantization_config,
    lower_to_hailo,
    prepare_pt2e,
)
from torch_hailo.lower import CompileResult, HailoCompileRequest
from torch_hailo.version import warn_if_untested_versions


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(4 * 4 * 4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResidualTiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.skip = nn.Conv2d(3, 8, 1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x) + residual
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class Branch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch = nn.Linear(4, 4)
        self.skip = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x) + self.skip(x)


class Flat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(12, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(x, 1))


class Normed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x))


class FakeCompiler:
    def __init__(self) -> None:
        self.request: HailoCompileRequest | None = None
        self.model: onnx.ModelProto | None = None

    def compile(self, request: HailoCompileRequest) -> CompileResult:
        self.request = request
        self.model = onnx.load(request.onnx_path)
        return CompileResult(hef=b"fake-hef", har=b"fake-har")


def _flow(model: nn.Module, example: tuple[torch.Tensor, ...], batches: int = 4):
    exported = torch.export.export(model.eval(), example).module()
    quantizer = HailoQuantizer(hw_arch="hailo8l").set_global(get_hailo_quantization_config())
    prepared = prepare_pt2e(exported, quantizer)
    with torch.no_grad():
        for _ in range(batches):
            prepared(*tuple(torch.randn_like(tensor) for tensor in example))
    quantized = convert_pt2e(prepared)
    return quantizer, quantized


class Pt2eFlowTest(unittest.TestCase):
    def test_tutorial_flow_records_inputs_and_builds_fp32_onnx(self) -> None:
        torch.manual_seed(0)
        eager = Tiny().eval()
        example = (torch.randn(1, 3, 16, 16),)
        eager_out = eager(*example)
        quantizer, quantized = _flow(eager, example, batches=3)

        float_out = quantizer.float_model(*example)
        self.assertTrue(torch.allclose(eager_out, float_out, atol=1e-5, rtol=1e-5))

        with torch.no_grad():
            reference = quantized(*example)
        self.assertEqual(reference.shape, eager_out.shape)
        self.assertTrue(torch.isfinite(reference).all())
        graph_text = str(quantized.graph)
        self.assertIn("quant", graph_text)

        names = list(quantizer.calibration_store.samples)
        self.assertEqual(len(names), 1)
        self.assertEqual(quantizer.calibration_store.num_samples(names[0]), 3)

        compiler = FakeCompiler()
        hailo_model = lower_to_hailo(quantized, example, quantizer, compiler=compiler)
        self.assertEqual(hailo_model.hef, b"fake-hef")
        self.assertEqual(hailo_model.har, b"fake-har")
        self.assertEqual(hailo_model.hw_arch, "hailo8l")
        self.assertIs(hailo_model.reference_model, quantized)
        self.assertEqual(hailo_model.diagnostics["calibration_samples"], 3)
        request = compiler.request
        self.assertIsNotNone(request)
        assert request is not None
        array = next(iter(request.calibration.values()))
        self.assertEqual(array.shape, (3, 3, 16, 16))
        self.assertEqual(array.dtype, np.float32)
        model = compiler.model
        self.assertIsNotNone(model)
        assert model is not None
        self.assertFalse(any(node.op_type == "QuantizeLinear" for node in model.graph.node))
        opset = next(
            item.version
            for item in model.opset_import
            if item.domain in ("", "ai.onnx")
        )
        self.assertLessEqual(opset, 17)
        self.assertEqual(request.input_shapes[request.input_names[0]], [1, 3, 16, 16])

    def test_compile_api_records_calibration_and_reference(self) -> None:
        torch.manual_seed(3)
        model = Tiny().eval()
        example = (torch.randn(1, 3, 16, 16),)
        calibration = [torch.randn(1, 3, 16, 16) for _ in range(5)]
        compiler = FakeCompiler()
        hailo_model = compile(
            model,
            example,
            calibration,
            hw_arch="hailo10h",
            compiler=compiler,
        )
        self.assertEqual(hailo_model.hef, b"fake-hef")
        self.assertIsNotNone(hailo_model.reference_model)
        self.assertEqual(hailo_model.diagnostics["calibration_samples"], 5)
        self.assertEqual(hailo_model.diagnostics["calibration_batches"], 5)
        self.assertTrue(hailo_model.diagnostics["notes"])
        assert compiler.request is not None
        array = next(iter(compiler.request.calibration.values()))
        self.assertEqual(array.shape, (5, 3, 16, 16))
        with torch.no_grad():
            ref = hailo_model.reference_model(*example)
        self.assertEqual(tuple(ref.shape), (1, 5))

    def test_save_and_load_directory(self) -> None:
        model = HailoModel(
            b"fake-hef",
            hw_arch="hailo10h",
            input_names=["x"],
            output_names=["y"],
            har=b"fake-har",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "artifact")
            model.save(path)
            self.assertTrue(os.path.isfile(os.path.join(path, "model.hef")))
            self.assertTrue(os.path.isfile(os.path.join(path, "model.har")))
            self.assertTrue(os.path.isfile(os.path.join(path, "model.json")))
            loaded = HailoModel.load(path)
            self.assertEqual(loaded.hef, b"fake-hef")
            self.assertEqual(loaded.har, b"fake-har")
            self.assertEqual(loaded.input_names, ["x"])
            self.assertEqual(loaded.output_names, ["y"])
            self.assertIsNone(loaded.reference_model)

    def test_forward_without_device_raises(self) -> None:
        model = HailoModel(
            b"fake-hef",
            hw_arch="hailo10h",
            input_names=["x"],
            output_names=["y"],
        )
        with self.assertRaises((ImportError, Exception)):
            model(torch.randn(1, 3, 4, 4))

    def test_emulate_requires_har(self) -> None:
        model = HailoModel(
            b"fake-hef",
            hw_arch="hailo10h",
            input_names=["x"],
            output_names=["y"],
        )
        with self.assertRaises(RuntimeError):
            model.emulate(torch.randn(1, 3, 4, 4))

    def test_compare_to_reference_requires_reference(self) -> None:
        model = HailoModel(
            b"fake-hef",
            hw_arch="hailo10h",
            input_names=["x"],
            output_names=["y"],
            har=b"fake-har",
        )
        with self.assertRaises(RuntimeError):
            model.compare_to_reference(torch.randn(1, 3, 4, 4))

    def test_flatten_records_model_input_not_flattened_activation(self) -> None:
        torch.manual_seed(1)
        example = (torch.randn(1, 3, 2, 2),)
        quantizer, quantized = _flow(Flat().eval(), example, batches=2)
        compiler = FakeCompiler()
        lower_to_hailo(quantized, example, quantizer, compiler=compiler)
        assert compiler.request is not None
        array = next(iter(compiler.request.calibration.values()))
        self.assertEqual(array.shape, (2, 3, 2, 2))

    def test_module_name_config_skips_other_linears(self) -> None:
        torch.manual_seed(2)
        example = (torch.randn(1, 4),)
        model = Branch().eval()
        exported = torch.export.export(model, example).module()
        quantizer = HailoQuantizer().set_module_name("branch", get_hailo_quantization_config())
        prepared = prepare_pt2e(exported, quantizer)
        with torch.no_grad():
            prepared(*example)
        quantized = convert_pt2e(prepared)
        linears = [
            node
            for node in quantized.graph.nodes
            if node.target is torch.ops.aten.linear.default
        ]
        self.assertEqual(len(linears), 2)
        quantized_linear, skipped_linear = linears
        self.assertIn("dequantize", str(quantized_linear.args[1].target))
        self.assertEqual(skipped_linear.args[1].op, "get_attr")
        self.assertIn("skip", str(skipped_linear.args[1].target))

    def test_missing_calibration_raises(self) -> None:
        example = (torch.randn(1, 4),)
        model = nn.Linear(4, 3).eval()
        exported = torch.export.export(model, example).module()
        quantizer = HailoQuantizer().set_global(get_hailo_quantization_config())
        prepared = prepare_pt2e(exported, quantizer)
        quantized = convert_pt2e(prepared)
        with self.assertRaises(RuntimeError):
            lower_to_hailo(quantized, example, quantizer, compiler=FakeCompiler())

    def test_batch_greater_than_one_is_rejected(self) -> None:
        example = (torch.randn(2, 4),)
        quantizer, quantized = _flow(nn.Linear(4, 3).eval(), example, batches=1)
        with self.assertRaises(ValueError):
            lower_to_hailo(quantized, example, quantizer, compiler=FakeCompiler())

    def test_layer_norm_is_rejected_before_compile(self) -> None:
        example = (torch.randn(1, 4),)
        quantizer, quantized = _flow(Normed().eval(), example, batches=1)
        with self.assertRaises(ValueError):
            lower_to_hailo(quantized, example, quantizer, compiler=FakeCompiler())

    def test_set_object_type_accepts_module_and_function(self) -> None:
        config = get_hailo_quantization_config(is_per_channel=False)
        quantizer = HailoQuantizer().set_object_type(nn.Conv2d, config).set_object_type(
            torch.nn.functional.linear, config
        )
        self.assertIn(nn.Conv2d, quantizer.module_type_config)
        self.assertIn(torch.nn.functional.linear, quantizer.operator_type_config)
        self.assertFalse(config.is_qat)
        self.assertEqual(config.weight.qscheme, torch.per_tensor_symmetric)

    def test_version_warning_helper_runs(self) -> None:
        messages = warn_if_untested_versions()
        self.assertIsInstance(messages, list)


@unittest.skipUnless(_HAS_DFC, "hailo_sdk_client is not installed")
class DfcEmulatorTest(unittest.TestCase):
    def test_residual_compile_and_compare(self) -> None:
        torch.manual_seed(0)
        model = ResidualTiny().eval()
        example = (torch.randn(1, 3, 16, 16),)
        calibration = [torch.randn(1, 3, 16, 16) for _ in range(8)]
        hailo_model = compile(
            model,
            example,
            calibration,
            hw_arch="hailo10h",
        )
        self.assertGreater(len(hailo_model.hef), 0)
        self.assertIsNotNone(hailo_model.har)
        report = hailo_model.compare_to_reference(*example)
        self.assertGreaterEqual(report.max_abs_error, 0.0)
        self.assertLess(report.mean_abs_error, 1.0)


if __name__ == "__main__":
    unittest.main()
