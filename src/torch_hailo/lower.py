"""Lower a Q/DQ reference model to a Hailo HEF."""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np
import onnx
import torch
from torch.fx import GraphModule

from torch_hailo.quantizer import HailoQuantizer
from torch_hailo.version import warn_if_untested_versions

_BLOCKED_OP_PARTS = (
    "layer_norm",
    "group_norm",
    "scaled_dot_product_attention",
    "embedding",
    "native_multi_head_attention",
)

_HEF_NAME = "model.hef"
_HAR_NAME = "model.har"
_META_NAME = "model.json"


@dataclass
class HailoCompileRequest:
    """FP32 ONNX and calibration inputs passed to the DFC."""

    onnx_path: str
    calibration: dict[str, np.ndarray]
    hw_arch: str
    net_name: str
    input_names: list[str]
    output_names: list[str]
    input_shapes: dict[str, list[int]]
    model_script: str | None = None


@dataclass
class CompileResult:
    """Binary artifacts produced by a Hailo compiler backend."""

    hef: bytes
    har: bytes | None = None


class HailoCompiler(Protocol):
    def compile(self, request: HailoCompileRequest) -> CompileResult | bytes: ...


class HailoDFCCompiler:
    """Build HEF and HAR bytes with the Hailo Dataflow Compiler.

    ``hailo_sdk_client`` is not on PyPI. Install the Developer Zone wheel on
    an x86_64 Linux compile host.
    """

    def compile(self, request: HailoCompileRequest) -> CompileResult:
        warn_if_untested_versions()
        try:
            from hailo_sdk_client import ClientRunner
        except ImportError as exc:
            raise ImportError(
                "Hailo Dataflow Compiler (hailo_sdk_client) is not installed. "
                "Install the Hailo Developer Zone DFC wheel on an x86_64 Linux "
                "compile host, then call lower_to_hailo again. The convert_pt2e "
                "reference model can be evaluated on CPU without the DFC."
            ) from exc

        runner = ClientRunner(hw_arch=request.hw_arch)
        translate_kwargs = {}
        if request.input_shapes:
            translate_kwargs["net_input_shapes"] = request.input_shapes
        try:
            runner.translate_onnx_model(request.onnx_path, request.net_name, **translate_kwargs)
        except TypeError:
            runner.translate_onnx_model(request.onnx_path, request.net_name)
        model_script = request.model_script or ""
        if request.hw_arch == "hailo8" and "allow_hailo8" not in model_script:
            # DFC 5.x disables hailo8 unless the model script opts in.
            model_script = "allow_hailo8()\n" + model_script
        if model_script:
            runner.load_model_script(model_script)
        calibration = _calibration_for_optimize(request.calibration)
        runner.optimize(calibration)
        compiled = runner.compile()
        hef = _as_hef_bytes(compiled)
        har: bytes | None = None
        with tempfile.TemporaryDirectory(prefix="torch_hailo_har_") as directory:
            har_path = os.path.join(directory, _HAR_NAME)
            runner.save_har(har_path)
            with open(har_path, "rb") as har_file:
                har = har_file.read()
        return CompileResult(hef=hef, har=har)


@dataclass
class AccuracyReport:
    """Absolute error between the CPU reference and the quantized emulator."""

    max_abs_error: float
    mean_abs_error: float


class HailoModel(torch.nn.Module):
    """Module that holds a HEF and optional HAR for HailoRT or DFC emulation.

    Inputs are ``float32`` tensors in the same layout passed to ``lower_to_hailo``.
    A 4-D NCHW tensor is converted when the device or emulator input shape is NHWC.
    """

    def __init__(
        self,
        hef: bytes,
        *,
        hw_arch: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        har: bytes | None = None,
        reference_model: GraphModule | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hef = bytes(hef)
        self.har = bytes(har) if har is not None else None
        self.hw_arch = hw_arch
        self.input_names = list(input_names)
        self.output_names = list(output_names)
        self.reference_model = reference_model
        self.diagnostics: dict[str, Any] = dict(diagnostics or {})
        self._session: _DeviceSession | None = None

    def save(self, path: str | os.PathLike) -> None:
        """Write ``model.hef`` and, when present, ``model.har`` under ``path``."""

        import json

        directory = os.fspath(path)
        if directory.endswith(".hef"):
            parent = os.path.dirname(directory) or "."
            os.makedirs(parent, exist_ok=True)
            with open(directory, "wb") as hef_file:
                hef_file.write(self.hef)
            return
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, _HEF_NAME), "wb") as hef_file:
            hef_file.write(self.hef)
        if self.har is not None:
            with open(os.path.join(directory, _HAR_NAME), "wb") as har_file:
                har_file.write(self.har)
        with open(os.path.join(directory, _META_NAME), "w", encoding="utf-8") as meta_file:
            json.dump(
                {
                    "hw_arch": self.hw_arch,
                    "input_names": self.input_names,
                    "output_names": self.output_names,
                    "diagnostics": self.diagnostics,
                },
                meta_file,
            )

    @classmethod
    def load(cls, path: str | os.PathLike, *, hw_arch: str | None = None) -> HailoModel:
        """Load a directory written by :meth:`save`.

        The CPU ``reference_model`` is not restored.
        """

        import json

        directory = os.fspath(path)
        hef_path = directory if directory.endswith(".hef") else os.path.join(directory, _HEF_NAME)
        har_path = None if directory.endswith(".hef") else os.path.join(directory, _HAR_NAME)
        meta_path = None if directory.endswith(".hef") else os.path.join(directory, _META_NAME)
        with open(hef_path, "rb") as hef_file:
            hef = hef_file.read()
        har = None
        if har_path is not None and os.path.isfile(har_path):
            with open(har_path, "rb") as har_file:
                har = har_file.read()
        meta: dict[str, Any] = {}
        if meta_path is not None and os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as meta_file:
                meta = json.load(meta_file)
        input_names = list(meta.get("input_names") or [])
        output_names = list(meta.get("output_names") or [])
        if not input_names or not output_names:
            inferred_inputs, inferred_outputs = _names_from_hef(hef)
            input_names = input_names or inferred_inputs
            output_names = output_names or inferred_outputs
        return cls(
            hef,
            hw_arch=hw_arch or str(meta.get("hw_arch") or "hailo10h"),
            input_names=input_names,
            output_names=output_names,
            har=har,
            diagnostics=meta.get("diagnostics"),
        )

    @contextlib.contextmanager
    def session(self) -> Iterator[HailoModel]:
        """Keep a HailoRT device configured for repeated ``forward`` calls."""

        if self._session is not None:
            raise RuntimeError("A HailoModel session is already active.")
        self._session = _DeviceSession(self)
        try:
            self._session.open()
            yield self
        finally:
            self._session.close()
            self._session = None

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        self._check_inputs(args, kwargs)
        if self._session is not None:
            return self._session.infer(args)
        return _infer_once(self, args)

    def emulate(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run the quantized model on the DFC host emulator."""

        self._check_inputs(args, kwargs)
        if self.har is None:
            raise RuntimeError(
                "No HAR is attached to this HailoModel. Recompile with the "
                "Hailo Dataflow Compiler so save_har can capture quantized state."
            )
        warn_if_untested_versions()
        try:
            from hailo_sdk_client import ClientRunner, InferenceContext
        except ImportError as exc:
            raise ImportError(
                "emulate() requires hailo_sdk_client from the Hailo Dataflow Compiler."
            ) from exc

        with tempfile.TemporaryDirectory(prefix="torch_hailo_emu_") as directory:
            har_path = os.path.join(directory, _HAR_NAME)
            with open(har_path, "wb") as har_file:
                har_file.write(self.har)
            runner = ClientRunner(hw_arch=self.hw_arch)
            runner.load_har(har_path)
            feed = _emulator_feed(args, self.input_names)
            with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
                result = runner.infer(ctx, feed, batch_size=1)
        return _emulator_outputs(result, args[0].device)

    def compare_to_reference(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> AccuracyReport:
        """Compare the CPU ``convert_pt2e`` reference against ``emulate``."""

        self._check_inputs(args, kwargs)
        if self.reference_model is None:
            raise RuntimeError(
                "No reference_model is attached. compare_to_reference needs the "
                "convert_pt2e GraphModule from the compile process."
            )
        with torch.no_grad():
            reference = self.reference_model(*[tensor.detach().cpu() for tensor in args])
        emulated = self.emulate(*args)
        ref_tensors = reference if isinstance(reference, tuple) else (reference,)
        emu_tensors = emulated if isinstance(emulated, tuple) else (emulated,)
        if len(ref_tensors) != len(emu_tensors):
            raise RuntimeError(
                f"Reference returned {len(ref_tensors)} outputs, emulator returned {len(emu_tensors)}."
            )
        diffs = []
        for left, right in zip(ref_tensors, emu_tensors):
            diffs.append((left.detach().cpu().float() - right.detach().cpu().float()).abs())
        stacked = torch.cat([diff.reshape(-1) for diff in diffs])
        return AccuracyReport(
            max_abs_error=float(stacked.max().item()),
            mean_abs_error=float(stacked.mean().item()),
        )

    def _check_inputs(self, args: Sequence[torch.Tensor], kwargs: Mapping[str, torch.Tensor]) -> None:
        if kwargs:
            raise TypeError("HailoModel accepts positional arguments only.")
        if len(args) != len(self.input_names):
            raise TypeError(
                f"Expected {len(self.input_names)} inputs, got {len(args)}."
            )


class _DeviceSession:
    def __init__(self, model: HailoModel) -> None:
        self.model = model
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._target = None
        self._activate = None
        self._pipeline = None
        self._input_infos = None
        self._output_infos = None

    def open(self) -> None:
        try:
            from hailo_platform import HEF, InferVStreams, VDevice
            from hailo_platform import FormatType, InputVStreamParams, OutputVStreamParams
        except ImportError as exc:
            raise ImportError(
                "Running a HEF requires hailo_platform (HailoRT) and a visible Hailo device."
            ) from exc

        self._tmpdir = tempfile.TemporaryDirectory(prefix="torch_hailo_")
        hef_path = os.path.join(self._tmpdir.name, _HEF_NAME)
        with open(hef_path, "wb") as hef_file:
            hef_file.write(self.model.hef)
        hef = HEF(hef_path)
        self._target = VDevice()
        self._target.__enter__()
        network_group = self._target.configure(hef)[0]
        self._input_infos = list(hef.get_input_vstream_infos())
        self._output_infos = list(hef.get_output_vstream_infos())
        input_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        self._activate = network_group.activate()
        self._activate.__enter__()
        self._pipeline = InferVStreams(network_group, input_params, output_params)
        self._pipeline.__enter__()

    def close(self) -> None:
        for handle in (self._pipeline, self._activate, self._target):
            if handle is not None:
                handle.__exit__(None, None, None)
        self._pipeline = None
        self._activate = None
        self._target = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def infer(self, args: Sequence[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, ...]:
        assert self._pipeline is not None and self._input_infos is not None
        assert self._output_infos is not None
        feeds = _feeds_for_vstreams(args, self.model.input_names, self._input_infos)
        result = self._pipeline.infer(feeds)
        outputs = _outputs_from_result(result, self._output_infos, args[0].device)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def _infer_once(model: HailoModel, args: Sequence[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, ...]:
    with model.session():
        assert model._session is not None
        return model._session.infer(args)


def lower_to_hailo(
    quantized_model: GraphModule,
    example_inputs: torch.Tensor | Sequence[torch.Tensor],
    quantizer: HailoQuantizer,
    *,
    hw_arch: str | None = None,
    output_path: str | os.PathLike | None = None,
    model_script: str | None = None,
    net_name: str = "model",
    calibration_data: np.ndarray | Mapping[str, np.ndarray] | None = None,
    allow_unsupported_ops: bool = False,
    compiler: HailoCompiler | None = None,
) -> HailoModel:
    """Lower a ``convert_pt2e`` model to a Hailo HEF.

    ``example_inputs`` must use the same static shape passed to ``torch.export``,
    with batch size 1. Calibration samples accumulate on ``quantizer`` when the
    model returned by ``prepare_pt2e`` is run under ``torch.no_grad()`` on
    representative data.
    """

    if not isinstance(quantizer, HailoQuantizer):
        raise TypeError("quantizer must be the HailoQuantizer instance passed to prepare_pt2e.")
    if quantizer.float_model is None:
        raise RuntimeError(
            "The FP32 graph is missing. Call prepare_pt2e with this quantizer first."
        )
    _require_converted(quantized_model)
    inputs = _normalize_inputs(example_inputs)
    _require_static_batch_one(inputs)
    if not allow_unsupported_ops:
        _reject_unsupported_ops(quantizer.float_model)

    calibration = _resolve_calibration(quantizer, inputs, calibration_data)
    arch = hw_arch or quantizer.hw_arch
    input_names = _placeholder_names(quantizer.float_model)
    if len(input_names) != len(inputs):
        raise RuntimeError(
            f"The FP32 graph has {len(input_names)} inputs, but {len(inputs)} examples were given."
        )
    output_names = _output_names(quantizer.float_model, inputs)
    input_shapes = {
        name: [int(dim) for dim in tensor.shape] for name, tensor in zip(input_names, inputs)
    }
    recorded = quantizer.calibration_store.num_samples()
    diagnostics: dict[str, Any] = {
        "calibration_samples": recorded,
        "notes": [],
    }
    if recorded < 1024:
        diagnostics["notes"].append(
            f"Calibration recorded {recorded} samples. "
            "The Hailo DFC may reduce the optimization level below the recommended set size of 1024."
        )

    with tempfile.TemporaryDirectory(prefix="torch_hailo_onnx_") as directory:
        onnx_path = os.path.join(directory, f"{net_name}.onnx")
        _export_onnx(
            quantizer.float_model,
            inputs,
            onnx_path,
            input_names,
            output_names,
        )
        _finalize_onnx(onnx_path)
        request = HailoCompileRequest(
            onnx_path=onnx_path,
            calibration=calibration,
            hw_arch=arch,
            net_name=net_name,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes,
            model_script=model_script,
        )
        backend = compiler if compiler is not None else HailoDFCCompiler()
        result = _normalize_compile_result(backend.compile(request))

    hailo_model = HailoModel(
        result.hef,
        hw_arch=arch,
        input_names=input_names,
        output_names=output_names,
        har=result.har,
        reference_model=quantized_model,
        diagnostics=diagnostics,
    )
    if output_path is not None:
        hailo_model.save(output_path)
    return hailo_model


def _normalize_compile_result(result: CompileResult | bytes) -> CompileResult:
    if isinstance(result, CompileResult):
        return result
    if isinstance(result, (bytes, bytearray)):
        return CompileResult(hef=bytes(result))
    raise TypeError(f"Compiler returned unsupported type {type(result).__name__}")


def _as_hef_bytes(compiled: object) -> bytes:
    if isinstance(compiled, (bytes, bytearray)):
        return bytes(compiled)
    if isinstance(compiled, os.PathLike) or isinstance(compiled, str):
        with open(compiled, "rb") as hef_file:
            return hef_file.read()
    payload = getattr(compiled, "hef", None)
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    raise TypeError(
        "Could not read ClientRunner.compile() output as HEF bytes: "
        f"{type(compiled).__name__}"
    )


def _names_from_hef(hef: bytes) -> tuple[list[str], list[str]]:
    try:
        from hailo_platform import HEF
    except ImportError:
        return ["input"], ["output"]
    try:
        with tempfile.TemporaryDirectory(prefix="torch_hailo_hef_") as directory:
            path = os.path.join(directory, _HEF_NAME)
            with open(path, "wb") as hef_file:
                hef_file.write(hef)
            loaded = HEF(path)
            inputs = [info.name for info in loaded.get_input_vstream_infos()]
            outputs = [info.name for info in loaded.get_output_vstream_infos()]
        return inputs or ["input"], outputs or ["output"]
    except Exception:
        return ["input"], ["output"]


def _normalize_inputs(
    example_inputs: torch.Tensor | Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if isinstance(example_inputs, torch.Tensor):
        return (example_inputs,)
    inputs = tuple(example_inputs)
    if not inputs or any(not isinstance(item, torch.Tensor) for item in inputs):
        raise TypeError("example_inputs must be a tensor or a tuple of tensors.")
    return inputs


def _require_static_batch_one(inputs: Sequence[torch.Tensor]) -> None:
    for index, tensor in enumerate(inputs):
        if tensor.ndim == 0 or int(tensor.shape[0]) != 1:
            raise ValueError(
                "Hailo compilation supports static shapes with batch size 1 only. "
                f"Input {index} has shape {tuple(tensor.shape)}. "
                "Set the batch dimension of the torch.export example_inputs to 1."
            )


def _require_converted(model: GraphModule) -> None:
    for node in model.graph.nodes:
        if node.op == "call_function" and "quant" in _op_name(node.target):
            return
    raise ValueError(
        "Pass the model returned by convert_pt2e to lower_to_hailo. "
        "The model returned by prepare_pt2e is not a quantized reference yet."
    )


def _reject_unsupported_ops(model: GraphModule) -> None:
    found: list[str] = []
    for node in model.graph.nodes:
        if node.op != "call_function":
            continue
        name = _op_name(node.target)
        if any(part in name for part in _BLOCKED_OP_PARTS):
            found.append(name)
    if found:
        unique = ", ".join(sorted(set(found)))
        raise ValueError(
            "This graph cannot be compiled to Hailo as a single model. "
            "Replace the unsupported ops or split the model. "
            f"Blocked ops: {unique}. "
            "Pass allow_unsupported_ops=True to lower_to_hailo to continue experimentally."
        )


def _resolve_calibration(
    quantizer: HailoQuantizer,
    inputs: Sequence[torch.Tensor],
    calibration_data: np.ndarray | Mapping[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    names = _placeholder_names(quantizer.float_model)
    if calibration_data is None:
        arrays = quantizer.calibration_store.arrays()
        if not arrays:
            raise RuntimeError(
                "No calibration inputs were recorded. Before convert_pt2e, run the model "
                "returned by prepare_pt2e on representative data inside torch.no_grad(). "
                "inference_mode can prevent observers from collecting statistics."
            )
    elif isinstance(calibration_data, np.ndarray):
        if len(names) != 1:
            raise ValueError("Pass calibration_data as a dict keyed by input name when there are multiple inputs.")
        arrays = {names[0]: np.ascontiguousarray(calibration_data, dtype=np.float32)}
    else:
        arrays = {
            name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in calibration_data.items()
        }
    missing = [name for name in names if name not in arrays]
    if missing:
        raise RuntimeError(
            "Calibration samples are missing graph inputs: " + ", ".join(missing)
        )
    resolved: dict[str, np.ndarray] = {}
    for name, tensor in zip(names, inputs):
        array = arrays[name]
        expected = tuple(int(dim) for dim in tensor.shape[1:])
        actual = tuple(int(dim) for dim in array.shape[1:])
        if array.ndim != tensor.ndim or actual != expected:
            raise ValueError(
                f"Calibration shape for input {name!r} is {array.shape}, "
                f"but the example input shape is {tuple(tensor.shape)}. "
                "Dimensions after the batch axis must match."
            )
        if array.shape[0] < 1:
            raise ValueError(f"Calibration samples for input {name!r} are empty.")
        resolved[name] = np.ascontiguousarray(array, dtype=np.float32)
    return resolved


def _placeholder_names(model: GraphModule) -> list[str]:
    names = []
    used: set[str] = set()
    for index, node in enumerate(node for node in model.graph.nodes if node.op == "placeholder"):
        name = _onnx_identifier(node.name, f"input_{index}")
        while name in used:
            name = f"{name}_{index}"
        used.add(name)
        names.append(name)
    return names


def _output_names(model: GraphModule, inputs: Sequence[torch.Tensor]) -> list[str]:
    with torch.no_grad():
        result = model(*[tensor.detach().cpu() for tensor in inputs])
    count = len(result) if isinstance(result, tuple) else 1
    if count == 1:
        return ["output"]
    return [f"output_{index}" for index in range(count)]


class _LegacyExportWrapper(torch.nn.Module):
    """Wrap a torch.export module so the legacy ONNX exporter can call train()."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def train(self, mode: bool = True):  # noqa: ARG002
        return self

    def forward(self, *args):
        return self.inner(*args)


def _export_onnx(
    model: GraphModule,
    inputs: Sequence[torch.Tensor],
    path: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> None:
    cpu_inputs = tuple(tensor.detach().cpu() for tensor in inputs)
    exportable = model.eval()
    try:
        exportable = exportable.cpu()
    except Exception:
        pass
    try:
        program = torch.export.export(exportable, cpu_inputs)
        torch.onnx.export(
            program,
            cpu_inputs,
            path,
            input_names=list(input_names),
            output_names=list(output_names),
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
        return
    except Exception:
        # torch 2.9 and onnxscript 0.5, which DFC 5.5 pins, do not finish the
        # dynamo exporter's opset conversion. The TorchScript exporter emits opset 17 directly.
        torch.onnx.export(
            _LegacyExportWrapper(exportable),
            cpu_inputs,
            path,
            input_names=list(input_names),
            output_names=list(output_names),
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )


def _finalize_onnx(path: str) -> None:
    model = onnx.load(path)
    blocked = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        }
    )
    if blocked:
        raise RuntimeError(
            "The ONNX graph contains QuantizeLinear/DequantizeLinear. "
            "The Hailo DFC does not import external Q/DQ quantization, so only an FP32 graph is compiled. "
            f"Found ops: {', '.join(blocked)}"
        )
    version = _default_opset(model)
    if version is not None and version > 17:
        model = onnx.version_converter.convert_version(model, 17)
    onnx.save(model, path)


def _default_opset(model: onnx.ModelProto) -> int | None:
    for imported in model.opset_import:
        if imported.domain in ("", "ai.onnx"):
            return int(imported.version)
    return None


def _calibration_for_optimize(calibration: Mapping[str, np.ndarray]) -> np.ndarray | dict[str, np.ndarray]:
    # The DFC converts ONNX NCHW inputs to Hailo NHWC and expects calibration in that layout.
    aligned = {name: _nchw_to_nhwc(array) for name, array in calibration.items()}
    if len(aligned) == 1:
        return next(iter(aligned.values()))
    return aligned


def _nchw_to_nhwc(array: np.ndarray) -> np.ndarray:
    if array.ndim != 4:
        return array
    return np.ascontiguousarray(np.transpose(array, (0, 2, 3, 1)))


def _op_name(target: object) -> str:
    name = getattr(target, "name", None)
    if callable(name):
        try:
            return str(target.name())
        except Exception:
            pass
    return str(target)


def _onnx_identifier(name: str, fallback: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    if not cleaned or cleaned[0].isdigit():
        return fallback
    return cleaned


def _feeds_for_vstreams(args, input_names, input_infos) -> dict[str, np.ndarray]:
    info_by_name = {info.name: info for info in input_infos}
    feeds = {}
    for tensor, name in zip(args, input_names):
        info = info_by_name.get(name, input_infos[len(feeds)] if input_infos else None)
        array = tensor.detach().cpu().contiguous().float().numpy()
        if info is not None:
            array = _align_layout(array, tuple(info.shape))
        feeds[info.name if info is not None else name] = array
    return feeds


def _align_layout(array: np.ndarray, vstream_shape: tuple[int, ...]) -> np.ndarray:
    if array.ndim == 4 and len(vstream_shape) == 4:
        nchw = array.shape
        nhwc = (nchw[0], nchw[2], nchw[3], nchw[1])
        if _shape_compatible(nhwc, vstream_shape) and not _shape_compatible(nchw, vstream_shape):
            return np.ascontiguousarray(np.transpose(array, (0, 2, 3, 1)))
    return array


def _shape_compatible(actual: tuple[int, ...], expected: tuple[int, ...]) -> bool:
    if len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected):
        if right in (-1, 0):
            continue
        if left != right:
            return False
    return True


def _outputs_from_result(result, output_infos, device: torch.device) -> list[torch.Tensor]:
    tensors = []
    for info in output_infos:
        value = result[info.name] if isinstance(result, dict) else result
        array = np.asarray(value)
        if array.ndim == 4:
            vshape = tuple(info.shape)
            nchw = (array.shape[0], array.shape[3], array.shape[1], array.shape[2])
            if _shape_compatible(nchw, vshape) or (
                len(vshape) == 4 and vshape[1] not in (-1, 0) and array.shape[-1] == vshape[1]
            ):
                array = np.transpose(array, (0, 3, 1, 2))
        tensors.append(torch.from_numpy(np.ascontiguousarray(array)).to(device))
    return tensors


def _emulator_feed(args: Sequence[torch.Tensor], input_names: Sequence[str]):
    arrays = []
    for tensor in args:
        array = tensor.detach().cpu().contiguous().float().numpy()
        arrays.append(_nchw_to_nhwc(array))
    if len(arrays) == 1:
        return arrays[0]
    return {name: array for name, array in zip(input_names, arrays)}


def _emulator_outputs(result, device: torch.device) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(result, dict):
        values = list(result.values())
    elif isinstance(result, (list, tuple)):
        values = list(result)
    else:
        values = [result]
    tensors = []
    for value in values:
        array = np.asarray(value)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim == 4 and array.shape[-1] not in (0, 1) and array.shape[1] != array.shape[-1]:
            # Prefer NHWC -> NCHW when channels look like the last axis.
            if array.shape[-1] <= 512 and array.shape[1] == array.shape[2]:
                array = np.transpose(array, (0, 3, 1, 2))
        tensors.append(torch.from_numpy(np.ascontiguousarray(array)).to(device))
    if len(tensors) == 1:
        return tensors[0]
    return tuple(tensors)
