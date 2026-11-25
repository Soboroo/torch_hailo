"""PT2E quantizer for the Hailo backend."""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn.functional as F
from torch.fx import Node
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
    get_module_name_filter,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

from torch_hailo.calibration import CalibrationStore

def _overloads(*packets: object) -> set[object]:
    targets: set[object] = set()
    for packet in packets:
        for name in packet.overloads():  # type: ignore[attr-defined]
            targets.add(getattr(packet, name))
    return targets


_WEIGHTED_TARGETS = _overloads(
    torch.ops.aten.linear,
    torch.ops.aten.conv1d,
    torch.ops.aten.conv2d,
    torch.ops.aten.conv3d,
    torch.ops.aten.convolution,
    torch.ops.aten.conv_transpose1d,
    torch.ops.aten.conv_transpose2d,
)
_TRANSPOSE_TARGETS = _overloads(
    torch.ops.aten.conv_transpose1d,
    torch.ops.aten.conv_transpose2d,
)
_RELU_TARGETS = {
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
}
_OPERATOR_ALIASES: dict[object, set[object]] = {
    F.linear: {torch.ops.aten.linear.default},
    torch.nn.Linear: {torch.ops.aten.linear.default},
    F.conv1d: {torch.ops.aten.conv1d.default, torch.ops.aten.convolution.default},
    F.conv2d: {torch.ops.aten.conv2d.default, torch.ops.aten.convolution.default},
    F.conv3d: {torch.ops.aten.conv3d.default, torch.ops.aten.convolution.default},
    torch.ops.aten.conv1d: {
        torch.ops.aten.conv1d.default,
        torch.ops.aten.convolution.default,
    },
    torch.ops.aten.conv2d: {
        torch.ops.aten.conv2d.default,
        torch.ops.aten.convolution.default,
    },
    torch.ops.aten.conv3d: {
        torch.ops.aten.conv3d.default,
        torch.ops.aten.convolution.default,
    },
    torch.ops.aten.linear: {torch.ops.aten.linear.default},
}


def _derive_bias_qparams(obs_or_fqs) -> tuple[torch.Tensor, torch.Tensor]:
    activation_scale, _ = obs_or_fqs[0].calculate_qparams()
    weight_scale, _ = obs_or_fqs[1].calculate_qparams()
    bias_scale = activation_scale.to(torch.float32) * weight_scale.to(torch.float32)
    bias_scale = bias_scale.reshape(-1)
    bias_zero_point = torch.zeros(bias_scale.shape, dtype=torch.int32)
    return bias_scale, bias_zero_point


def _module_type_filter(module_type: Callable) -> Callable[[Node], bool]:
    type_name = module_type.__module__ + "." + module_type.__qualname__

    def module_filter(node: Node) -> bool:
        stack = node.meta.get("nn_module_stack", {})
        names = []
        for _path, stacked_type in stack.values():
            if isinstance(stacked_type, type):
                names.append(stacked_type.__module__ + "." + stacked_type.__qualname__)
            else:
                names.append(str(stacked_type))
        return type_name in names

    return module_filter


def _annotation(node: Node) -> QuantizationAnnotation | None:
    annotation = node.meta.get(Q_ANNOTATION_KEY)
    if isinstance(annotation, QuantizationAnnotation):
        return annotation
    return None


def _is_annotated(node: Node) -> bool:
    annotation = _annotation(node)
    return annotation is not None and annotation._annotated


def _arg(node: Node, index: int, name: str):
    if index < len(node.args):
        return node.args[index]
    return node.kwargs.get(name)


def _is_relu(node: Node) -> bool:
    return node.op == "call_function" and node.target in _RELU_TARGETS


def _weight_axis(node: Node) -> int:
    if node.target in _TRANSPOSE_TARGETS:
        return 1
    if node.target is torch.ops.aten.convolution.default and bool(_arg(node, 6, "transposed")):
        return 1
    return 0


def _matches_operator(node: Node, operator: object) -> bool:
    target = node.target
    packet = getattr(target, "overloadpacket", None)
    aliases = _OPERATOR_ALIASES.get(operator, set())
    return target is operator or packet is operator or target in aliases or packet in aliases


def _float_placeholder(node: Node) -> bool:
    if node.op != "placeholder":
        return False
    value = node.meta.get("val", None)
    if isinstance(value, torch.Tensor):
        return value.is_floating_point() and value.ndim > 0
    return True


def _recording_observer_cls(store: CalibrationStore, input_name: str):
    from torchao.quantization.pt2e import MinMaxObserver

    class RecordingMinMaxObserver(MinMaxObserver):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            store.add(input_name, x)
            return super().forward(x)

    RecordingMinMaxObserver.__name__ = "RecordingMinMaxObserver"
    RecordingMinMaxObserver.__qualname__ = "RecordingMinMaxObserver"
    return RecordingMinMaxObserver


class HailoQuantizer(Quantizer):
    """Hailo quantizer passed to PT2E ``prepare_pt2e`` and ``convert_pt2e``.

    ``set_global``, ``set_module_name``, ``set_module_type``, and
    ``set_object_type`` choose which ops are observed as int8. One instance
    corresponds to one model and one ``prepare_pt2e`` call. Graph inputs are
    stored in ``calibration_store`` during the calibration forward, and
    ``lower_to_hailo`` sends those samples to the Dataflow Compiler.
    """

    def __init__(self, hw_arch: str = "hailo8", max_calibration_samples: int = 1024) -> None:
        super().__init__()
        if not hw_arch:
            raise ValueError("hw_arch is empty.")
        self.hw_arch = hw_arch
        self.calibration_store = CalibrationStore(max_calibration_samples)
        self.float_model: torch.fx.GraphModule | None = None
        self.global_config: QuantizationConfig | None = None
        self.module_name_config: dict[str, QuantizationConfig] = {}
        self.module_type_config: dict[Callable, QuantizationConfig] = {}
        self.operator_type_config: dict[object, QuantizationConfig] = {}
        self._recording_specs: dict[str, QuantizationSpec] = {}

    def set_global(self, quantization_config: QuantizationConfig) -> HailoQuantizer:
        self._reject_qat(quantization_config)
        self.global_config = quantization_config
        return self

    def set_module_name(
        self, module_name: str, quantization_config: QuantizationConfig
    ) -> HailoQuantizer:
        self._reject_qat(quantization_config)
        self.module_name_config[module_name] = quantization_config
        return self

    def set_module_type(
        self, module_type: Callable, quantization_config: QuantizationConfig
    ) -> HailoQuantizer:
        self._reject_qat(quantization_config)
        self.module_type_config[module_type] = quantization_config
        return self

    def set_operator_type(
        self, operator_type: object, quantization_config: QuantizationConfig
    ) -> HailoQuantizer:
        self._reject_qat(quantization_config)
        self.operator_type_config[operator_type] = quantization_config
        return self

    def set_object_type(
        self, object_type: object, quantization_config: QuantizationConfig
    ) -> HailoQuantizer:
        """Dispatch a module type to ``set_module_type`` and anything else to ``set_operator_type``."""

        if isinstance(object_type, type) and issubclass(object_type, torch.nn.Module):
            return self.set_module_type(object_type, quantization_config)
        return self.set_operator_type(object_type, quantization_config)

    def reset_calibration(self) -> None:
        self.calibration_store.clear()

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        if self.float_model is None:
            try:
                self.float_model = copy.deepcopy(model)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to copy the FP32 graph used for Hailo lowering. "
                    "Pass the module produced by torch.export to prepare_pt2e."
                ) from exc
            self.float_model.eval()

        for module_name, config in self.module_name_config.items():
            self._annotate_config(model, config, get_module_name_filter(module_name))
        for module_type, config in self.module_type_config.items():
            self._annotate_config(model, config, _module_type_filter(module_type))
        for operator_type, config in self.operator_type_config.items():
            self._annotate_config(
                model,
                config,
                lambda node, op=operator_type: _matches_operator(node, op),
            )
        if self.global_config is not None:
            self._annotate_config(model, self.global_config, lambda _node: True)
        self._observe_unannotated_inputs(model)
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        if self._activation_config() is None:
            raise ValueError(
                "HailoQuantizer has no quantization config. "
                "Call set_global(get_hailo_quantization_config()) first."
            )
        for node in model.graph.nodes:
            annotation = _annotation(node)
            if annotation is None:
                continue
            specs = list(annotation.input_qspec_map.values())
            if annotation.output_qspec is not None:
                specs.append(annotation.output_qspec)
            for spec in specs:
                if getattr(spec, "is_dynamic", False):
                    raise ValueError(
                        "HailoQuantizer supports static PTQ only. "
                        "Dynamic activation quantization is not available."
                    )

    def prepare_obs_or_fq_callback(
        self,
        model: torch.fx.GraphModule,
        edge_or_node_to_obs_or_fq: dict,
    ) -> None:
        for edge_or_node, observer in edge_or_node_to_obs_or_fq.items():
            if not isinstance(edge_or_node, tuple) or len(edge_or_node) != 2:
                continue
            producer = edge_or_node[0]
            if not isinstance(producer, Node) or producer.op != "placeholder":
                continue
            self._patch_input_observer(observer, producer.name)

    def _reject_qat(self, quantization_config: QuantizationConfig) -> None:
        if quantization_config.is_qat:
            raise ValueError(
                "HailoQuantizer does not support PT2E QAT. "
                "Use static PTQ from get_hailo_quantization_config()."
            )

    def _activation_config(self) -> QuantizationConfig | None:
        if self.global_config is not None:
            return self.global_config
        for config in (
            *self.module_name_config.values(),
            *self.module_type_config.values(),
            *self.operator_type_config.values(),
        ):
            if config.input_activation is not None:
                return config
        return None

    def _annotate_config(
        self,
        model: torch.fx.GraphModule,
        config: QuantizationConfig,
        predicate: Callable[[Node], bool],
    ) -> None:
        if config.input_activation is None or config.weight is None or config.output_activation is None:
            return
        for node in model.graph.nodes:
            if node.op != "call_function" or node.target not in _WEIGHTED_TARGETS:
                continue
            if not predicate(node) or _is_annotated(node):
                continue
            self._annotate_weighted(node, config)

    def _annotate_weighted(self, node: Node, config: QuantizationConfig) -> None:
        activation = _arg(node, 0, "input")
        weight = _arg(node, 1, "weight")
        bias = _arg(node, 2, "bias")
        if not isinstance(activation, Node) or not isinstance(weight, Node):
            return

        tail = node
        users = list(node.users)
        if len(users) == 1 and _is_relu(users[0]) and not _is_annotated(users[0]):
            tail = users[0]

        if activation.op == "placeholder" and _float_placeholder(activation):
            activation_spec: QuantizationSpec = self._recording_spec(
                config.input_activation, activation.name
            )
        else:
            activation_spec = config.input_activation
        input_qspec_map: dict[Node, object] = {
            activation: activation_spec,
            weight: self._weight_spec(config.weight, _weight_axis(node)),
        }
        if isinstance(bias, Node):
            per_channel = config.weight.qscheme == torch.per_channel_symmetric
            input_qspec_map[bias] = DerivedQuantizationSpec(
                derived_from=[(activation, node), (weight, node)],
                derive_qparams_fn=_derive_bias_qparams,
                dtype=torch.int32,
                quant_min=-(2**31),
                quant_max=2**31 - 1,
                qscheme=(
                    torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric
                ),
                ch_axis=0 if per_channel else None,
            )
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=None if tail is not node else config.output_activation,
            _annotated=True,
        )
        if tail is not node:
            tail.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                output_qspec=config.output_activation,
                _annotated=True,
            )

    def _observe_unannotated_inputs(self, model: torch.fx.GraphModule) -> None:
        config = self._activation_config()
        if config is None or config.input_activation is None:
            return
        observed: set[Node] = set()
        for node in model.graph.nodes:
            annotation = _annotation(node)
            if annotation is None:
                continue
            observed.update(annotation.input_qspec_map)

        for node in list(model.graph.nodes):
            if not _float_placeholder(node) or node in observed:
                continue
            spec = self._recording_spec(config.input_activation, node.name)
            for user in list(node.users):
                if user.op != "call_function":
                    continue
                annotation = _annotation(user)
                if annotation is None:
                    user.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                        input_qspec_map={node: spec},
                        allow_implicit_sharing=False,
                        _annotated=True,
                    )
                elif node not in annotation.input_qspec_map:
                    annotation.input_qspec_map[node] = spec
                    annotation.allow_implicit_sharing = False

    def _recording_spec(self, base: QuantizationSpec, input_name: str) -> QuantizationSpec:
        cached = self._recording_specs.get(input_name)
        if cached is not None:
            return cached
        observer_cls = _recording_observer_cls(self.calibration_store, input_name)
        spec = QuantizationSpec(
            dtype=base.dtype,
            quant_min=base.quant_min,
            quant_max=base.quant_max,
            qscheme=base.qscheme,
            ch_axis=base.ch_axis,
            is_dynamic=base.is_dynamic,
            observer_or_fake_quant_ctr=observer_cls.with_args(eps=2**-12),
        )
        self._recording_specs[input_name] = spec
        return spec

    def _weight_spec(self, base: QuantizationSpec, ch_axis: int) -> QuantizationSpec:
        if base.ch_axis == ch_axis:
            return base
        return QuantizationSpec(
            dtype=base.dtype,
            quant_min=base.quant_min,
            quant_max=base.quant_max,
            qscheme=base.qscheme,
            ch_axis=ch_axis,
            is_dynamic=base.is_dynamic,
            observer_or_fake_quant_ctr=base.observer_or_fake_quant_ctr,
        )

    def _patch_input_observer(self, observer: torch.nn.Module, input_name: str) -> None:
        if getattr(observer, "_hailo_records_inputs", False):
            return
        original = observer.forward
        store = self.calibration_store

        def forward(x: torch.Tensor) -> torch.Tensor:
            store.add(input_name, x)
            return original(x)

        observer.forward = forward
        observer._hailo_records_inputs = True
