"""One-call export, static int8 PTQ, and Hailo lowering."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from torch_hailo.lower import HailoCompiler, HailoModel, lower_to_hailo
from torch_hailo.qconfig import get_hailo_quantization_config
from torch_hailo.quantizer import HailoQuantizer


def compile(
    model: nn.Module,
    example_inputs: torch.Tensor | Sequence[torch.Tensor],
    calibration_inputs: Iterable[Any],
    *,
    hw_arch: str = "hailo10h",
    quantizer: HailoQuantizer | None = None,
    output_path: str | os.PathLike | None = None,
    model_script: str | None = None,
    net_name: str = "model",
    allow_unsupported_ops: bool = False,
    compiler: HailoCompiler | None = None,
) -> HailoModel:
    """Export, calibrate, convert, and lower a model to a Hailo HEF.

    ``calibration_inputs`` yields either a tensor, a tuple of tensors matching
    ``example_inputs``, or a mapping keyed by input name. Calibration runs under
    ``torch.no_grad()``.
    """

    inputs = _as_tuple(example_inputs)
    exported = torch.export.export(model.eval(), inputs).module()
    active = quantizer
    if active is None:
        active = HailoQuantizer(hw_arch=hw_arch).set_global(get_hailo_quantization_config())
    elif active.global_config is None and not (
        active.module_name_config or active.module_type_config or active.operator_type_config
    ):
        active.set_global(get_hailo_quantization_config())
    prepared = prepare_pt2e(exported, active)
    sample_count = 0
    with torch.no_grad():
        for batch in calibration_inputs:
            prepared(*_batch_as_args(batch, inputs))
            sample_count += 1
    quantized = convert_pt2e(prepared)
    hailo_model = lower_to_hailo(
        quantized,
        inputs,
        active,
        hw_arch=hw_arch or active.hw_arch,
        output_path=output_path,
        model_script=model_script,
        net_name=net_name,
        allow_unsupported_ops=allow_unsupported_ops,
        compiler=compiler,
    )
    hailo_model.reference_model = quantized
    hailo_model.diagnostics = _diagnostics(sample_count, active)
    return hailo_model


def _diagnostics(sample_count: int, quantizer: HailoQuantizer) -> dict[str, Any]:
    recorded = quantizer.calibration_store.num_samples()
    notes: list[str] = []
    if recorded < 1024:
        notes.append(
            f"Calibration recorded {recorded} samples (forward batches: {sample_count}). "
            "The Hailo DFC may reduce the optimization level below the recommended set size of 1024."
        )
    return {
        "calibration_samples": recorded,
        "calibration_batches": sample_count,
        "notes": notes,
    }


def _as_tuple(example_inputs: torch.Tensor | Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    if isinstance(example_inputs, torch.Tensor):
        return (example_inputs,)
    return tuple(example_inputs)


def _batch_as_args(
    batch: Any,
    example: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if isinstance(batch, torch.Tensor):
        return (batch,)
    if isinstance(batch, Mapping):
        if len(batch) != len(example):
            raise ValueError(
                f"Calibration mapping has {len(batch)} entries, expected {len(example)}."
            )
        return tuple(batch[key] for key in batch)
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        return tuple(batch)
    raise TypeError(
        "Each calibration item must be a tensor, a sequence of tensors, or a mapping."
    )
