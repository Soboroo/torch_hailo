"""PT2E QuantizationConfig aligned with the Hailo reference PTQ."""

from __future__ import annotations

import torch
from torchao.quantization.pt2e import MinMaxObserver, PerChannelMinMaxObserver
from torchao.quantization.pt2e.quantizer import QuantizationConfig, QuantizationSpec

# Hailo defaults to 8-bit activations and 8-bit weights. Activations are
# asymmetric (per-tensor). Weights are symmetric, with a scale per output channel.
_OBSERVER_EPS = 2**-12


def get_hailo_quantization_config(
    *,
    is_per_channel: bool = True,
) -> QuantizationConfig:
    """Build a config for static PTQ.

    Activations are ``int8`` per-tensor affine and weights are ``int8`` symmetric.
    When ``is_per_channel`` is true, each output channel has its own weight scale.
    This config is used by the ``convert_pt2e`` reference model. The Hailo DFC
    does not import these scales and re-quantizes from the inputs captured at
    lowering time.
    """

    activation = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=_OBSERVER_EPS),
    )
    if is_per_channel:
        weight_observer = PerChannelMinMaxObserver.with_args(eps=_OBSERVER_EPS)
        weight_qscheme = torch.per_channel_symmetric
    else:
        weight_observer = MinMaxObserver.with_args(eps=_OBSERVER_EPS)
        weight_qscheme = torch.per_tensor_symmetric
    weight = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=weight_qscheme,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer,
    )
    return QuantizationConfig(
        input_activation=activation,
        output_activation=activation,
        weight=weight,
        bias=None,
        is_qat=False,
    )
