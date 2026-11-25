"""Collect model inputs during calibration for Hailo DFC ``optimize``."""

from __future__ import annotations

import weakref

import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor


class CalibrationStore:
    """Stack floating-point inputs seen by the prepared model's forward pass."""

    def __init__(self, max_samples: int = 1024) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1.")
        self.max_samples = max_samples
        self.samples: dict[str, list[Tensor]] = {}
        self._last_tensor: dict[str, weakref.ReferenceType[Tensor]] = {}

    def clear(self) -> None:
        self.samples.clear()
        self._last_tensor.clear()

    def num_samples(self, name: str | None = None) -> int:
        names = (name,) if name is not None else tuple(self.samples)
        total = 0
        for key in names:
            for part in self.samples.get(key, ()):
                total += int(part.shape[0])
        return total

    def add(self, name: str, tensor: Tensor) -> None:
        if not isinstance(tensor, Tensor) or isinstance(tensor, FakeTensor):
            return
        if tensor.is_meta or not tensor.is_floating_point() or tensor.ndim == 0:
            return
        previous = self._last_tensor.get(name)
        if previous is not None and previous() is tensor:
            return
        self._last_tensor[name] = weakref.ref(tensor)
        remaining = self.max_samples - self.num_samples(name)
        if remaining <= 0:
            return
        cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if int(cpu.shape[0]) > remaining:
            cpu = cpu[:remaining].contiguous()
        self.samples.setdefault(name, []).append(cpu)

    def arrays(self) -> dict[str, "object"]:
        import numpy as np

        arrays: dict[str, np.ndarray] = {}
        for name, parts in self.samples.items():
            if not parts:
                continue
            tail = tuple(parts[0].shape[1:])
            for part in parts[1:]:
                if tuple(part.shape[1:]) != tail:
                    raise ValueError(
                        f"Calibration tensors for input {name!r} have different shapes. "
                        "Run forward with the same static shape used for prepare."
                    )
            arrays[name] = np.ascontiguousarray(torch.cat(parts, dim=0).numpy())
        return arrays
