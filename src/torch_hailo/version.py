"""Warn when the installed DFC stack does not match the tested pin set."""

from __future__ import annotations

import warnings


_EXPECTED = {
    "hailo_sdk_client": (5, 5),
    "torch": (2, 9),
    "torchao": (0, 17),
}


def _major_minor(version: str) -> tuple[int, int] | None:
    parts = str(version).split("+", 1)[0].split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def warn_if_untested_versions() -> list[str]:
    """Return and emit warnings for packages outside the tested major.minor pins."""

    messages: list[str] = []
    for name, expected in _EXPECTED.items():
        try:
            module = __import__(name)
        except ImportError:
            continue
        raw = getattr(module, "__version__", None)
        if raw is None:
            continue
        actual = _major_minor(str(raw))
        if actual is None or actual != expected:
            message = (
                f"{name} {raw} is installed; torch-hailo was tested with "
                f"{name} {expected[0]}.{expected[1]}.x."
            )
            messages.append(message)
            warnings.warn(message, UserWarning, stacklevel=2)
    return messages
