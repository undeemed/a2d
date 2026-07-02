from __future__ import annotations

from collections.abc import Callable

import pytest
from a2d_core.registry import Registry


def test_register_get_round_trip() -> None:
    reg: Registry[Callable[[], str]] = Registry("thing")

    @reg.register("alpha")
    def alpha() -> str:
        return "a"

    assert reg.get("alpha") is alpha
    assert reg.names() == ["alpha"]


def test_duplicate_raises() -> None:
    reg: Registry[str] = Registry("thing")
    reg.register("dup")("first")
    with pytest.raises(ValueError):
        reg.register("dup")("second")


def test_unknown_raises() -> None:
    reg: Registry[str] = Registry("thing")
    with pytest.raises(KeyError):
        reg.get("missing")
