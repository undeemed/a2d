"""Generic name -> object registry: the Python analog of the Rust ``inventory`` seam.

Extension points (ingest / transform / objectives / eval / ...) each create a
``Registry`` and expose its ``register`` decorator so new implementations
self-register on import instead of editing a central switch (SPEC-HANDOFF 3.3).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self.kind}: {name!r} already registered")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"{self.kind}: unknown {name!r}; registered: {self.names()}") from None

    def names(self) -> list[str]:
        return sorted(self._items)
