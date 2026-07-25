"""FormatAdapter interface + registry. Each format implements `parse()` -> the
same canonical `Document`; downstream never learns the format. A new format = one
more registered adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.canonical.model import Document, SourceFormat
from src.ingest.resolver import ResolvedDoc


class FormatAdapter(ABC):
    """Contract: ResolvedDoc -> canonical Document."""

    source_format: SourceFormat

    @abstractmethod
    def parse(self, resolved: ResolvedDoc) -> Document:
        ...


_REGISTRY: dict[SourceFormat, type[FormatAdapter]] = {}


def register(fmt: SourceFormat):
    """Decorator to register an adapter for a source format."""
    def _wrap(cls: type[FormatAdapter]) -> type[FormatAdapter]:
        _REGISTRY[fmt] = cls
        cls.source_format = fmt
        return cls
    return _wrap


def get_adapter(fmt: SourceFormat) -> FormatAdapter:
    """Factory — pick the adapter for a resolved document's format."""
    if fmt not in _REGISTRY:
        raise ValueError(f"no adapter registered for format: {fmt}")
    return _REGISTRY[fmt]()
