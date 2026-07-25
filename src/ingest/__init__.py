"""Ingestion package.

Importing this package registers all FormatAdapters with the registry in
`base.py` (each adapter module calls @register on import), so `get_adapter(fmt)`
resolves without callers importing each adapter explicitly.
"""
from src.ingest import dwg, pdf_native, pdf_scanned  # noqa: F401  (side effect: register)
