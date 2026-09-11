from .gears import GEARS
from .pertdata import PertData

try:
    from adapters.base import BaseSCFMAdapter
    from adapters.registry import build_adapter, register_adapter
except Exception:
    BaseSCFMAdapter = None
    build_adapter = None
    register_adapter = None

__all__ = [
    "GEARS",
    "PertData",
    "BaseSCFMAdapter",
    "build_adapter",
    "register_adapter",
]
