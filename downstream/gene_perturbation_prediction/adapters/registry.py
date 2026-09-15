from .native_gears import NativeGEARSAdapter
from .random_adapter import RandomAdapter
from .sckite_adapter import ScKITEAdapter


ADAPTER_REGISTRY = {
    "native_gears": NativeGEARSAdapter,
    "random": RandomAdapter,
    "sckite": ScKITEAdapter,
}


def register_adapter(name, adapter_cls):
    if name in ADAPTER_REGISTRY:
        raise ValueError(f"Adapter '{name}' is already registered.")
    ADAPTER_REGISTRY[name] = adapter_cls


def build_adapter(name="native_gears", **kwargs):
    if name not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter: {name}. Available adapters: {list(ADAPTER_REGISTRY.keys())}")
    adapter_cls = ADAPTER_REGISTRY[name]
    return adapter_cls(**kwargs)
