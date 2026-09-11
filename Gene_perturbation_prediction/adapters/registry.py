from .native_gears import NativeGEARSAdapter  # 导入原始 GEARS adapter。 #
from .stage2_adapter import Stage2Adapter  # 导入 Stage2 adapter。 #


ADAPTER_REGISTRY = {  # 定义 adapter 注册表。 #
    "native_gears": NativeGEARSAdapter,  # 注册原始 GEARS adapter。 #
    "stage2": Stage2Adapter,  # 注册 Stage2 adapter。 #
}  # 注册表结束。 #


def register_adapter(name, adapter_cls):  # 定义动态注册 adapter 的函数。 #
    if name in ADAPTER_REGISTRY:  # 如果名称已存在。 #
        raise ValueError(f"Adapter '{name}' is already registered.")  # 抛出重复注册错误。 #
    ADAPTER_REGISTRY[name] = adapter_cls  # 注册 adapter 类。 #


def build_adapter(name="native_gears", **kwargs):  # 根据 adapter 名称构建 adapter。 #
    if name not in ADAPTER_REGISTRY:  # 如果 adapter 名称不存在。 #
        raise ValueError(f"Unknown adapter: {name}. Available adapters: {list(ADAPTER_REGISTRY.keys())}")  # 抛出未知 adapter 错误。 #
    adapter_cls = ADAPTER_REGISTRY[name]  # 获取 adapter 类。 #
    return adapter_cls(**kwargs)  # 实例化并返回 adapter。 #
