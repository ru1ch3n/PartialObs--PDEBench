"""Built-in baselines and extension API."""

from .base import (
    METHOD_REGISTRY,
    Method,
    MethodCapabilities,
    available_methods,
    capabilities_for,
    create_method,
    discover_methods,
    method_discovery_errors,
    register_method,
)
from .interpolation import (
    BilinearInterpolation,
    MeanFill,
    NearestInterpolation,
    Persistence,
    RBFInterpolation,
    ZeroFill,
)

# Neural dependencies are optional at import time. neural.py provides friendly
# construction errors when PyTorch is unavailable.
from .neural import (  # noqa: E402
    AutoregressiveModel,
    CompactCNO2d,
    CompactFNO2d,
    CompactResidualEncoder,
    ConvLSTM,
    MaskChannelUNet,
    MaskedAutoencoderSmall,
    create_autoregressive_fno,
    create_model,
)

BUILTIN_METHODS = {
    "zero": ZeroFill,
    "mean": MeanFill,
    "nearest": NearestInterpolation,
    "bilinear": BilinearInterpolation,
    "rbf": RBFInterpolation,
    "persistence": Persistence,
    "unet": MaskChannelUNet,
    "fno": CompactFNO2d,
    "cno": CompactCNO2d,
    "convlstm": ConvLSTM,
    "autoregressive": AutoregressiveModel,
    "autoregressive_fno": create_autoregressive_fno,
    "residual_cnn": CompactResidualEncoder,
    "mae_small": MaskedAutoencoderSmall,
}


def install_builtin_methods(registry=None) -> tuple[str, ...]:
    """Mirror built-ins into the project-wide registry when it is available.

    The method package retains its small standalone registry so external code
    can import it independently. CLI startup may call this helper to expose the
    same factories through :mod:`pdeobs.registry`.
    """

    if registry is None:
        try:
            from ..registry import METHOD_REGISTRY as registry
        except (ImportError, AttributeError):
            return ()
    installed = []
    for name, factory in BUILTIN_METHODS.items():
        if name not in registry:
            registry.register(name, obj=factory)
            installed.append(name)
    return tuple(installed)


# Importing pdeobs.methods is an explicit request for method functionality, so
# synchronizing the central registry here does not burden lightweight imports.
install_builtin_methods()

__all__ = [
    "METHOD_REGISTRY",
    "Method",
    "MethodCapabilities",
    "available_methods",
    "capabilities_for",
    "create_method",
    "discover_methods",
    "method_discovery_errors",
    "register_method",
    "ZeroFill",
    "MeanFill",
    "NearestInterpolation",
    "BilinearInterpolation",
    "RBFInterpolation",
    "Persistence",
    "MaskChannelUNet",
    "CompactFNO2d",
    "CompactCNO2d",
    "CompactResidualEncoder",
    "ConvLSTM",
    "AutoregressiveModel",
    "MaskedAutoencoderSmall",
    "create_model",
    "create_autoregressive_fno",
    "BUILTIN_METHODS",
    "install_builtin_methods",
]
