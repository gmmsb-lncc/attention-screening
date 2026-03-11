"""DGL graphbolt compatibility shim for GraphBAN.

DGL >= 2.0 requires a C++ graphbolt library compiled for the exact PyTorch
version. When running with a PyTorch version that doesn't have a pre-built
graphbolt binary, DGL import fails.

GraphBAN does NOT use graphbolt features (it only uses basic DGL graphs and
GCN from dgllife). This shim stubs out graphbolt before DGL loads, allowing
the rest of DGL to work normally.

Usage:
    import dgl_compat  # must be first import
    import dgl
    from dgllife.utils import smiles_to_bigraph
"""

import sys
import types


def patch_setuptools():
    """Ensure setuptools.extern is importable (fixes broken setuptools installs)."""
    try:
        import setuptools.extern  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        import setuptools
        if not hasattr(setuptools, 'extern'):
            stub = types.ModuleType('setuptools.extern')
            stub.__path__ = []
            stub.__file__ = 'stub'
            sys.modules['setuptools.extern'] = stub


def patch_torchdata_datapipes():
    """Stub torchdata.datapipes for newer torchdata (>=0.8) that removed it.

    DGL 2.x graphbolt imports ``from torchdata.datapipes.iter import
    IterDataPipe``. Newer torchdata ships without datapipes. Providing a
    lightweight stub prevents the ImportError.
    """
    try:
        from torchdata.datapipes.iter import IterDataPipe  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        pass

    class _FakeIterDataPipe:
        pass

    for mod_name in ["torchdata.datapipes", "torchdata.datapipes.iter",
                     "torchdata.datapipes.map"]:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = []
            stub.__file__ = "stub"
            stub.IterDataPipe = _FakeIterDataPipe
            sys.modules[mod_name] = stub


def patch_graphbolt():
    """Stub out dgl.graphbolt to avoid C++ library load failure."""
    if "dgl" in sys.modules:
        return

    for mod_name in [
        "dgl.graphbolt",
        "dgl.graphbolt.impl",
        "dgl.graphbolt.base",
        "dgl.graphbolt.dataloader",
    ]:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = []
            stub.__file__ = "stub"
            sys.modules[mod_name] = stub


# Auto-apply on import
patch_setuptools()
patch_torchdata_datapipes()
patch_graphbolt()
