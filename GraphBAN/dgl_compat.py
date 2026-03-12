"""DGL graphbolt compatibility shim for GraphBAN.

DGL >= 2.0 requires a C++ graphbolt library compiled for the exact PyTorch
version. When running with a PyTorch version that doesn't have a pre-built
graphbolt binary, DGL import fails and unconditionally calls sys.exit(1) with:

    "Stopping RUNTIME. Colaboratory will restart automatically. Please run again."

This shim prevents that crash through three layers of protection:

  1. Sets DGL_GRAPHBOLT_DISABLE=1 env var (respected by some DGL builds).
  2. Pre-stubs dgl.graphbolt.* in sys.modules so DGL's Python __init__.py
     finds empty stubs instead of running the graphbolt load path.
  3. Eagerly imports DGL while the stubs are active AND temporarily intercepts
     sys.exit so any residual graphbolt crash is converted to ImportError
     instead of killing the process.

GraphBAN does NOT use graphbolt — it only needs basic DGL graph ops and
dgllife GCN. Stubbing graphbolt is safe.

Usage:
    import dgl_compat  # must be first import, before any dgl/dgllife code
    import dgl
    from dgllife.utils import smiles_to_bigraph
"""

import os
import sys
import types
import warnings

# Layer 1: env var (respected by newer DGL builds).
os.environ.setdefault("DGL_GRAPHBOLT_DISABLE", "1")


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
    """Stub out dgl.graphbolt.* to avoid C++ library load failure.

    Pre-populates sys.modules with empty stubs so that DGL's __init__.py
    finds them already resolved and skips the real graphbolt loader.
    """
    if "dgl" in sys.modules:
        return

    # Cover all submodules that DGL 2.x graphbolt/__init__.py may try to load.
    for mod_name in [
        "dgl.graphbolt",
        "dgl.graphbolt.impl",
        "dgl.graphbolt.base",
        "dgl.graphbolt.dataloader",
        "dgl.graphbolt.feature_store",
        "dgl.graphbolt.item_set",
        "dgl.graphbolt.graph_storage",
        "dgl.graphbolt.minibatch",
        "dgl.graphbolt.sampling_graph",
    ]:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = []
            stub.__file__ = "stub"
            sys.modules[mod_name] = stub


def eager_import_dgl():
    """Pre-import DGL while graphbolt stubs are active.

    DGL is NOT imported at run_baseline.py module level — it is only imported
    later when setup_graphban_imports() loads GraphBAN's models.py.  By that
    point sys.exit interception is no longer in effect.  Importing DGL here
    (while stubs are fresh and sys.exit is intercepted) ensures it is cached
    in sys.modules before GraphBAN's code runs, so GraphBAN's `import dgl`
    just retrieves the already-loaded module.

    Also intercepts sys.exit(1) as a belt-and-suspenders measure: if graphbolt
    loading somehow reaches the sys.exit(1) path despite all stubs, we convert
    it to ImportError instead of killing the process.
    """
    if "dgl" in sys.modules:
        return

    _orig_exit = sys.exit

    def _intercept(code=0):
        if code != 0:
            raise ImportError(
                f"DGL initialization called sys.exit({code}). "
                "Likely graphbolt C extension version mismatch — "
                "suppressed by dgl_compat. GraphBAN will continue."
            )
        _orig_exit(code)

def eager_import_dgl():
    """Pre-import DGL while graphbolt stubs are active.

    DGL is NOT imported at run_baseline.py module level — it is only imported
    later when setup_graphban_imports() loads GraphBAN's models.py.  By that
    point sys.modules stubs are already in place from patch_graphbolt(), so
    DGL will pick them up.  This function pre-triggers that import while stubs
    are fresh and DGL_GRAPHBOLT_DISABLE is already set.

    NOTE: On some DGL conda builds, graphbolt loading happens inside a compiled
    C extension (_graphbolt.so) that calls libc exit() directly, making Python
    sys.exit monkey-patching useless. For those builds the only reliable fix is
    to rename the .so so it can never be dlopen()'d — this must be done once on
    the server before running:

        find ~/miniconda/envs/graphban/ -path "*/dgl/graphbolt/*.so" \\
            -exec mv {} {}.disabled \\;
    """
    if "dgl" in sys.modules:
        return

    try:
        import dgl  # noqa: F401
    except Exception as exc:
        warnings.warn(
            f"dgl_compat: DGL pre-import failed: {exc}. "
            "GraphBAN may fail at runtime.",
            stacklevel=2,
        )


# Auto-apply on import
patch_setuptools()
patch_torchdata_datapipes()
patch_graphbolt()
eager_import_dgl()
