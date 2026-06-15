"""
Lightweight package init.

Some submodules (e.g. datasets) rely on optional third-party dependencies.
To keep `import steer` usable in minimal environments (e.g. config loading),
we guard star-imports with best-effort fallbacks.
"""

def _try_star_import(module_name: str):
    try:
        module = __import__(module_name, globals(), locals(), ["*"])
        globals().update({k: getattr(module, k) for k in getattr(module, "__all__", [])})
        # If __all__ is not defined, do nothing (avoid polluting namespace unpredictably).
    except Exception:
        # Optional dependency missing or import error; keep package importable.
        pass


_try_star_import("steer.datasets")
_try_star_import("steer.evaluate")
_try_star_import("steer.models")
_try_star_import("steer.utils")
_try_star_import("steer.vector_generators")
_try_star_import("steer.vector_appliers")