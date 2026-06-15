"""
Vector applier package.

Keep this __init__ lightweight to avoid importing optional dependencies during
configuration-only workflows (e.g. hyperparameter loading).

Import applier implementations from their submodules when needed.
"""

__all__ = []