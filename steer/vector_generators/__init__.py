"""
Vector generator package.

Keep this __init__ lightweight to avoid importing optional dependencies during
configuration-only workflows (e.g. hyperparameter loading).

Import generator implementations from their submodules when needed.
"""

__all__ = []