"""
LM-steer vector generator subpackage.

Kept lightweight, but we DO re-export the small helper classes
`Hack_no_grad` / `Projected_Adaptor` because they are imported at the
package level across the codebase, e.g.:

    from ...vector_generators.lm_steer import Hack_no_grad

These helpers only depend on torch, so re-exporting them is safe and does
not pull any optional/heavy dependencies.
"""

from .lm_steer_helper import Hack_no_grad, Projected_Adaptor

__all__ = ["Hack_no_grad", "Projected_Adaptor"]
