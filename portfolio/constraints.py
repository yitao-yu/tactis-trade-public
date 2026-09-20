"""
Portfolio allocation constraints.

All constraints are applied as a post-processing projection that each
allocator runs on its raw output before returning final weights.
"""

from dataclasses import dataclass


@dataclass
class PortfolioConstraints:
    """
    Structural constraints applied to portfolio weights after construction.

    Parameters
    ----------
    max_assets : int | None
        Maximum number of non-zero positions (cardinality constraint).
    max_weight : float | None
        Per-asset position cap ``|w_i| ≤ max_weight``.
    allow_short : bool
        If ``False``, negative weights are zeroed before final normalisation.
    max_gross_exposure : float | None
        Cap on total gross exposure ``Σ|w_i|``.
        Only meaningful when ``allow_short=True``.
    dollar_neutral : bool
        If ``True``, skip cardinality, gross-exposure cap, and normalisation.
        Dollar neutrality must be enforced in the allocator itself.
        ```allow_short``` must be ``True``.
    beta_neutral : bool
        If ``True``, the allocator should enforce ``βᵀw = 0``.
        Enforcement is done inside the allocator; the constraint pipeline
        does not touch betas.
    position_limit : float | None
        Hard cap ``|w_i| ≤ position_limit`` (applied in step 3).
        Overrides ```max_weight``` for dollar-neutral allocators.
    """

    max_assets: int | None = None
    max_weight: float | None = None
    allow_short: bool = False
    max_gross_exposure: float | None = None
    dollar_neutral: bool = False
    beta_neutral: bool = False
    position_limit: float | None = None
