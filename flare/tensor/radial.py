"""Differentiable radial functions using the FLARE C++ B2 convention."""

import torch


def quadratic_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Return ``(cutoff - r)**2`` inside the cutoff and zero outside."""
    return (cutoff - distances).clamp_min(0).square()


def chebyshev_radial(
    distances: torch.Tensor, n_radial: int, cutoff: float
) -> torch.Tensor:
    """Evaluate ``T_n(r / cutoff) * (cutoff - r)**2`` for ``n=0..N-1``.

    FLARE evaluates Chebyshev polynomials on [0, 1], without remapping to
    [-1, 1]. The recurrence avoids the endpoint singularities introduced by
    differentiating an ``acos`` implementation.
    """
    if n_radial < 1:
        raise ValueError("n_radial must be positive")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    x = distances / cutoff
    values = [torch.ones_like(x)]
    if n_radial > 1:
        values.append(x)
    for _ in range(2, n_radial):
        values.append(2 * x * values[-1] - values[-2])
    return torch.stack(values, dim=-1) * quadratic_cutoff(
        distances, cutoff
    ).unsqueeze(-1)
