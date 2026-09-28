"""Local-environment covariance functions on raw descriptor tensors."""

import math

import torch


def _normalization_data(values, empty_threshold):
    # Decide emptiness without squaring raw values: their norm can underflow
    # in float32 close to a cutoff. The detached scale cancels analytically
    # from normalization and avoids differentiating a nonsmooth max.
    scale = values.detach().abs().amax(dim=-1, keepdim=True)
    nonzero_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    scaled_norm = torch.linalg.vector_norm(
        values.detach() / nonzero_scale, dim=-1, keepdim=True
    )
    nonzero_norm = torch.where(
        scaled_norm > 0, scaled_norm, torch.ones_like(scaled_norm)
    )
    active = (scale > 0) & (scale >= empty_threshold / nonzero_norm)
    safe_scale = torch.where(active, scale, torch.ones_like(scale))
    scaled = torch.where(active, values, torch.zeros_like(values)) / safe_scale
    norm_squared = scaled.square().sum(dim=-1, keepdim=True)
    denominator = torch.where(active, norm_squared, torch.ones_like(norm_squared)).sqrt()
    return scaled / denominator, active.squeeze(-1), safe_scale * denominator


def _normalize_descriptors(values, empty_threshold):
    return _normalization_data(values, empty_threshold)[0]


def normalized_dot_product(
    left,
    right,
    left_species,
    right_species,
    amplitude=1.0,
    power=2,
    empty_threshold=1e-8,
):
    """Return ``amplitude**2 * (qhat_left @ qhat_right.T)**power``.

    Cross-central-species covariances and environments with raw descriptor
    norm below ``empty_threshold`` are zero, matching FLARE's legacy kernel.
    ``power`` is a positive integer; ``amplitude`` is a scalar or scalar tensor
    (and may carry gradients). Input descriptors must share a dtype/device.
    Descriptors and amplitude must be finite; invalid descriptors are not
    treated as empty environments.

    The empty threshold is a discontinuous model convention. Derivatives are
    only meaningful away from it. In particular, normalization can cancel a
    cutoff envelope for a lone neighbor: finite values and gradients do not
    imply the normalized model is continuous when that neighbor disappears.
    """
    if not isinstance(power, int) or isinstance(power, bool) or power < 1:
        raise ValueError("power must be a positive integer")
    if not math.isfinite(empty_threshold) or empty_threshold < 0:
        raise ValueError("empty_threshold must be finite and nonnegative")
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("descriptors must be matrices with matching feature counts")
    if left.shape[1] == 0:
        raise ValueError("descriptors must contain at least one feature")
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError("descriptor tensors must have the same dtype and device")
    if not left.is_floating_point() or not right.is_floating_point():
        raise ValueError("descriptors must be floating-point tensors")
    if not bool(torch.isfinite(left.detach()).all()) or not bool(torch.isfinite(right.detach()).all()):
        raise ValueError("descriptors must contain only finite values")
    for species, descriptors in ((left_species, left), (right_species, right)):
        if species.shape != (descriptors.shape[0],) or species.dtype != torch.long:
            raise ValueError("species must be a torch.long vector with one code per row")
        if species.device != descriptors.device:
            raise ValueError("species and descriptors must be on the same device")
    amplitude = torch.as_tensor(amplitude, dtype=left.dtype, device=left.device)
    if amplitude.ndim != 0 or not bool(torch.isfinite(amplitude.detach())):
        raise ValueError("amplitude must be a finite scalar")
    left_unit = _normalize_descriptors(left, empty_threshold)
    right_unit = _normalize_descriptors(right, empty_threshold)
    covariance = (left_unit @ right_unit.T).pow(power) * amplitude.square()
    return covariance * (left_species[:, None] == right_species[None, :])
