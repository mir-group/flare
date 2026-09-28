"""Detached, coefficient-first B2 cache for sparse observation assembly.

The cache holds per-center coefficients and local edge-basis derivatives. It
maps inducing B2 features through the coefficients before contracting with
edges, avoiding a persistent edge-by-feature Q tensor. Geometry and descriptor
parameters are fixed when the cache is built; inducing descriptors and kernel
amplitude remain differentiable when a block is assembled.
"""

from dataclasses import dataclass
from typing import Tuple

import torch

from .b2 import B2
from .kernels import _normalization_data
from .neighbors import get_edge_vectors, neighbor_list
from .observations import ObservationLayout
from .structures import _validate_geometry


@dataclass(frozen=True)
class _NeighborGroup:
    species: int
    first: torch.Tensor
    second: torch.Tensor
    vectors: torch.Tensor
    derivatives: torch.Tensor


@dataclass(frozen=True)
class _CenterTile:
    species: int
    centers: torch.Tensor
    groups: Tuple[_NeighborGroup, ...]


@dataclass(frozen=True)
class GroupedLambdaCache:
    """Fixed-geometry data for repeated inducing-to-observation blocks.

    Rebuild after changing geometry, topology, B2 settings, or descriptor
    parameters. The frozen container does not protect its tensors against
    in-place mutation.
    """

    values: torch.Tensor
    species: torch.Tensor
    coefficients: torch.Tensor
    values_unit: torch.Tensor
    values_active: torch.Tensor
    values_norm: torch.Tensor
    lambda_q: Tuple[torch.Tensor, ...]
    tiles: Tuple[_CenterTile, ...]
    volume: torch.Tensor
    n_channels: int
    n_radial: int
    lmax: int
    n_species: int

    @property
    def n_atoms(self):
        return self.values.shape[0]

    @property
    def n_features(self):
        return self.values.shape[1]


def _symmetric_b2_weights(values, n_channels, lmax):
    """Expand packed B2 rows, including both diagonal product-rule terms."""
    pairs = torch.triu_indices(n_channels, n_channels, device=values.device)
    packed = values.reshape(len(values), len(pairs[0]), lmax + 1).transpose(1, 2)
    result = values.new_zeros((len(values), lmax + 1, n_channels * n_channels))
    result = result.index_add(2, pairs[0] * n_channels + pairs[1], packed)
    result = result.index_add(2, pairs[1] * n_channels + pairs[0], packed)
    return result.reshape(len(values), lmax + 1, n_channels, n_channels)


@torch.no_grad()
def build_grouped_lambda_cache(
    descriptor, positions, cell, species, pbc=True, edges=None, center_tile=32,
):
    """Cache a B2 structure for coefficient-first E/F/stress assembly.

    Edge topology is fixed and can be supplied by the caller. The returned
    tensors are detached from geometry and descriptor parameters. Center and
    neighbor-species groups use padded edge rows; invalid rows have exactly
    zero basis derivatives.
    """
    if not isinstance(descriptor, B2):
        raise TypeError("descriptor must be a B2")
    if not isinstance(center_tile, int) or isinstance(center_tile, bool) or center_tile < 1:
        raise ValueError("center_tile must be a positive integer")
    _validate_geometry(positions, cell)
    descriptor._validate_species(positions, species)
    if edges is None:
        edges = neighbor_list(positions, cell, descriptor.cutoff, pbc)
    vectors = get_edge_vectors(positions, cell, *edges)
    if bool((vectors.square().sum(dim=-1) == 0).any()):
        raise ValueError("B2 is undefined for coincident atoms")
    terms = descriptor.local_terms(positions, cell, species, pbc=pbc, edges=edges)
    values = terms.values.detach().clone()
    codes = species.detach().clone()
    coefficients = terms.coefficients.detach().clone()
    values_unit, values_active, values_norm = _normalization_data(values, 1e-8)
    symmetric = _symmetric_b2_weights(values_unit, descriptor.n_channels, descriptor.lmax)
    lambda_q = []
    for angular in range(descriptor.lmax + 1):
        harmonic = slice(angular ** 2, (angular + 1) ** 2)
        lambda_q.append(torch.einsum(
            "iab,ibh->iah", symmetric[:, angular], coefficients[:, :, harmonic],
        ))

    tiles = []
    for center_code in codes.unique().tolist():
        centers = torch.nonzero(codes == center_code, as_tuple=True)[0]
        for start in range(0, len(centers), center_tile):
            selected = centers[start:start + center_tile]
            edge_indices = torch.nonzero(
                torch.isin(terms.first, selected), as_tuple=True,
            )[0]
            first = terms.first[edge_indices]
            second = terms.second[edge_indices]
            local = torch.searchsorted(selected, first)
            neighbor_codes = codes[second]
            groups = []
            for neighbor_code in neighbor_codes.unique().tolist():
                per_center = [
                    torch.nonzero((local == center) & (neighbor_codes == neighbor_code),
                                  as_tuple=True)[0]
                    for center in range(len(selected))
                ]
                width = max(len(indices) for indices in per_center)
                indices = first.new_zeros((len(selected), width))
                valid = torch.zeros_like(indices, dtype=torch.bool)
                for center, current in enumerate(per_center):
                    indices[center, :len(current)] = current
                    valid[center, :len(current)] = True
                derivative = terms.basis_derivatives[edge_indices[indices]]
                derivative = derivative * valid[..., None, None, None]
                groups.append(_NeighborGroup(
                    int(neighbor_code), first[indices].detach().clone(),
                    second[indices].detach().clone(),
                    terms.vectors[edge_indices[indices]].detach().clone(),
                    derivative.detach().clone(),
                ))
            tiles.append(_CenterTile(int(center_code), selected.detach().clone(),
                                     tuple(groups)))

    return GroupedLambdaCache(
        values, codes, coefficients, values_unit, values_active, values_norm,
        tuple(lambda_q), tuple(tiles), torch.linalg.det(cell).abs().detach().clone(),
        descriptor.n_channels, descriptor.n_radial, descriptor.lmax,
        descriptor.n_species,
    )


def _validate_inputs(cache, inducing, inducing_species, layout, amplitude, power,
                     sparse_chunk):
    if not isinstance(cache, GroupedLambdaCache):
        raise TypeError("cache must be a GroupedLambdaCache")
    if not isinstance(layout, ObservationLayout) or layout.n_atoms != cache.n_atoms:
        raise ValueError("layout must describe the cached atoms")
    if layout.device != cache.values.device:
        raise ValueError("layout and cache must share a device")
    if not isinstance(sparse_chunk, int) or isinstance(sparse_chunk, bool) or sparse_chunk < 1:
        raise ValueError("sparse_chunk must be a positive integer")
    if (not isinstance(inducing, torch.Tensor) or inducing.ndim != 2
            or inducing.shape[1] != cache.n_features
            or inducing.dtype != cache.values.dtype
            or inducing.device != cache.values.device):
        raise ValueError("inducing descriptors must match the cached dtype, device, and features")
    if not bool(torch.isfinite(inducing.detach()).all()):
        raise ValueError("inducing descriptors must be finite")
    if (not isinstance(inducing_species, torch.Tensor)
            or inducing_species.shape != (len(inducing),)
            or inducing_species.dtype != torch.long
            or inducing_species.device != cache.values.device):
        raise ValueError("inducing_species must be a torch.long vector on the cache device")
    if bool(((inducing_species < 0) | (inducing_species >= cache.n_species)).any()):
        raise ValueError("inducing species code outside the cached vocabulary")
    if not isinstance(power, int) or isinstance(power, bool) or power < 1:
        raise ValueError("power must be a positive integer")
    amplitude = torch.as_tensor(amplitude, dtype=cache.values.dtype,
                                device=cache.values.device)
    if amplitude.ndim != 0 or not bool(torch.isfinite(amplitude.detach())):
        raise ValueError("amplitude must be a finite scalar")
    if bool(layout.stress_mask.any()) and not bool(torch.isfinite(cache.volume) & (cache.volume > 0)):
        raise ValueError("stress observations require a finite positive cached cell volume")
    return amplitude


def cached_grouped_lambda_observation_covariance(
    cache, inducing_descriptors, inducing_species, layout,
    amplitude=1.0, power=2, sparse_chunk=32,
):
    """Assemble one cached structure's inducing-to-E/F/stress block.

    The output remains differentiable with respect to inducing descriptors
    and amplitude. The cache is detached from geometry and descriptor settings.
    """
    amplitude = _validate_inputs(cache, inducing_descriptors, inducing_species,
                                 layout, amplitude, power, sparse_chunk)
    if not len(inducing_descriptors) or not layout.size:
        return cache.values.new_zeros((len(inducing_descriptors), layout.size)) + 0 * (
            inducing_descriptors.sum() + amplitude
        )
    inducing_unit, inducing_active, _ = _normalization_data(inducing_descriptors, 1e-8)
    need_force = bool(layout.force_mask.any())
    need_stress = bool(layout.stress_mask.any())
    row, column = [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]
    blocks = []
    for start in range(0, len(inducing_descriptors), sparse_chunk):
        stop = min(start + sparse_chunk, len(inducing_descriptors))
        units = inducing_unit[start:stop]
        dots = units @ cache.values_unit.T
        gate = inducing_species[start:stop, None] == cache.species[None, :]
        energy = (dots.pow(power) * amplitude.square() * gate).sum(dim=1, keepdim=True)
        forces = cache.values.new_zeros((stop - start, cache.n_atoms, 3)) if need_force else None
        stress = cache.values.new_zeros((stop - start, 3, 3)) if need_stress else None
        if need_force or need_stress:
            species_rows = {}
            symmetric = {}
            for code in {tile.species for tile in cache.tiles}:
                rows = torch.nonzero(inducing_species[start:stop] == code, as_tuple=True)[0]
                species_rows[code] = rows
                if len(rows):
                    symmetric[code] = _symmetric_b2_weights(
                        units[rows], cache.n_channels, cache.lmax,
                    )
            for tile in cache.tiles:
                rows = species_rows[tile.species]
                if not len(rows) or not tile.groups:
                    continue
                centers = tile.centers
                c = dots[rows[:, None], centers[None, :]]
                active = (inducing_active[start:stop][rows, None]
                          & cache.values_active[centers][None, :])
                beta = (amplitude.square() * power
                        * (torch.ones_like(c) if power == 1 else c.pow(power - 1))
                        / cache.values_norm[centers, 0][None, :]) * active
                gradients = [cache.values.new_zeros((
                    len(rows), len(centers), group.first.shape[1], 3,
                )) for group in tile.groups]
                for angular in range(cache.lmax + 1):
                    harmonic = slice(angular ** 2, (angular + 1) ** 2)
                    h = (angular + 1) ** 2 - angular ** 2
                    coefficient = cache.coefficients[centers, :, harmonic]
                    flattened = coefficient.permute(1, 0, 2).reshape(cache.n_channels, -1)
                    mapped = (symmetric[tile.species][:, angular] @ flattened).reshape(
                        len(rows), cache.n_channels, len(centers), h,
                    ).permute(0, 2, 1, 3)
                    mapped = beta[..., None, None] * (
                        mapped - c[..., None, None]
                        * cache.lambda_q[angular][centers][None, ...]
                    )
                    for index, group in enumerate(tile.groups):
                        channels = slice(group.species * cache.n_radial,
                                         (group.species + 1) * cache.n_radial)
                        gradients[index] = gradients[index] + torch.einsum(
                            "mirh,ierhc->miec", mapped[:, :, channels, :],
                            group.derivatives[:, :, :, harmonic, :],
                        )
                for group, gradient in zip(tile.groups, gradients):
                    first = group.first.reshape(-1)
                    second = group.second.reshape(-1)
                    gradient = gradient.reshape(len(rows), -1, 3)
                    if forces is not None:
                        flat_force = forces.view(-1, 3)
                        first_index = (rows[:, None] * cache.n_atoms + first[None, :]).reshape(-1)
                        second_index = (rows[:, None] * cache.n_atoms + second[None, :]).reshape(-1)
                        flat_force.index_add_(0, first_index, gradient.reshape(-1, 3))
                        flat_force.index_add_(0, second_index, -gradient.reshape(-1, 3))
                    if stress is not None:
                        local = -torch.einsum(
                            "mec,ed->mcd", gradient, group.vectors.reshape(-1, 3),
                        ) / cache.volume
                        stress.index_add_(0, rows, local)
        columns = [energy] if layout.energy else []
        if forces is not None:
            columns.append(forces[:, layout.force_mask])
        if stress is not None:
            columns.append(stress[:, row, column][:, layout.stress_mask])
        blocks.append(torch.cat(columns, dim=1))
    return torch.cat(blocks, dim=0) + 0 * (inducing_descriptors.sum() + amplitude)


def assemble_cached_grouped_lambda_observation_covariance(
    caches, inducing_descriptors, inducing_species, layouts,
    amplitude=1.0, power=2, sparse_chunk=32,
):
    """Concatenate blocks for a sequence of grouped Lambda caches."""
    if len(caches) != len(layouts):
        raise ValueError("one observation layout is required per GroupedLambdaCache")
    if not caches:
        amplitude = torch.as_tensor(amplitude, dtype=inducing_descriptors.dtype,
                                    device=inducing_descriptors.device)
        return inducing_descriptors.new_zeros((len(inducing_descriptors), 0)) + 0 * (
            inducing_descriptors.sum() + amplitude
        )
    return torch.cat([
        cached_grouped_lambda_observation_covariance(
            cache, inducing_descriptors, inducing_species, layout,
            amplitude=amplitude, power=power, sparse_chunk=sparse_chunk,
        )
        for cache, layout in zip(caches, layouts)
    ], dim=1)
