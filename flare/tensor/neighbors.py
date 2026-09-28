"""Correctness-first Torch neighbor lists and differentiable edge geometry.

Topology is discrete and built outside autograd. Reconstruct edge vectors from
the original positions and cells to differentiate with respect to coordinates
or strain. This all-pairs implementation targets reference cases and small
systems: work scales with N**2 times the number of searched lattice images,
and temporary pair storage is O(N**2). A cell-list implementation can replace
the search without changing the edge representation.
"""

from itertools import product
import math
from numbers import Real
from typing import Tuple

import torch

from .structures import (
    PBC,
    StructureBatch,
    _pbc_tensor,
    _validate_geometry,
    _validate_periodic_cell,
)


Edges = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _cutoff_value(cutoff: float) -> float:
    if isinstance(cutoff, torch.Tensor):
        if cutoff.ndim != 0 or not cutoff.is_floating_point():
            raise TypeError("cutoff must be a real scalar")
        cutoff = cutoff.detach().item()
    if isinstance(cutoff, bool) or not isinstance(cutoff, Real):
        raise TypeError("cutoff must be a real scalar")
    cutoff = float(cutoff)
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    return cutoff


def _empty_edges(device: torch.device) -> Edges:
    empty = torch.empty(0, dtype=torch.long, device=device)
    return empty, empty.clone(), torch.empty((0, 3), dtype=torch.long, device=device)


@torch.no_grad()
def neighbor_list(
    positions: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    pbc: PBC = True,
) -> Edges:
    """Return directed ``(first, second, shifts)`` for distances < cutoff.

    ``shifts`` contains integer row-lattice coefficients. Both directions of
    every pair and all periodic images within the cutoff are included; only
    the same atom at zero shift is excluded. Unwrapped positions and skew
    cells are supported. Nonperiodic lattice rows can be zero or dependent;
    periodic rows must be independent. Outputs are int64 on the input device.
    """
    _validate_geometry(positions, cell)
    cutoff = _cutoff_value(cutoff)
    pbc = _pbc_tensor(pbc, positions.device)
    _validate_periodic_cell(cell, pbc)
    if len(positions) == 0:
        return _empty_edges(positions.device)

    periodic_axes = torch.nonzero(pbc, as_tuple=True)[0]
    wrapping = torch.zeros((len(positions), 3), dtype=torch.long, device=positions.device)
    bounds = [0, 0, 0]
    if len(periodic_axes):
        basis = cell[periodic_axes]
        # For a partial lattice, the pseudoinverse supplies a dual basis in
        # its span without inventing nonperiodic cell vectors. The norm of
        # each dual vector bounds its fractional coordinate for |r| < cutoff.
        dual = torch.linalg.pinv(basis)
        fractional = positions @ dual
        wrap_values = torch.floor(fractional)
        if not bool(torch.isfinite(wrap_values).all()) or bool(
            (wrap_values.abs() >= 2**62).any()
        ):
            raise ValueError("unwrapped positions exceed supported lattice index range")
        wrapping[:, periodic_axes] = wrap_values.to(torch.long)
        extent_values = torch.ceil(cutoff * torch.linalg.vector_norm(dual, dim=0))
        if not bool(torch.isfinite(extent_values).all()) or bool(
            (extent_values >= 2**61).any()
        ):
            raise ValueError("cutoff and cell exceed supported lattice index range")
        extents = extent_values.to(torch.long)
        for axis, extent in zip(periodic_axes.tolist(), extents.tolist()):
            bounds[axis] = extent

    wrapped = positions - wrapping.to(dtype=positions.dtype) @ cell
    differences = wrapped.unsqueeze(0) - wrapped.unsqueeze(1)
    first_chunks, second_chunks, shift_chunks = [], [], []
    for image in product(*(range(-extent, extent + 1) for extent in bounds)):
        image_shift = torch.tensor(image, dtype=torch.long, device=positions.device)
        vectors = differences + image_shift.to(dtype=positions.dtype) @ cell
        keep = (vectors * vectors).sum(dim=-1) < cutoff * cutoff
        if image == (0, 0, 0):
            keep.fill_diagonal_(False)
        first, second = torch.nonzero(keep, as_tuple=True)
        if first.numel():
            first_chunks.append(first)
            second_chunks.append(second)
            shift_chunks.append(image_shift + wrapping[first] - wrapping[second])

    if not first_chunks:
        return _empty_edges(positions.device)
    first = torch.cat(first_chunks)
    second = torch.cat(second_chunks)
    shifts = torch.cat(shift_chunks)
    order = torch.argsort(first, stable=True)
    return first[order], second[order], shifts[order]


def get_edge_vectors(
    positions: torch.Tensor,
    cell: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct edge vectors with coordinate and cell autograd intact."""
    return positions[second] - positions[first] + shifts.to(dtype=cell.dtype) @ cell


@torch.no_grad()
def batched_neighbor_list(batch: StructureBatch, cutoff: float) -> Edges:
    """Build independent lists and return global atom indices for the batch."""
    cutoff = _cutoff_value(cutoff)
    first_chunks, second_chunks, shift_chunks = [], [], []
    for index in range(batch.num_structures):
        start, end = int(batch.ptr[index]), int(batch.ptr[index + 1])
        first, second, shifts = neighbor_list(
            batch.positions[start:end], batch.cells[index], cutoff, batch.pbc[index]
        )
        first_chunks.append(first + start)
        second_chunks.append(second + start)
        shift_chunks.append(shifts)
    if not first_chunks:
        return _empty_edges(batch.positions.device)
    return torch.cat(first_chunks), torch.cat(second_chunks), torch.cat(shift_chunks)


def get_batched_edge_vectors(
    batch: StructureBatch,
    first: torch.Tensor,
    second: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct global edges using their central atom's structure cell."""
    cells = batch.cells[batch.structure_indices[first]]
    image_vectors = torch.bmm(shifts.to(dtype=cells.dtype).unsqueeze(1), cells).squeeze(1)
    return batch.positions[second] - batch.positions[first] + image_vectors
