"""Tensor inputs for the optional Torch backend.

Species are nonnegative indices in a model-wide vocabulary, never renumbered
according to the species present in an individual structure. Cells use row
vectors, as in ASE. Tensor inputs retain their device, dtype, and autograd graph.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch


PBC = Union[bool, Sequence[bool], torch.Tensor]


def _pbc_tensor(pbc: PBC, device: torch.device) -> torch.Tensor:
    """Normalize scalar or three-axis boundary conditions without guessing."""
    if isinstance(pbc, bool):
        return torch.full((3,), pbc, dtype=torch.bool, device=device)
    if isinstance(pbc, torch.Tensor):
        if pbc.dtype != torch.bool:
            raise TypeError("pbc must contain booleans")
        if pbc.ndim == 0:
            return pbc.to(device=device).expand(3)
        if pbc.shape != (3,):
            raise ValueError("pbc must be a scalar or have shape (3,)")
        return pbc.to(device=device)
    if not isinstance(pbc, (tuple, list)) or len(pbc) != 3:
        raise ValueError("pbc must be a boolean or three booleans")
    if not all(isinstance(value, bool) for value in pbc):
        raise TypeError("pbc must contain booleans")
    return torch.tensor(pbc, dtype=torch.bool, device=device)


def _validate_geometry(positions: torch.Tensor, cell: torch.Tensor) -> None:
    if not isinstance(positions, torch.Tensor) or not isinstance(cell, torch.Tensor):
        raise TypeError("positions and cell must be Torch tensors")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must have dtype float32 or float64")
    if cell.dtype != positions.dtype or cell.device != positions.device:
        raise ValueError("positions and cell must have the same dtype and device")
    if not bool(torch.isfinite(positions).all()) or not bool(torch.isfinite(cell).all()):
        raise ValueError("positions and cell must be finite")


def _validate_periodic_cell(cell: torch.Tensor, pbc: torch.Tensor) -> None:
    # Only periodic rows must span a lattice: molecules may have a zero cell,
    # and wires/slabs need not supply the nonperiodic cell vectors.
    periodic_basis = cell.detach()[pbc]
    if (
        periodic_basis.shape[0]
        and int(torch.linalg.matrix_rank(periodic_basis)) != periodic_basis.shape[0]
    ):
        raise ValueError("periodic cell vectors must be linearly independent")


@dataclass(frozen=True)
class StructureBatch:
    """A ragged batch of structures with a fixed global species vocabulary.

    ``ptr[b]:ptr[b + 1]`` selects structure ``b`` from ``positions`` and
    ``species``. Empty structures are allowed. All tensors share a device;
    positions/cells are float32 or float64, species/ptr are int64, and pbc is
    bool. Frozen fields prevent replacement, not in-place tensor mutation:
    callers must not mutate inputs while a derivative graph is in use.
    """

    positions: torch.Tensor
    species: torch.Tensor
    cells: torch.Tensor
    pbc: torch.Tensor
    ptr: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (self.positions, self.species, self.cells, self.pbc, self.ptr)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("StructureBatch fields must be Torch tensors")
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.positions.dtype not in (torch.float32, torch.float64):
            raise TypeError("positions must have dtype float32 or float64")
        if any(value.device != self.positions.device for value in tensors[1:]):
            raise ValueError("all StructureBatch tensors must share a device")
        if self.cells.ndim != 3 or self.cells.shape[1:] != (3, 3):
            raise ValueError("cells must have shape (B, 3, 3)")
        if self.cells.dtype != self.positions.dtype:
            raise ValueError("positions and cells must have the same dtype")
        if self.species.dtype != torch.long:
            raise TypeError("species must have dtype int64")
        if self.species.shape != (len(self.positions),):
            raise ValueError("species must have shape (N,)")
        if bool((self.species < 0).any()):
            raise ValueError("species must be nonnegative global vocabulary indices")
        if self.pbc.dtype != torch.bool:
            raise TypeError("pbc must have dtype bool")
        if self.pbc.shape != (len(self.cells), 3):
            raise ValueError("pbc must have shape (B, 3)")
        if self.ptr.dtype != torch.long:
            raise TypeError("ptr must have dtype int64")
        if self.ptr.shape != (len(self.cells) + 1,):
            raise ValueError("ptr must have shape (B + 1,)")
        if int(self.ptr[0]) != 0 or int(self.ptr[-1]) != len(self.positions):
            raise ValueError("ptr must start at zero and end at the number of atoms")
        if bool((self.ptr[1:] < self.ptr[:-1]).any()):
            raise ValueError("ptr must be nondecreasing")
        if not bool(torch.isfinite(self.positions).all()) or not bool(
            torch.isfinite(self.cells).all()
        ):
            raise ValueError("positions and cells must be finite")
        for cell, pbc in zip(self.cells, self.pbc):
            _validate_periodic_cell(cell, pbc)

    @property
    def num_structures(self) -> int:
        return len(self.cells)

    @property
    def structure_indices(self) -> torch.Tensor:
        """The structure index of each atom, on the input device."""
        return torch.repeat_interleave(
            torch.arange(self.num_structures, device=self.positions.device),
            self.ptr[1:] - self.ptr[:-1],
        )

    @classmethod
    def from_single(
        cls,
        positions: torch.Tensor,
        species: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        pbc: PBC = False,
    ) -> "StructureBatch":
        """Wrap one structure without copying or detaching its geometry."""
        if not isinstance(positions, torch.Tensor):
            raise TypeError("positions must be a Torch tensor")
        if cell is None:
            cell = positions.new_zeros((3, 3))
        _validate_geometry(positions, cell)
        return cls(
            positions=positions,
            species=species,
            cells=cell.unsqueeze(0),
            pbc=_pbc_tensor(pbc, positions.device).unsqueeze(0),
            ptr=torch.tensor([0, len(positions)], dtype=torch.long, device=positions.device),
        )
