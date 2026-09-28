"""Torch B2 power spectra with the legacy C++ feature packing.

Descriptors are raw, unnormalized values. Normalization and the legacy empty
environment threshold belong to the kernel, not the descriptor calculation.
"""

import math
from dataclasses import dataclass

import torch
from e3nn import get_optimization_defaults, set_optimization_defaults
from e3nn.o3 import SphericalHarmonics
from torch import nn

from .neighbors import (
    batched_neighbor_list,
    get_batched_edge_vectors,
    get_edge_vectors,
    neighbor_list,
)
from .radial import chebyshev_radial


@dataclass(frozen=True)
class B2LocalTerms:
    """Raw B2 values and local quantities used by derivative assembly.

    ``basis_derivatives[e, n, h, mu]`` is the Cartesian derivative of the
    radial/harmonic basis contribution on directed edge ``e``.  The neighbor
    species selects the corresponding combined species/radial channel.  These
    terms are internal building blocks: their layout is intentionally local to
    one structure and is not a persistent training cache API.
    """

    values: torch.Tensor
    coefficients: torch.Tensor
    basis_derivatives: torch.Tensor
    first: torch.Tensor
    second: torch.Tensor
    shifts: torch.Tensor
    vectors: torch.Tensor


class B2(nn.Module):
    """Species-resolved B2 with Chebyshev radial functions and quadratic cutoff.

    ``species`` contains fixed global integer codes in ``[0, n_species)``.
    Combining a species and radial channel gives ``s * n_radial + n``. The
    output packs channel pairs ``a <= b`` with angular order ``l`` varying
    fastest, without rescaling off-diagonal pairs. Thus its dimension and
    Euclidean metric match the C++ B2 implementation.

    Neighbor topology is discrete. Supply precomputed ``edges=(i, j, shifts)``
    when applying derivative transforms; edge vectors are always recomputed
    from the supplied positions and cell so coordinate and strain graphs are
    preserved. Geometry derivatives are defined away from neighbor changes
    and coincident atoms. Precomputed edges must exclude coincident atoms;
    the topology-building path checks this before descriptor evaluation.
    """

    def __init__(
        self,
        n_species: int,
        n_radial: int = 8,
        lmax: int = 3,
        cutoff: float = 3.0,
    ):
        super().__init__()
        for name, value, minimum in (
            ("n_species", n_species, 1),
            ("n_radial", n_radial, 1),
            ("lmax", lmax, 0),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")
        self.n_species = n_species
        self.n_radial = n_radial
        self.lmax = lmax
        self.cutoff = float(cutoff)
        self.n_channels = n_species * n_radial
        self.n_features = self.n_channels * (self.n_channels + 1) // 2 * (lmax + 1)
        # The legacy TorchScript evaluator can fail inside torch.func after
        # ordinary forward/backward calls. Choose eager tensor operations at
        # construction; e3nn exposes this through its optimization defaults.
        # Restore ALL defaults (including e3nn 0.6's jit_mode) on every exit.
        defaults = get_optimization_defaults()
        try:
            set_optimization_defaults(jit_script_fx=False)
            self.harmonics = SphericalHarmonics(
                list(range(lmax + 1)), normalize=True, normalization="integral"
            )
        finally:
            set_optimization_defaults(**defaults)

    def _validate_species(self, positions, species):
        if species.ndim != 1 or species.shape[0] != positions.shape[0]:
            raise ValueError("species must have one entry per atom")
        if species.dtype != torch.long:
            raise ValueError("species must use torch.long global species codes")
        if species.device != positions.device:
            raise ValueError("species and positions must be on the same device")
        if bool(((species < 0) | (species >= self.n_species)).any()):
            raise ValueError("species code outside the fixed global vocabulary")

    def _edge_basis(self, vectors):
        distances = torch.linalg.vector_norm(vectors, dim=-1)
        radial = chebyshev_radial(distances, self.n_radial, self.cutoff)
        harmonics = self.harmonics(vectors)
        return radial.unsqueeze(-1) * harmonics.unsqueeze(-2)

    def _coefficients(self, positions, species, first, second, bonds):
        # Accumulate directly into atom/species channels, avoiding a full
        # edge x species one-hot tensor.
        coefficients = positions.new_zeros(
            (positions.shape[0] * self.n_species, self.n_radial,
             (self.lmax + 1) ** 2)
        ).index_add(0, first * self.n_species + species[second], bonds)
        coefficients = coefficients.reshape(
            positions.shape[0], self.n_channels, (self.lmax + 1) ** 2
        )
        return coefficients

    def _values_from_coefficients(self, coefficients, positions, cells):
        pairs = torch.triu_indices(
            self.n_channels, self.n_channels, device=positions.device
        )
        spectra = []
        for angular in range(self.lmax + 1):
            block = coefficients[..., angular**2 : (angular + 1) ** 2]
            gram = block @ block.transpose(-1, -2)
            spectra.append(gram[:, pairs[0], pairs[1]])
        result = torch.stack(spectra, dim=-1).reshape(
            positions.shape[0], self.n_features
        )
        # Retain coordinate/cell dependence even for a completely empty edge
        # list, including a differentiable zero for second derivatives.
        return result + 0 * (positions.square().sum() + cells.square().sum())

    def _from_edges(self, positions, cells, species, first, second, vectors):
        self._validate_species(positions, species)
        bonds = self._edge_basis(vectors)
        coefficients = self._coefficients(positions, species, first, second, bonds)
        return self._values_from_coefficients(coefficients, positions, cells)

    def _basis_derivatives(self, vectors):
        """Return d(radial * harmonic)/d(edge vector) for every directed edge.

        One forward-mode directional derivative per Cartesian component gives
        the diagonal, per-edge Jacobian without constructing an edge-by-edge
        Jacobian.  The basis function is edge-separable, so a tangent that
        selects component ``mu`` on every edge yields its local derivative on
        every edge simultaneously.
        """
        if not len(vectors):
            values = self._edge_basis(vectors)
            derivatives = vectors.new_zeros(
                (0, self.n_radial, (self.lmax + 1) ** 2, 3)
            ) + 0 * vectors.sum()
            return values, derivatives
        derivatives = []
        values = None
        for component in range(3):
            tangent = torch.zeros_like(vectors)
            tangent[:, component] = 1
            current, derivative = torch.func.jvp(
                self._edge_basis, (vectors,), (tangent,)
            )
            if values is None:
                values = current
            derivatives.append(derivative)
        return values, torch.stack(derivatives, dim=-1)

    def local_terms(self, positions, cell, species, pbc=True, edges=None):
        """Return raw B2 values, coefficients, and local basis derivatives.

        This is the common foundation for streamed-Q and Lambda derivative
        assembly.  Like :meth:`forward`, supplied edges are trusted fixed
        topology and geometry is reconstructed from differentiable tensors.
        """
        if positions.ndim != 2 or positions.shape[-1] != 3 or cell.shape != (3, 3):
            raise ValueError("positions and cell must have shapes (n_atoms, 3) and (3, 3)")
        if not positions.is_floating_point() or positions.dtype != cell.dtype:
            raise ValueError("positions and cell must have the same floating-point dtype")
        if positions.device != cell.device:
            raise ValueError("positions and cell must be on the same device")
        self._validate_species(positions, species)
        build_topology = edges is None
        if edges is None:
            edges = neighbor_list(positions, cell, self.cutoff, pbc)
        first, second, shifts = edges
        vectors = get_edge_vectors(positions, cell, first, second, shifts)
        if build_topology and bool((vectors.detach().square().sum(dim=-1) == 0).any()):
            raise ValueError("B2 is undefined for coincident atoms")
        bonds, basis_derivatives = self._basis_derivatives(vectors)
        coefficients = self._coefficients(positions, species, first, second, bonds)
        values = self._values_from_coefficients(coefficients, positions, cell)
        return B2LocalTerms(
            values, coefficients, basis_derivatives, first, second, shifts, vectors
        )

    def forward(self, positions, cell, species, pbc=True, edges=None):
        """Return an ``(n_atoms, n_features)`` tensor on the input device/dtype."""
        if positions.ndim != 2 or positions.shape[-1] != 3 or cell.shape != (3, 3):
            raise ValueError("positions and cell must have shapes (n_atoms, 3) and (3, 3)")
        if not positions.is_floating_point() or positions.dtype != cell.dtype:
            raise ValueError("positions and cell must have the same floating-point dtype")
        if positions.device != cell.device:
            raise ValueError("positions and cell must be on the same device")
        build_topology = edges is None
        if edges is None:
            edges = neighbor_list(positions, cell, self.cutoff, pbc)
        first, second, shifts = edges
        vectors = get_edge_vectors(positions, cell, first, second, shifts)
        if build_topology and bool((vectors.detach().square().sum(dim=-1) == 0).any()):
            raise ValueError("B2 is undefined for coincident atoms")
        return self._from_edges(positions, cell, species, first, second, vectors)

    def forward_batch(self, batch, edges=None):
        """Evaluate a ragged ``StructureBatch``, preserving its atom ordering."""
        build_topology = edges is None
        if edges is None:
            edges = batched_neighbor_list(batch, self.cutoff)
        first, second, shifts = edges
        vectors = get_batched_edge_vectors(batch, first, second, shifts)
        if build_topology and bool((vectors.detach().square().sum(dim=-1) == 0).any()):
            raise ValueError("B2 is undefined for coincident atoms")
        return self._from_edges(
            batch.positions, batch.cells, batch.species, first, second, vectors
        )
