"""Explicit energy/force/stress layouts and inducing-observation operators.

Observation columns are structure-major: total energy, atom-major Cartesian
forces, then native stresses in ``[xx, xy, xz, yy, yz, zz]`` order, omitting
masked entries. The direct-autograd operator recomputes descriptors inside
each inducing chunk. The streamed-Q operator instead builds local derivative
blocks once per structure and releases them as it advances through its
central atoms. Cached-Q is an explicit alternative that retains detached local
blocks for a fixed training structure. Lambda contracts kernel sensitivity
through coefficients before edge-basis derivatives and is transient like Q.
Direct, streamed-Q, and Lambda returned tensors retain the graph needed to
differentiate kernel and descriptor parameters; cached-Q returned tensors
retain only kernel-side graphs. Chunking bounds the intermediates of an
individual Jacobian evaluation or kernel-sensitivity contraction, not the
total graph retained by a differentiable training objective.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .kernels import _normalization_data, normalized_dot_product
from .neighbors import get_edge_vectors, neighbor_list
from .structures import _validate_geometry


def native_stress_to_ase(stress):
    """Convert native stresses to ASE's sign and ``xx,yy,zz,yz,xz,xy`` order."""
    if not isinstance(stress, torch.Tensor) or stress.ndim < 1 or stress.shape[-1] != 6:
        raise ValueError("stress must be a tensor with final dimension 6")
    return -stress[..., [0, 3, 5, 4, 2, 1]]


def ase_stress_to_native(stress):
    """Convert ASE stresses to native sign/order, without a shear factor."""
    if not isinstance(stress, torch.Tensor) or stress.ndim < 1 or stress.shape[-1] != 6:
        raise ValueError("stress must be a tensor with final dimension 6")
    return -stress[..., [0, 5, 4, 1, 3, 2]]


@dataclass(frozen=True)
class ObservationLayout:
    """Select scalar observations from one structure.

    Masks are boolean tensors on the geometry's device. Frozen fields do not
    protect against in-place tensor mutation; callers must keep masks unchanged
    while their packed data or covariance matrices remain in use.
    """

    energy: bool
    force_mask: torch.Tensor
    stress_mask: torch.Tensor

    def __post_init__(self):
        if not isinstance(self.energy, bool):
            raise TypeError("energy must be a boolean")
        if not isinstance(self.force_mask, torch.Tensor) or not isinstance(self.stress_mask, torch.Tensor):
            raise TypeError("observation masks must be Torch tensors")
        if self.force_mask.dtype != torch.bool or self.stress_mask.dtype != torch.bool:
            raise TypeError("observation masks must have boolean dtype")
        if self.force_mask.ndim != 2 or self.force_mask.shape[1] != 3:
            raise ValueError("force_mask must have shape (n_atoms, 3)")
        if self.stress_mask.shape != (6,):
            raise ValueError("stress_mask must have shape (6,)")
        if self.force_mask.device != self.stress_mask.device:
            raise ValueError("observation masks must share a device")

    @property
    def n_atoms(self):
        return self.force_mask.shape[0]

    @property
    def device(self):
        return self.force_mask.device

    @property
    def size(self):
        return int(self.energy) + int(self.force_mask.sum()) + int(self.stress_mask.sum())

    @property
    def full_indices(self):
        """Selected column indices in the full ``1 + 3*N + 6`` layout."""
        mask = torch.cat((
            torch.tensor([self.energy], dtype=torch.bool, device=self.device),
            self.force_mask.flatten(), self.stress_mask,
        ))
        return torch.nonzero(mask, as_tuple=True)[0]

    @property
    def noise_kind(self):
        """Per-observation indices: 0 energy, 1 force, 2 stress."""
        kinds = torch.cat((
            torch.zeros(1, dtype=torch.long, device=self.device),
            torch.ones(3 * self.n_atoms, dtype=torch.long, device=self.device),
            torch.full((6,), 2, dtype=torch.long, device=self.device),
        ))
        return kinds[self.full_indices]

    @classmethod
    def from_masks(cls, n_atoms, energy=False, force_mask=None, stress_mask=None, device="cpu"):
        """Build a layout; an omitted force/stress mask selects no such labels."""
        if not isinstance(n_atoms, int) or isinstance(n_atoms, bool) or n_atoms < 0:
            raise ValueError("n_atoms must be a nonnegative integer")
        force = (torch.zeros((n_atoms, 3), dtype=torch.bool, device=device)
                 if force_mask is None else torch.as_tensor(force_mask, device=device))
        stress = (torch.zeros(6, dtype=torch.bool, device=device)
                  if stress_mask is None else torch.as_tensor(stress_mask, device=device))
        result = cls(energy, force, stress)
        if result.n_atoms != n_atoms:
            raise ValueError("force_mask must have one row per atom")
        return result

    @classmethod
    def full(cls, n_atoms, device="cpu"):
        # Let from_masks validate n_atoms before constructing sized tensors.
        empty = cls.from_masks(n_atoms, device=device)
        return cls(True, torch.ones_like(empty.force_mask), torch.ones_like(empty.stress_mask))

    @classmethod
    def energy_only(cls, n_atoms, device="cpu"):
        return cls.from_masks(n_atoms, energy=True, device=device)

    def pack_labels(self, energy=None, forces=None, stress=None, species=None, atomic_offsets=None):
        """Pack selected labels, subtracting atomic offsets from total energy.

        Forces have shape ``(N,3)`` and stress has native shape/order ``(6,)``.
        Selected labels must be finite; unselected components may contain NaN.
        ``atomic_offsets`` is a vector indexed by global integer ``species``.
        Labels retain their floating dtype and gradient history. Python values
        use the dtype of the first supplied selected tensor, or Torch's default.
        """
        requested = ((self.energy, energy), (bool(self.force_mask.any()), forces),
                     (bool(self.stress_mask.any()), stress))
        tensors = [value for selected, value in requested
                   if selected and isinstance(value, torch.Tensor)]
        dtype = tensors[0].dtype if tensors else torch.get_default_dtype()
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("labels must have floating-point dtype float32 or float64")
        pieces = []
        for selected, value, shape, mask, name in (
            (self.energy, energy, (), None, "energy"),
            (bool(self.force_mask.any()), forces, (self.n_atoms, 3), self.force_mask, "forces"),
            (bool(self.stress_mask.any()), stress, (6,), self.stress_mask, "stress"),
        ):
            if not selected:
                continue
            if value is None:
                raise ValueError(f"selected {name} labels are required")
            if isinstance(value, torch.Tensor) and (value.device != self.device or value.dtype != dtype):
                raise ValueError("label tensors must share a floating dtype and the layout device")
            value = torch.as_tensor(value, dtype=dtype, device=self.device)
            if value.shape != shape:
                raise ValueError(f"{name} labels must have shape {shape}")
            value = value.reshape(1) if mask is None else value[mask]
            if name == "energy" and atomic_offsets is not None:
                if not isinstance(species, torch.Tensor) or species.dtype != torch.long or species.shape != (self.n_atoms,):
                    raise ValueError("atomic offsets require a torch.long species vector of shape (n_atoms,)")
                if species.device != self.device:
                    raise ValueError("species and labels must share a device")
                offsets = torch.as_tensor(atomic_offsets, dtype=dtype, device=self.device)
                if offsets.ndim != 1 or bool(((species < 0) | (species >= len(offsets))).any()):
                    raise ValueError("atomic_offsets must cover the global species codes")
                value = value - offsets[species].sum()
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"selected {name} labels and atomic offsets must be finite")
            pieces.append(value)
        return torch.cat(pieces) if pieces else torch.empty(0, dtype=dtype, device=self.device)

    def noise_variance(self, noise_std, relative_noise=None):
        """Return ``(std * relative_multiplier)**2`` for selected observations.

        ``noise_std`` and ``relative_noise`` have shape ``(3,)`` in E/F/S
        order. Multipliers default to one. Only selected kinds must be finite
        and strictly positive. Neither operation detaches trainable noises.
        """
        noise_std = torch.as_tensor(noise_std, device=self.device)
        if noise_std.shape != (3,) or noise_std.dtype not in (torch.float32, torch.float64):
            raise ValueError("noise_std must be a floating-point tensor of shape (3,)")
        relative_noise = (torch.ones_like(noise_std) if relative_noise is None
                          else torch.as_tensor(relative_noise, dtype=noise_std.dtype, device=self.device))
        if relative_noise.shape != (3,):
            raise ValueError("relative_noise must have shape (3,)")
        std = noise_std[self.noise_kind]
        multiplier = relative_noise[self.noise_kind]
        if not bool((torch.isfinite(std) & (std > 0) & torch.isfinite(multiplier) & (multiplier > 0)).all()):
            raise ValueError("selected noise standard deviations and multipliers must be finite and positive")
        variance = (std * multiplier).square()
        if not bool((torch.isfinite(variance) & (variance > 0)).all()):
            raise ValueError("selected noise variances must be finite and positive")
        return variance


@dataclass(frozen=True)
class CachedQ:
    """Detached geometry-dependent data for one training structure.

    Build this object with build_q_cache after a training structure is
    accepted. It owns copies of its raw B2 values, edge vectors, and raw-B2
    derivative blocks. The cache remains valid when inducing environments or
    kernel hyperparameters change, but must be rebuilt if the training geometry
    or any descriptor parameter changes. It deliberately does not retain an
    autograd path to geometry or descriptor parameters.
    """

    values: torch.Tensor
    species: torch.Tensor
    first: torch.Tensor
    second: torch.Tensor
    vectors: torch.Tensor
    q: torch.Tensor
    centers: torch.Tensor
    edge_ptr: torch.Tensor
    volume: torch.Tensor
    n_species: int

    @property
    def n_atoms(self):
        return len(self.values)

    @property
    def n_features(self):
        return self.values.shape[1]

    def __post_init__(self):
        floating = (self.values, self.vectors, self.q, self.volume)
        integer = (self.species, self.first, self.second, self.centers, self.edge_ptr)
        if (not isinstance(self.n_species, int) or isinstance(self.n_species, bool)
                or self.n_species < 1):
            raise ValueError("n_species must be a positive integer")
        if not all(isinstance(value, torch.Tensor) for value in floating + integer):
            raise TypeError("CachedQ fields must be Torch tensors")
        if (self.values.ndim != 2 or not self.values.is_floating_point()
                or self.values.shape[1] == 0):
            raise ValueError("CachedQ values must be a nonempty-feature floating matrix")
        n_atoms, n_features, n_edges = len(self.values), self.n_features, len(self.first)
        if (self.species.shape != (n_atoms,) or self.species.dtype != torch.long
                or self.first.shape != (n_edges,) or self.first.dtype != torch.long
                or self.second.shape != (n_edges,) or self.second.dtype != torch.long):
            raise ValueError("CachedQ atom and edge indices must be matching int64 vectors")
        if self.vectors.shape != (n_edges, 3) or self.q.shape != (n_edges, n_features, 3):
            raise ValueError("CachedQ vectors and Q blocks have incompatible shapes")
        if self.centers.dtype != torch.long or self.centers.ndim != 1:
            raise ValueError("CachedQ centers must be an int64 vector")
        if self.edge_ptr.dtype != torch.long or self.edge_ptr.shape != (len(self.centers) + 1,):
            raise ValueError("CachedQ edge_ptr must have one more entry than centers")
        if self.volume.shape != ():
            raise ValueError("CachedQ volume must be scalar")
        if any(value.device != self.values.device for value in floating[1:] + integer):
            raise ValueError("CachedQ tensors must share a device")
        if any(value.dtype != self.values.dtype for value in floating[1:]):
            raise ValueError("CachedQ floating tensors must share a dtype")
        if any(value.requires_grad for value in floating):
            raise ValueError("CachedQ tensors must be detached from geometry")
        if (bool((self.species < 0).any()) or bool((self.species >= self.n_species).any())
                or bool((self.first < 0).any()) or bool((self.first >= n_atoms).any())
                or bool((self.second < 0).any()) or bool((self.second >= n_atoms).any())):
            raise ValueError("CachedQ contains invalid atom or species indices")
        if (int(self.edge_ptr[0]) != 0 or int(self.edge_ptr[-1]) != n_edges
                or bool((self.edge_ptr[1:] < self.edge_ptr[:-1]).any())):
            raise ValueError("CachedQ edge_ptr must partition all Q edges")
        for center, start, stop in zip(
            self.centers.tolist(), self.edge_ptr[:-1].tolist(), self.edge_ptr[1:].tolist(),
        ):
            if center < 0 or center >= n_atoms or not bool((self.first[start:stop] == center).all()):
                raise ValueError("CachedQ blocks must be grouped by central atom")


def _edge_b2_derivatives(descriptor, coefficients, basis_derivatives, neighbor_species):
    """Build local raw-B2 edge derivatives for one central atom's edges.

    ``basis_derivatives`` has one radial/harmonic derivative per edge.  Each
    edge occupies only the radial channels belonging to its neighbor species;
    the B2 product rule then creates a compact ``(edge, feature, Cartesian)``
    Q block.  Callers stream this block one central atom at a time.
    """
    n_edges = len(basis_derivatives)
    if not n_edges:
        return basis_derivatives.new_zeros((0, descriptor.n_features, 3))
    one_hot = torch.nn.functional.one_hot(
        neighbor_species, num_classes=descriptor.n_species
    ).to(dtype=basis_derivatives.dtype)
    pairs = torch.triu_indices(
        descriptor.n_channels, descriptor.n_channels, device=basis_derivatives.device
    )
    pieces = []
    for angular in range(descriptor.lmax + 1):
        start, stop = angular ** 2, (angular + 1) ** 2
        # dC has shape (edge, species, radial, harmonic, Cartesian), then
        # flattens to the species-major combined-channel convention of B2.
        dcoefficients = (
            one_hot[:, :, None, None, None]
            * basis_derivatives[:, None, :, start:stop, :]
        ).reshape(n_edges, descriptor.n_channels, stop - start, 3)
        coefficient = coefficients[:, start:stop]
        derivative_gram = (
            torch.einsum("eahc,bh->eabc", dcoefficients, coefficient)
            + torch.einsum("ah,ebhc->eabc", coefficient, dcoefficients)
        )
        pieces.append(derivative_gram[:, pairs[0], pairs[1], :])
    return torch.stack(pieces, dim=2).reshape(n_edges, descriptor.n_features, 3)


def _q_chunk_state(values, values_species, inducing_descriptors, inducing_species,
                   layout, amplitude, power, chunk_size):
    """Normalize descriptor values and allocate per-inducing-chunk results."""
    inducing_unit, inducing_active, _ = _normalization_data(inducing_descriptors, 1e-8)
    values_unit, values_active, values_norm = _normalization_data(values, 1e-8)
    n_inducing, n_atoms = len(inducing_descriptors), len(values)
    chunks = []
    for start in range(0, n_inducing, chunk_size):
        stop = min(start + chunk_size, n_inducing)
        units = inducing_unit[start:stop]
        species_chunk = inducing_species[start:stop]
        dots = units @ values_unit.T
        gate = species_chunk[:, None] == values_species[None, :]
        covariance = dots.pow(power) * amplitude.square() * gate
        chunks.append({
            "species": species_chunk,
            "unit": units,
            "active": inducing_active[start:stop],
            "dots": dots,
            "energy": covariance.sum(dim=1, keepdim=True),
            "forces": (values.new_zeros((stop - start, n_atoms, 3))
                       if bool(layout.force_mask.any()) else None),
            "stresses": (values.new_zeros((stop - start, 3, 3))
                         if bool(layout.stress_mask.any()) else None),
        })
    return chunks, values_unit, values_active, values_norm


def _kernel_sensitivity_chunks(
    chunks, values_species, values_unit, values_active, values_norm,
    inducing_species, center, amplitude, power,
):
    """Yield matching sparse rows and d(kernel)/d(raw B2) for one center."""
    if not bool((inducing_species == values_species[center]).any()):
        return
    for chunk in chunks:
        rows = torch.nonzero(chunk["species"] == values_species[center], as_tuple=True)[0]
        if not len(rows):
            continue
        dot = chunk["dots"][rows, center]
        power_factor = torch.ones_like(dot) if power == 1 else dot.pow(power - 1)
        coefficient = amplitude.square() * power * power_factor / values_norm[center, 0]
        # The derivative of a normalized dot product is
        # a^2 p c^(p-1) (z_hat - c q_hat) / ||q||. Empty and unlike
        # central-species pairs have zero covariance and derivative.
        sensitivity = coefficient[:, None] * (
            chunk["unit"][rows] - dot[:, None] * values_unit[center]
        )
        active = chunk["active"][rows] & values_active[center]
        sensitivity = sensitivity * active[:, None]
        yield chunk, rows, sensitivity


def _accumulate_edge_gradient(chunk, rows, edge_gradient, first, second, vectors, volume):
    """Scatter one central atom's edge gradients into selected E/F/stress rows."""
    if chunk["forces"] is not None:
        local = edge_gradient.new_zeros((len(rows), chunk["forces"].shape[1], 3))
        local = local.index_add(1, first, edge_gradient)
        local = local.index_add(1, second, -edge_gradient)
        chunk["forces"] = chunk["forces"].index_add(0, rows, local)
    if chunk["stresses"] is not None:
        local = -torch.einsum("mec,ed->mcd", edge_gradient, vectors) / volume
        chunk["stresses"] = chunk["stresses"].index_add(0, rows, local)


def _accumulate_q_block(
    chunks, values_species, values_unit, values_active, values_norm,
    inducing_species, center, first, second, vectors, q_block,
    amplitude, power, volume,
):
    """Contract one central atom's raw-B2 Q block into chunked E/F/stress rows."""
    for chunk, rows, sensitivity in _kernel_sensitivity_chunks(
        chunks, values_species, values_unit, values_active, values_norm,
        inducing_species, center, amplitude, power,
    ):
        edge_gradient = torch.einsum("md,edc->mec", sensitivity, q_block)
        _accumulate_edge_gradient(
            chunk, rows, edge_gradient, first, second, vectors, volume,
        )


def _lambda_edge_gradient(
    descriptor, coefficients, basis_derivatives, neighbor_species, sensitivity,
):
    """Contract raw-B2 sensitivity through coefficients before edge derivatives.

    This is the Lambda contraction. For each angular channel, it maps
    d(kernel)/d(B2) onto d(kernel)/d(coefficient), then contracts directly
    with the radial/harmonic derivative on every edge.
    """
    n_rows, n_edges = len(sensitivity), len(basis_derivatives)
    pairs = torch.triu_indices(
        descriptor.n_channels, descriptor.n_channels, device=sensitivity.device
    )
    sensitivity = sensitivity.reshape(n_rows, len(pairs[0]), descriptor.lmax + 1)
    channels = (
        neighbor_species[:, None] * descriptor.n_radial
        + torch.arange(descriptor.n_radial, device=sensitivity.device)
    )
    edge_gradient = sensitivity.new_zeros((n_rows, n_edges, 3))
    for angular in range(descriptor.lmax + 1):
        # A packed diagonal B2 component contributes twice to its coefficient;
        # off-diagonal components contribute once to each coefficient channel.
        start, stop = angular ** 2, (angular + 1) ** 2
        weights = sensitivity[:, :, angular, None]
        coefficient = coefficients[:, start:stop]
        coefficient_sensitivity = sensitivity.new_zeros(
            (n_rows, descriptor.n_channels, stop - start)
        ).index_add(
            1, pairs[0], weights * coefficient[pairs[1]][None]
        ).index_add(
            1, pairs[1], weights * coefficient[pairs[0]][None]
        )
        selected = coefficient_sensitivity[:, channels, :]
        edge_gradient = edge_gradient + torch.einsum(
            "merh,erhc->mec", selected, basis_derivatives[:, :, start:stop, :]
        )
    return edge_gradient


def _finish_q_chunks(chunks, layout):
    """Select requested columns and concatenate inducing chunks."""
    results = []
    for chunk in chunks:
        columns = [chunk["energy"]] if layout.energy else []
        if chunk["forces"] is not None:
            columns.append(chunk["forces"][:, layout.force_mask])
        if chunk["stresses"] is not None:
            row, column = [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]
            columns.append(chunk["stresses"][:, row, column][:, layout.stress_mask])
        results.append(torch.cat(columns, dim=1))
    return torch.cat(results, dim=0)


def _q_observation_covariance(
    descriptor, inducing_descriptors, inducing_species, positions, cell, species,
    layout, pbc, amplitude, power, chunk_size, edges, needs_forces, needs_stress, volume,
):
    """Explicit streamed-Q implementation of one structure's K_uy block."""
    # An energy-only block has no derivative contraction, so preserve the
    # ordinary descriptor-forward cost rather than forming unused local terms.
    terms = (descriptor.local_terms(positions, cell, species, pbc, edges=edges)
             if needs_forces or needs_stress else None)
    values = (terms.values if terms is not None
              else descriptor(positions, cell, species, pbc, edges=edges))
    amplitude = torch.as_tensor(amplitude, dtype=positions.dtype, device=positions.device)
    chunks, values_unit, values_active, values_norm = _q_chunk_state(
        values, species, inducing_descriptors, inducing_species, layout,
        amplitude, power, chunk_size,
    )

    if terms is not None and len(terms.first):
        # User-provided fixed topology need not arrive in central-atom order.
        order = torch.argsort(terms.first, stable=True)
        first, second = terms.first[order], terms.second[order]
        vectors, derivatives = terms.vectors[order], terms.basis_derivatives[order]
        centers, counts = torch.unique_consecutive(first, return_counts=True)
        offset = 0
        for center, count in zip(centers.tolist(), counts.tolist()):
            edge_slice = slice(offset, offset + count)
            offset += count
            if not bool((inducing_species == species[center]).any()):
                continue
            q_block = _edge_b2_derivatives(
                descriptor, terms.coefficients[center], derivatives[edge_slice],
                species[second[edge_slice]],
            )
            _accumulate_q_block(
                chunks, species, values_unit, values_active, values_norm,
                inducing_species, center, first[edge_slice], second[edge_slice],
                vectors[edge_slice], q_block, amplitude, power, volume,
            )

    result = _finish_q_chunks(chunks, layout)
    return result + 0 * (
        positions.square().sum() + cell.square().sum()
        + inducing_descriptors.sum() + torch.as_tensor(amplitude, dtype=positions.dtype, device=positions.device)
    )


def _lambda_observation_covariance(
    descriptor, inducing_descriptors, inducing_species, positions, cell, species,
    layout, pbc, amplitude, power, chunk_size, edges, needs_forces, needs_stress, volume,
):
    """Explicit Lambda implementation of one structure's K_uy block."""
    terms = (descriptor.local_terms(positions, cell, species, pbc, edges=edges)
             if needs_forces or needs_stress else None)
    values = (terms.values if terms is not None
              else descriptor(positions, cell, species, pbc, edges=edges))
    amplitude = torch.as_tensor(amplitude, dtype=positions.dtype, device=positions.device)
    chunks, values_unit, values_active, values_norm = _q_chunk_state(
        values, species, inducing_descriptors, inducing_species, layout,
        amplitude, power, chunk_size,
    )
    if terms is not None and len(terms.first):
        order = torch.argsort(terms.first, stable=True)
        first, second = terms.first[order], terms.second[order]
        vectors, derivatives = terms.vectors[order], terms.basis_derivatives[order]
        centers, counts = torch.unique_consecutive(first, return_counts=True)
        offset = 0
        for center, count in zip(centers.tolist(), counts.tolist()):
            edge_slice = slice(offset, offset + count)
            offset += count
            for chunk, rows, sensitivity in _kernel_sensitivity_chunks(
                chunks, species, values_unit, values_active, values_norm,
                inducing_species, center, amplitude, power,
            ):
                edge_gradient = _lambda_edge_gradient(
                    descriptor, terms.coefficients[center], derivatives[edge_slice],
                    species[second[edge_slice]], sensitivity,
                )
                _accumulate_edge_gradient(
                    chunk, rows, edge_gradient, first[edge_slice], second[edge_slice],
                    vectors[edge_slice], volume,
                )
    result = _finish_q_chunks(chunks, layout)
    return result + 0 * (
        positions.square().sum() + cell.square().sum()
        + inducing_descriptors.sum() + torch.as_tensor(amplitude, dtype=positions.dtype, device=positions.device)
    )


def build_q_cache(descriptor, positions, cell, species, pbc=True, edges=None):
    """Detach the geometry-dependent B2 values and Q blocks of one structure.

    The returned cache is intended for sparse-environment updates after the
    training structure has been accepted. Rebuild it after changing positions,
    cell, topology, descriptor settings, or descriptor parameters.
    """
    _validate_geometry(positions, cell)
    descriptor._validate_species(positions, species)
    if edges is None:
        edges = neighbor_list(positions, cell, descriptor.cutoff, pbc)
    first, second, shifts = edges
    vectors = get_edge_vectors(positions, cell, first, second, shifts)
    if bool((vectors.detach().square().sum(dim=-1) == 0).any()):
        raise ValueError("B2 is undefined for coincident atoms")
    terms = descriptor.local_terms(positions, cell, species, pbc, edges=edges)
    if len(terms.first):
        order = torch.argsort(terms.first, stable=True)
        first, second, vectors = (
            terms.first[order], terms.second[order], terms.vectors[order]
        )
        derivatives = terms.basis_derivatives[order]
        centers, counts = torch.unique_consecutive(first, return_counts=True)
        edge_ptr = torch.cat((
            torch.zeros(1, dtype=torch.long, device=positions.device),
            counts.cumsum(0),
        ))
        blocks = []
        offset = 0
        for center, count in zip(centers.tolist(), counts.tolist()):
            edge_slice = slice(offset, offset + count)
            offset += count
            blocks.append(_edge_b2_derivatives(
                descriptor, terms.coefficients[center], derivatives[edge_slice],
                species[second[edge_slice]],
            ))
        q = torch.cat(blocks, dim=0)
    else:
        first, second, vectors = terms.first, terms.second, terms.vectors
        q = terms.values.new_zeros((0, descriptor.n_features, 3))
        centers = terms.first.new_empty(0)
        edge_ptr = torch.zeros(1, dtype=torch.long, device=positions.device)
    return CachedQ(
        values=terms.values.detach().clone(),
        species=species.detach().clone(),
        first=first.detach().clone(),
        second=second.detach().clone(),
        vectors=vectors.detach().clone(),
        q=q.detach().clone(),
        centers=centers.detach().clone(),
        edge_ptr=edge_ptr.detach().clone(),
        volume=torch.linalg.det(cell).abs().detach().clone(),
        n_species=descriptor.n_species,
    )


def _validate_cached_q_inputs(cache, inducing_descriptors, inducing_species, layout,
                              amplitude, power, chunk_size):
    if not isinstance(cache, CachedQ):
        raise TypeError("cache must be a CachedQ")
    if not isinstance(layout, ObservationLayout) or layout.n_atoms != cache.n_atoms:
        raise ValueError("layout must describe the cached atoms")
    if layout.device != cache.values.device:
        raise ValueError("layout and cache must share a device")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if (inducing_descriptors.ndim != 2 or inducing_descriptors.shape[1] != cache.n_features
            or inducing_descriptors.dtype != cache.values.dtype
            or inducing_descriptors.device != cache.values.device):
        raise ValueError("inducing descriptors must match the cached dtype, device, and features")
    if not bool(torch.isfinite(inducing_descriptors.detach()).all()):
        raise ValueError("inducing descriptors must be finite")
    if (inducing_species.shape != (len(inducing_descriptors),) or inducing_species.dtype != torch.long
            or inducing_species.device != cache.values.device):
        raise ValueError("inducing_species must be a torch.long vector on the cache device")
    if bool(((inducing_species < 0) | (inducing_species >= cache.n_species)).any()):
        raise ValueError("inducing species code outside the cached vocabulary")
    if not isinstance(power, int) or isinstance(power, bool) or power < 1:
        raise ValueError("power must be a positive integer")
    amplitude = torch.as_tensor(amplitude, dtype=cache.values.dtype, device=cache.values.device)
    if amplitude.ndim != 0 or not bool(torch.isfinite(amplitude.detach())):
        raise ValueError("amplitude must be a finite scalar")
    if bool(layout.stress_mask.any()) and not bool(torch.isfinite(cache.volume) & (cache.volume > 0)):
        raise ValueError("stress observations require a finite positive cached cell volume")
    return amplitude


def cached_q_observation_covariance(
    cache, inducing_descriptors, inducing_species, layout,
    amplitude=1.0, power=2, chunk_size=32,
):
    """Assemble one cached training structure's K_uy block.

    The cache is detached from training geometry and descriptor parameters.
    The returned block remains differentiable with respect to inducing
    descriptors, amplitude, and other kernel-side operations.
    """
    amplitude = _validate_cached_q_inputs(
        cache, inducing_descriptors, inducing_species, layout,
        amplitude, power, chunk_size,
    )
    if not len(inducing_descriptors) or not layout.size:
        return cache.values.new_zeros((len(inducing_descriptors), layout.size)) + 0 * (
            inducing_descriptors.sum() + amplitude
        )
    chunks, values_unit, values_active, values_norm = _q_chunk_state(
        cache.values, cache.species, inducing_descriptors, inducing_species,
        layout, amplitude, power, chunk_size,
    )
    if bool(layout.force_mask.any()) or bool(layout.stress_mask.any()):
        for center, start, stop in zip(
            cache.centers.tolist(), cache.edge_ptr[:-1].tolist(), cache.edge_ptr[1:].tolist(),
        ):
            edge_slice = slice(start, stop)
            _accumulate_q_block(
                chunks, cache.species, values_unit, values_active, values_norm,
                inducing_species, center, cache.first[edge_slice], cache.second[edge_slice],
                cache.vectors[edge_slice], cache.q[edge_slice], amplitude, power, cache.volume,
            )
    return _finish_q_chunks(chunks, layout) + 0 * (inducing_descriptors.sum() + amplitude)


def assemble_cached_q_observation_covariance(
    caches, inducing_descriptors, inducing_species, layouts,
    amplitude=1.0, power=2, chunk_size=32,
):
    """Concatenate cached-Q blocks for a sequence of training structures."""
    if len(caches) != len(layouts):
        raise ValueError("one observation layout is required per CachedQ")
    if not caches:
        if layouts:
            raise ValueError("layouts require at least one CachedQ")
        amplitude = torch.as_tensor(
            amplitude, dtype=inducing_descriptors.dtype, device=inducing_descriptors.device,
        )
        return inducing_descriptors.new_zeros((len(inducing_descriptors), 0)) + 0 * (
            inducing_descriptors.sum() + amplitude
        )
    return torch.cat([
        cached_q_observation_covariance(
            cache, inducing_descriptors, inducing_species, layout,
            amplitude=amplitude, power=power, chunk_size=chunk_size,
        )
        for cache, layout in zip(caches, layouts)
    ], dim=1)


def inducing_observation_covariance(
    descriptor, inducing_descriptors, inducing_species, positions, cell, species,
    layout, pbc=True, amplitude=1.0, power=2, chunk_size=32, edges=None,
    assembly="autograd",
):
    """Build an ``(M, layout.size)`` inducing-to-observation covariance.

    ``assembly='autograd'`` directly differentiates the vector of kernel sums.
    ``assembly='q'`` forms bounded local raw-B2 derivative blocks and contracts
    them with analytic kernel sensitivities. ``assembly='lambda'`` contracts
    analytic kernel sensitivities through B2 coefficients before the
    edge-basis derivative. All paths share the same model conventions and are
    expected to be numerically equivalent.

    Topology is built once outside derivative transforms unless trusted edges
    are supplied. Coordinate and strain Jacobians are only evaluated for
    requested observation kinds. Stress differentiates a full deformation of
    both positions and cell and is ``-d/dstrain / abs(det(cell))``; shear
    components receive no factor of two. Geometry derivatives assume unchanged
    topology and stay away from coincident atoms and the kernel's empty threshold.
    """
    _validate_geometry(positions, cell)
    if not isinstance(layout, ObservationLayout) or layout.n_atoms != len(positions):
        raise ValueError("layout must describe the supplied atoms")
    if layout.device != positions.device:
        raise ValueError("layout and geometry must share a device")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if assembly not in ("autograd", "q", "lambda"):
        raise ValueError("assembly must be 'autograd', 'q', or 'lambda'")
    if (inducing_descriptors.ndim != 2 or inducing_descriptors.dtype != positions.dtype
            or inducing_descriptors.device != positions.device):
        raise ValueError("inducing descriptors must be a matrix with the geometry dtype/device")
    if not bool(torch.isfinite(inducing_descriptors.detach()).all()):
        raise ValueError("inducing descriptors must be finite")
    if (inducing_species.shape != (len(inducing_descriptors),) or inducing_species.dtype != torch.long
            or inducing_species.device != positions.device):
        raise ValueError("inducing_species must be a torch.long vector on the geometry device")
    # Validate fixed species even when there are no references or observations.
    descriptor._validate_species(positions, species)
    if inducing_descriptors.shape[1] != descriptor.n_features:
        raise ValueError("inducing descriptors do not match the descriptor feature count")
    if bool(((inducing_species < 0) | (inducing_species >= descriptor.n_species)).any()):
        raise ValueError("inducing species code outside the fixed global vocabulary")
    if not isinstance(power, int) or isinstance(power, bool) or power < 1:
        raise ValueError("power must be a positive integer")
    amplitude = torch.as_tensor(amplitude, dtype=positions.dtype, device=positions.device)
    if amplitude.ndim != 0 or not bool(torch.isfinite(amplitude)):
        raise ValueError("amplitude must be a finite scalar")
    needs_forces = bool(layout.force_mask.any())
    needs_stress = bool(layout.stress_mask.any())
    volume = torch.linalg.det(cell).abs() if needs_stress else None
    if needs_stress and not bool(torch.isfinite(volume) & (volume > 0)):
        raise ValueError("stress observations require a finite positive cell volume")
    if not len(inducing_descriptors) or not layout.size:
        return positions.new_zeros((len(inducing_descriptors), layout.size)) + 0 * (
            positions.sum() + cell.sum() + inducing_descriptors.sum() + amplitude)
    if edges is None:
        edges = neighbor_list(positions, cell, descriptor.cutoff, pbc)
        vectors = get_edge_vectors(positions, cell, *edges)
        if bool((vectors.detach().square().sum(dim=-1) == 0).any()):
            raise ValueError("B2 is undefined for coincident atoms")
    if assembly == "q":
        return _q_observation_covariance(
            descriptor, inducing_descriptors, inducing_species, positions, cell, species,
            layout, pbc, amplitude, power, chunk_size, edges, needs_forces, needs_stress, volume,
        )
    if assembly == "lambda":
        return _lambda_observation_covariance(
            descriptor, inducing_descriptors, inducing_species, positions, cell, species,
            layout, pbc, amplitude, power, chunk_size, edges, needs_forces, needs_stress, volume,
        )
    identity = torch.eye(3, dtype=positions.dtype, device=positions.device)
    zero_strain = torch.zeros_like(cell)
    row, column = [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]
    chunks = []
    for start in range(0, len(inducing_descriptors), chunk_size):
        inducing = inducing_descriptors[start:start + chunk_size]
        central_species = inducing_species[start:start + chunk_size]

        def energy_columns(coordinates, lattice):
            values = descriptor(coordinates, lattice, species, pbc, edges=edges)
            return normalized_dot_product(
                inducing, values, central_species, species, amplitude=amplitude, power=power,
            ).sum(dim=1)

        if needs_forces and needs_stress:
            def deformed(coordinates, strain):
                deformation = identity + strain
                energy = energy_columns(coordinates @ deformation.T, cell @ deformation.T)
                return energy, energy

            (position_jacobian, strain_jacobian), energy = torch.func.jacrev(
                deformed, argnums=(0, 1), has_aux=True,
            )(positions, zero_strain)
        elif needs_forces:
            def positioned(coordinates):
                energy = energy_columns(coordinates, cell)
                return energy, energy

            position_jacobian, energy = torch.func.jacrev(positioned, has_aux=True)(positions)
        elif needs_stress:
            def strained(strain):
                deformation = identity + strain
                energy = energy_columns(positions @ deformation.T, cell @ deformation.T)
                return energy, energy

            strain_jacobian, energy = torch.func.jacrev(strained, has_aux=True)(zero_strain)
        else:
            energy = energy_columns(positions, cell)
        columns = [energy[:, None]] if layout.energy else []
        if needs_forces:
            columns.append(-position_jacobian[:, layout.force_mask])
        if needs_stress:
            columns.append((-strain_jacobian[:, row, column] / volume)[:, layout.stress_mask])
        chunks.append(torch.cat(columns, dim=1))
    return torch.cat(chunks, dim=0)


def assemble_observation_covariance(
    descriptor, inducing_descriptors, inducing_species, batch, layouts,
    amplitude=1.0, power=2, chunk_size=32, assembly="autograd",
):
    """Concatenate per-structure selected columns from a ``StructureBatch``.

    Geometry is processed one structure at a time. As with the single-structure
    operator, a returned differentiable matrix retains its computation graph;
    release it after fitting unless further derivatives are needed.
    """
    if len(layouts) != batch.num_structures:
        raise ValueError("one observation layout is required per structure")
    matrices = []
    for index, layout in enumerate(layouts):
        start, end = int(batch.ptr[index]), int(batch.ptr[index + 1])
        matrices.append(inducing_observation_covariance(
            descriptor, inducing_descriptors, inducing_species,
            batch.positions[start:end], batch.cells[index], batch.species[start:end],
            layout, pbc=batch.pbc[index], amplitude=amplitude, power=power,
            chunk_size=chunk_size, assembly=assembly,
        ))
    if matrices:
        return torch.cat(matrices, dim=1)
    amplitude = torch.as_tensor(amplitude, dtype=batch.positions.dtype, device=batch.positions.device)
    return batch.positions.new_zeros((len(inducing_descriptors), 0)) + 0 * (
        batch.positions.sum() + batch.cells.sum() + inducing_descriptors.sum() + amplitude)
