"""Mean prediction by differentiating one weighted scalar energy."""

from dataclasses import dataclass
from typing import Optional

import torch

from .kernels import normalized_dot_product
from .neighbors import get_edge_vectors, neighbor_list


@dataclass(frozen=True)
class MeanPrediction:
    """Total energy, optional forces, and optional stress in native ordering."""

    energy: torch.Tensor
    forces: Optional[torch.Tensor]
    stress: Optional[torch.Tensor]


def predict_mean_efs(
    descriptor,
    inducing_descriptors,
    inducing_species,
    alpha,
    positions,
    cell,
    species,
    pbc=True,
    *,
    amplitude=1.0,
    power=2,
    atomic_offsets=None,
    forces=True,
    stress=False,
    create_graph=False,
    edges=None,
):
    """Predict with a fixed posterior using one scalar-energy reverse pass.

    ``alpha`` contains inducing coefficients, as returned by ``fit_sparse_gp``.
    References, coefficients, amplitude, and atomic offsets are held fixed:
    coordinate derivatives must not differentiate through fitting or reference
    selection. Energy includes the sum of optional per-species atomic offsets.

    Stress is ``-dE/dstrain / volume`` in ``xx,xy,xz,yy,yz,zz`` order; use
    ``native_stress_to_ase`` at an ASE boundary. Stress requires nonzero volume.
    Outputs are detached by default. ``create_graph=True`` preserves geometry
    derivatives of the outputs, for example a force Hessian. Precomputed edges
    must satisfy the B2 trusted-topology contract.
    """
    if alpha.ndim != 1 or alpha.shape[0] != inducing_descriptors.shape[0]:
        raise ValueError("alpha must have one coefficient per inducing environment")
    if alpha.dtype != positions.dtype or alpha.device != positions.device:
        raise ValueError("alpha and positions must share dtype and device")
    if not bool(torch.isfinite(alpha.detach()).all()):
        raise ValueError("alpha must be finite")
    if not isinstance(forces, bool) or not isinstance(stress, bool):
        raise TypeError("forces and stress must be booleans")
    if (inducing_species.shape != (len(inducing_descriptors),)
            or inducing_species.dtype != torch.long
            or inducing_species.device != positions.device):
        raise ValueError("inducing_species must be a torch.long vector on the geometry device")
    if bool(((inducing_species < 0) | (inducing_species >= descriptor.n_species)).any()):
        raise ValueError("inducing species code outside the fixed global vocabulary")
    if edges is None:
        edges = neighbor_list(positions, cell, descriptor.cutoff, pbc)
        vectors = get_edge_vectors(positions, cell, *edges)
        if bool((vectors.detach().square().sum(-1) == 0).any()):
            raise ValueError("B2 is undefined for coincident atoms")
    if stress:
        volume = torch.linalg.det(cell).abs()
        if not bool(torch.isfinite(volume.detach())) or float(volume.detach()) <= 0:
            raise ValueError("stress requires a finite, positive cell volume")

    fixed_references = inducing_descriptors.detach()
    fixed_alpha = alpha.detach()
    fixed_amplitude = torch.as_tensor(
        amplitude, dtype=positions.dtype, device=positions.device
    ).detach()
    if fixed_amplitude.ndim != 0 or not bool(torch.isfinite(fixed_amplitude)):
        raise ValueError("amplitude must be a finite scalar")
    energy_offset = positions.new_zeros(())
    if atomic_offsets is not None:
        offsets = torch.as_tensor(
            atomic_offsets, dtype=positions.dtype, device=positions.device
        ).detach()
        if offsets.shape != (descriptor.n_species,) or not bool(torch.isfinite(offsets).all()):
            raise ValueError("atomic_offsets must be a finite vector over the global species vocabulary")
        descriptor._validate_species(positions, species)
        energy_offset = offsets[species].sum()

    def energy_function(coordinates, strain):
        if stress:
            deformation = torch.eye(3, dtype=coordinates.dtype, device=coordinates.device) + strain
            coordinates = coordinates @ deformation.T
            lattice = cell @ deformation.T
        else:
            lattice = cell
        values = descriptor(coordinates, lattice, species, pbc, edges=edges)
        columns = normalized_dot_product(
            fixed_references, values, inducing_species, species,
            amplitude=fixed_amplitude, power=power,
        ).sum(dim=1)
        return columns @ fixed_alpha + energy_offset

    strain = torch.zeros_like(cell)
    force_values, stress_values = None, None
    if forces and stress:
        (position_gradient, strain_gradient), energy = torch.func.grad_and_value(
            energy_function, argnums=(0, 1)
        )(positions, strain)
        force_values = -position_gradient
    elif forces:
        position_gradient, energy = torch.func.grad_and_value(
            energy_function, argnums=0
        )(positions, strain)
        force_values = -position_gradient
    elif stress:
        strain_gradient, energy = torch.func.grad_and_value(
            energy_function, argnums=1
        )(positions, strain)
    else:
        energy = energy_function(positions, strain)
    if stress:
        stress_values = -strain_gradient[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]] / volume
    if not create_graph:
        energy = energy.detach()
        force_values = None if force_values is None else force_values.detach()
        stress_values = None if stress_values is None else stress_values.detach()
    return MeanPrediction(energy, force_values, stress_values)
