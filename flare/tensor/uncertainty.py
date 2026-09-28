"""Exact prior diagonals for selected energy, force, and stress observations.

The two arguments of the structure kernel are differentiated independently.
For a normalized local descriptor ``u_i``, the mixed derivative of
``(u_i @ u_j)**power`` can be evaluated from descriptor directional derivatives
without constructing an observation-by-observation covariance matrix.
"""

import torch

from .b2 import B2
from .kernels import _normalization_data
from .lambda_cache import (
    build_grouped_lambda_cache, cached_grouped_lambda_observation_covariance,
)
from .neighbors import get_edge_vectors, neighbor_list
from .observations import (
    ObservationLayout, _edge_b2_derivatives, inducing_observation_covariance,
)
from .linalg import SparsePosterior
from .structures import _validate_geometry


_STRESS_ROWS = (0, 0, 0, 1, 1, 2)
_STRESS_COLUMNS = (0, 1, 2, 1, 2, 2)


def _mixed_directional_variance(unit, derivative, gram, species_gate, power):
    """Mixed derivative and an absolute-sum scale for roundoff checks."""
    derivative_gram = derivative @ derivative.T
    result = gram.pow(power - 1) * derivative_gram
    if power > 1:
        cross = derivative @ unit.T
        result = result + (power - 1) * gram.pow(power - 2) * cross * cross.T
    contributions = power * result * species_gate
    return contributions.sum(), contributions.abs().sum()


def _check_variance(value, scale):
    tolerance = 128 * torch.finfo(value.dtype).eps * scale.detach().abs()
    if bool(value.detach() < -tolerance):
        raise RuntimeError("prior variance is negative beyond floating-point roundoff")
    return value.clamp_min(0)


def prior_observation_variance(
    descriptor, positions, cell, species, layout, pbc=True,
    amplitude=1.0, power=2, edges=None,
):
    """Return exact latent prior variances in ``ObservationLayout`` order.

    Output order is total energy, selected atom-major Cartesian forces, then
    selected native stresses ``[xx,xy,xz,yy,yz,zz]``. Forces are derivatives
    of the total local-energy sum; stresses deform positions and cell together
    and divide by ``abs(det(cell))``. No observation noise is included.

    Local edge-basis derivatives are built once, then accumulated into only
    the affected center descriptors for each force component. Stress uses the
    same edge derivatives. This avoids reevaluating B2 per output and never
    forms a full observation covariance or feature outer product. Topology is
    fixed during differentiation; supplied edges follow B2's trusted-topology
    contract. Geometry derivatives are defined away from topology changes and
    the normalized kernel's empty-environment threshold.
    """
    if not isinstance(descriptor, B2):
        raise TypeError("descriptor must be a B2")
    _validate_geometry(positions, cell)
    descriptor._validate_species(positions, species)
    if not isinstance(layout, ObservationLayout) or layout.n_atoms != len(positions):
        raise ValueError("layout must describe the supplied atoms")
    if layout.device != positions.device:
        raise ValueError("layout and geometry must share a device")
    if not isinstance(power, int) or isinstance(power, bool) or power < 1:
        raise ValueError("power must be a positive integer")
    amplitude = torch.as_tensor(amplitude, dtype=positions.dtype, device=positions.device)
    if amplitude.ndim != 0 or not bool(torch.isfinite(amplitude.detach())):
        raise ValueError("amplitude must be a finite scalar")

    needs_stress = bool(layout.stress_mask.any())
    volume = torch.linalg.det(cell).abs() if needs_stress else None
    if needs_stress and not bool(torch.isfinite(volume.detach()) & (volume.detach() > 0)):
        raise ValueError("stress observations require a finite positive cell volume")
    if not layout.size:
        return positions.new_zeros(0) + 0 * (positions.sum() + cell.sum() + amplitude)

    if edges is None:
        edges = neighbor_list(positions, cell, descriptor.cutoff, pbc)
        vectors = get_edge_vectors(positions, cell, *edges)
        if bool((vectors.detach().square().sum(dim=-1) == 0).any()):
            raise ValueError("B2 is undefined for coincident atoms")

    needs_derivatives = bool(layout.force_mask.any()) or needs_stress
    terms = (descriptor.local_terms(positions, cell, species, pbc, edges=edges)
             if needs_derivatives else None)
    raw = (terms.values if terms is not None
           else descriptor(positions, cell, species, pbc, edges=edges))
    unit, active, norms = _normalization_data(raw, 1e-8)
    gram = unit @ unit.T
    species_gate = species[:, None] == species[None, :]
    scale = amplitude.square()
    results = []
    if layout.energy:
        energy = scale * (gram.pow(power) * species_gate).sum()
        results.append(_check_variance(energy, scale * gram.abs().pow(power).sum()))

    if needs_derivatives:
        # Every edge derivative belongs to one central environment. A position
        # perturbation affects that center through its own edges and through
        # edges whose neighbor is the perturbed atom. Keep only those centers
        # for each atom instead of a dense (atom, center, feature, xyz) array.
        force_terms = [[] for _ in range(len(positions))] if bool(layout.force_mask.any()) else None
        stress_terms = [] if needs_stress else None
        for center in range(len(positions)):
            selected = torch.nonzero(terms.first == center, as_tuple=True)[0]
            second = terms.second[selected]
            edge_raw = _edge_b2_derivatives(
                descriptor, terms.coefficients[center],
                terms.basis_derivatives[selected], species[second],
            )
            center_unit = unit[center]
            center_norm = norms[center, 0]
            center_active = active[center]
            if force_terms is not None and len(selected):
                affected = torch.unique(torch.cat((second, second.new_tensor([center]))),
                                        sorted=True)
                local = edge_raw.new_zeros((len(affected), descriptor.n_features, 3))
                local = local.index_add(0, torch.searchsorted(affected, second), edge_raw)
                center_row = torch.searchsorted(affected, second.new_tensor([center]))
                local = local.index_add(0, center_row, -edge_raw.sum(dim=0, keepdim=True))
                projection = torch.einsum("f,kfc->kc", center_unit, local)
                directional = ((local - center_unit[None, :, None]
                                * projection[:, None, :]) / center_norm) * center_active
                for atom, derivative in zip(affected.tolist(), directional.unbind(0)):
                    force_terms[atom].append((center, derivative))
            if stress_terms is not None:
                edge_vectors = terms.vectors[selected]
                raw_stress = torch.einsum("efc,ed->fcd", edge_raw, edge_vectors)
                projection = torch.einsum("f,fcd->cd", center_unit, raw_stress)
                directional = ((raw_stress - center_unit[:, None, None]
                                * projection[None, :, :]) / center_norm) * center_active
                stress_terms.append(directional)

        for coordinate in torch.nonzero(layout.force_mask.reshape(-1), as_tuple=True)[0].tolist():
            atom, component = divmod(coordinate, 3)
            affected = force_terms[atom]
            if not affected:
                results.append(positions.new_zeros(()) + 0 * (raw.sum() + scale))
                continue
            centers = torch.tensor([item[0] for item in affected],
                                   dtype=torch.long, device=positions.device)
            derivative = torch.stack([item[1][:, component] for item in affected])
            variance, absolute_sum = _mixed_directional_variance(
                unit[centers], derivative, gram[centers][:, centers],
                species_gate[centers][:, centers], power,
            )
            results.append(_check_variance(scale * variance, scale * absolute_sum))

        if needs_stress:
            all_stress = (torch.stack(stress_terms) if stress_terms else
                          raw.new_zeros((0, descriptor.n_features, 3, 3)) + 0 * raw.sum())
            for component in torch.nonzero(layout.stress_mask, as_tuple=True)[0].tolist():
                derivative = all_stress[:, :, _STRESS_ROWS[component],
                                        _STRESS_COLUMNS[component]]
                variance, absolute_sum = _mixed_directional_variance(
                    unit, derivative, gram, species_gate, power,
                )
                stress_scale = scale / volume.square()
                results.append(_check_variance(stress_scale * variance,
                                               stress_scale * absolute_sum))
    return torch.stack(results)


def predict_variance_efs(
    posterior, descriptor, inducing_descriptors, inducing_species,
    positions, cell, species, pbc=True, *, variance_type="SOR", layout=None,
    amplitude=1.0, power=2, chunk_size=32, assembly="auto", edges=None,
):
    """Predict latent SOR or DTC variances for all E/F/stress outputs.

    With ``layout=None``, return the native ``variance_efs`` order:
    ``[energy, atom-major xyz forces, xx,xy,xz,yy,yz,zz stresses]``.
    An explicit ``ObservationLayout`` selects components in the same order.
    The result has one variance per selected output and excludes observation
    noise. Use ``native_stress_to_ase`` only for mean stresses; variances do
    not change sign, but ASE ordering still differs.

    The inducing block is assembled once. DTC additionally computes exact
    prior diagonals by differentiating both arguments of the structure kernel.
    ``assembly='auto'`` uses a short-lived grouped Lambda cache for fixed
    geometry and transient Lambda assembly when geometry requires gradients.
    The grouped cache retains gradients through inducing descriptors and
    amplitude, but not through geometry. Either path can be selected
    explicitly. Supplied edges follow the trusted-topology contract.
    """
    if not isinstance(posterior, SparsePosterior):
        raise TypeError("posterior must be a SparsePosterior")
    if not isinstance(descriptor, B2):
        raise TypeError("descriptor must be a B2")
    _validate_geometry(positions, cell)
    descriptor._validate_species(positions, species)
    if variance_type not in ("SOR", "DTC"):
        raise ValueError("variance_type must be 'SOR' or 'DTC'")
    if layout is None:
        layout = ObservationLayout.full(len(positions), device=positions.device)
    if edges is None:
        edges = neighbor_list(positions, cell, descriptor.cutoff, pbc)
        vectors = get_edge_vectors(positions, cell, *edges)
        if bool((vectors.detach().square().sum(dim=-1) == 0).any()):
            raise ValueError("B2 is undefined for coincident atoms")
    if assembly == "auto":
        assembly = ("lambda" if torch.is_grad_enabled()
                    and (positions.requires_grad or cell.requires_grad) else "grouped")
    if assembly == "grouped":
        cache = build_grouped_lambda_cache(
            descriptor, positions, cell, species, pbc=pbc, edges=edges,
        )
        block = cached_grouped_lambda_observation_covariance(
            cache, inducing_descriptors, inducing_species, layout,
            amplitude=amplitude, power=power, sparse_chunk=chunk_size,
        )
    else:
        block = inducing_observation_covariance(
            descriptor, inducing_descriptors, inducing_species, positions, cell,
            species, layout, pbc=pbc, amplitude=amplitude, power=power,
            chunk_size=chunk_size, edges=edges, assembly=assembly,
        )
    if variance_type == "SOR":
        return posterior.sor_variance(block)
    prior = prior_observation_variance(
        descriptor, positions, cell, species, layout, pbc=pbc,
        amplitude=amplitude, power=power, edges=edges,
    )
    return posterior.dtc_variance(block, prior)
