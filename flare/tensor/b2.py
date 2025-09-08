import torch
import numpy as np
import time
import math
from e3nn.o3 import spherical_harmonics
from flare.tensor.neighbors import get_neighbors_auto


def get_edge_dist(pos_tensor, cell_tensor, first_index, second_index, shifts):
    shift_tensor = shifts.to(dtype=pos_tensor.dtype)
    edge_vec = torch.index_select(pos_tensor, 0, second_index) - torch.index_select(
        pos_tensor, 0, first_index
    )
    edge_shift = shift_tensor @ cell_tensor
    edge_vec = edge_vec + edge_shift
    sqrt_jitter = 1e-10
    edge_dist_sq = torch.sum(edge_vec**2, -1)
    edge_dist = torch.sqrt(edge_dist_sq + sqrt_jitter)

    return edge_vec, edge_dist


def compute_b2(
    positions: np.ndarray,
    cell: np.ndarray,
    numbers_coded: np.ndarray,
    cutoff: float,
    n_radial=8,
    lmax=3,
    basis_type="bessel",
    cutoff_type="quadratic",
):
    # make tensors
    strain_tensor = torch.eye(3, dtype=torch.float32, requires_grad=True)
    init_pos = torch.tensor(positions, dtype=torch.float32, requires_grad=True)
    init_cell = torch.tensor(cell, dtype=torch.float32)
    numbers = torch.tensor(numbers_coded, dtype=torch.int64)
    n_atoms = len(positions)
    n_harm = (lmax + 1) * (lmax + 1)

    # apply strain
    pos_tensor = init_pos @ strain_tensor.T
    cell_tensor = init_cell @ strain_tensor.T

    # get edges
    first_index, second_index, shifts = get_neighbors_auto(init_cell, init_pos, cutoff)
    edge_vec, edge_dist = get_edge_dist(
        pos_tensor, cell_tensor, first_index, second_index, shifts
    )
    n_edges = len(edge_dist)
    num_2 = numbers[second_index]

    # compute radial basis functions
    radial_vals = compute_radial_basis(
        edge_dist,
        n_radial,
        cutoff,
        basis_type,
        cutoff_type,
    )

    # add species index
    n_species = len(set(numbers_coded))
    rs_vals = torch.zeros(n_edges, n_radial, n_species)
    num_exp = num_2.unsqueeze(-1).expand(-1, n_radial).unsqueeze(-1)
    rs_vals.scatter_(-1, num_exp, radial_vals.unsqueeze(-1))

    # compute spherical harmonics
    harmonics = spherical_harmonics(
        list(range(lmax + 1)),
        edge_vec,
        normalize=True,
    )

    # compute single bond descriptors
    rs_exp = rs_vals.unsqueeze(-1)
    harm_exp = harmonics.unsqueeze(-2).unsqueeze(-2)
    single_bond = rs_exp * harm_exp

    # reduce edges
    sb_reduced = torch.zeros(n_atoms, n_radial, n_species, n_harm)
    ind_exp = first_index.view(-1, 1, 1, 1).expand(-1, n_radial, n_species, n_harm)
    sb_reduced.scatter_add_(0, ind_exp, single_bond)

    # broadcast radial and species indices
    sb_broad = sb_reduced.unsqueeze(1).unsqueeze(1) * sb_reduced.unsqueeze(3).unsqueeze(
        3
    )

    # reduce m
    b2_vals = reduce_lm(sb_broad)

    # normalize b2
    dims = tuple(range(1, b2_vals.dim()))
    norm = torch.norm(b2_vals, p=2, dim=dims, keepdim=True)
    b2_norm = b2_vals / norm
    b2_norm[torch.isnan(b2_norm)] = 0
    b2_flat = b2_norm.reshape(n_atoms, -1)

    return b2_flat, init_pos, strain_tensor


def compute_radial_basis(
    edge_dist: torch.Tensor,
    n_radial: int,
    cutoff: float,
    basis_type: str = "bessel",
    cutoff_type: str = "quadratic",
) -> torch.Tensor:
    """Compute radial basis functions with a specified basis type and cutoff
    envelope."""

    if basis_type == "bessel":
        return bessel_radial(edge_dist, n_radial, cutoff, cutoff_type)
    elif basis_type == "chebyshev":
        return chebyshev_radial(edge_dist, n_radial, cutoff, cutoff_type=cutoff_type)
    else:
        raise ValueError(f"Unsupported basis type: {basis_type}")


def chebyshev_radial(
    edge_dist: torch.Tensor,
    n_radial: int,
    cutoff: float,
    cutoff_type: str = "quadratic",
    r_min: float = 0.0,
) -> torch.Tensor:
    """Compute Chebyshev radial basis functions of the first kind, scaled to
    [r_min, cutoff], and modulated by a cutoff envelope."""

    # TODO: Check the convention here - Chebyshev is defined over [-1 ,1], but
    # the C++ implementation (I think) evaluates over [0, 1].
    x = (edge_dist - r_min) / (cutoff - r_min)
    # x = ((edge_dist - r_min) / (cutoff - r_min)) * 2 - 1
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)  # for numerical stability inside arccos

    # Compute Chebyshev basis values using cosine formula
    theta = torch.acos(x)  # (N_edges,)
    n = torch.arange(n_radial, device=edge_dist.device, dtype=edge_dist.dtype)
    cheb_vals = torch.cos(theta.unsqueeze(-1) * n)

    # Apply cutoff envelope
    envelope = compute_cutoff(edge_dist, cutoff, cutoff_type)

    return cheb_vals * envelope


def bessel_radial(
    edge_dist: torch.Tensor,
    n_radial: int,
    cutoff: float,
    cutoff_type: str = "quadratic",
) -> torch.Tensor:
    """Compute radial basis functions using sine-weighted Bessel functions
    modulated by a quadratic cutoff envelope."""

    bessel_weights = torch.linspace(start=1.0, end=n_radial, steps=n_radial) * math.pi
    bessel_num = torch.sin(bessel_weights * edge_dist.unsqueeze(-1) / cutoff)
    bessel_prefactor = 2 / cutoff
    bessel_vals = bessel_prefactor * (bessel_num / edge_dist.unsqueeze(-1))
    envelope = compute_cutoff(edge_dist, cutoff, cutoff_type)

    return bessel_vals * envelope


def compute_cutoff(
    edge_dist: torch.Tensor,
    cutoff: float,
    cutoff_type: str = "quadratic",
) -> torch.Tensor:
    """Dispatch to appropriate cutoff function."""

    if cutoff_type == "quadratic":
        return quadratic_cutoff(edge_dist, cutoff).unsqueeze(-1)
    elif cutoff_type == "cosine":
        return cosine_cutoff(edge_dist, cutoff).unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported cutoff type: {cutoff_type}")


def quadratic_cutoff(edge_dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Quadratic envelope: (cutoff - r)^2"""

    return (cutoff - edge_dist).clamp(min=0.0) ** 2


def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Cosine envelope: 0.5 * (cos(pi * r / cutoff) + 1)"""

    result = 0.5 * (torch.cos(math.pi * edge_dist / cutoff) + 1.0)
    result[edge_dist > cutoff] = 0.0

    return result


def reduce_lm(lm_tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor with l and m indices to a rotationally invariant tensor
    with only l indices."""

    n_harm = lm_tensor.shape[-1]
    lmax = math.sqrt(n_harm) - 1
    assert round(lmax) == lmax, "Input tensor has incorrect size"

    l_tensor = torch.zeros(
        *lm_tensor.shape[:-1],
        int(lmax) + 1,
    )

    m_indices = torch.arange(n_harm)
    l_indices = torch.sqrt(m_indices).long()
    l_indices = l_indices.unsqueeze(0).expand(*lm_tensor.shape[:-1], -1)
    l_tensor.scatter_add_(dim=-1, index=l_indices, src=lm_tensor)

    return l_tensor


if __name__ == "__main__":
    from ase.build import bulk

    np.random.seed(1)

    crystal = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    supercell = crystal.repeat((5, 5, 5))
    cell = supercell.cell[:]
    positions = supercell.positions
    numbers = supercell.numbers
    species_map = {11: 0, 17: 1}
    n_species = len(species_map)
    numbers_coded = np.array([species_map[num] for num in numbers])

    n_radial = 12
    cutoff = 6.0
    lmax = 3
    basis_type = "chebyshev"

    for n in range(20):
        time0 = time.time()
        b2_flat, init_pos, strain_tensor = compute_b2(
            positions,
            cell,
            numbers_coded,
            cutoff,
            n_radial,
            lmax,
            basis_type,
        )
        time1 = time.time()
        print(time1 - time0)

    print(b2_flat)
