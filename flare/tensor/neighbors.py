import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list
from ase.geometry import complete_cell, minkowski_reduce
import time
from typing import Tuple, Union
from itertools import product


def get_neighbors_ase(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    pbc: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the neighbor list for a set of atoms using periodic boundary
    conditions."""

    pbc = (pbc,) * 3
    first_index, second_index, shifts = primitive_neighbor_list(
        "ijS",
        pbc,
        cell,
        positions,
        cutoff=cutoff,
        self_interaction=True,
        use_scaled_positions=False,
    )
    bad_edge = first_index == second_index
    bad_edge &= np.all(shifts == 0, axis=1)
    keep_edge = ~bad_edge
    first_index = first_index[keep_edge]
    second_index = second_index[keep_edge]
    shifts = shifts[keep_edge]

    return first_index, second_index, shifts


def get_box_heights(cell: torch.Tensor) -> Tuple[float, float, float]:
    a, b, c = cell.unbind(dim=-2)

    def height(v, w1, w2):
        cross = torch.cross(w1, w2, dim=-1)
        volume = torch.abs((v * cross).sum(dim=-1))
        base_area = torch.linalg.norm(cross, dim=-1)
        return volume / base_area

    h_a = height(a, b, c)
    h_b = height(b, a, c)
    h_c = height(c, a, b)

    return h_a, h_b, h_c


def get_min_box_height(cell: torch.Tensor) -> float:
    h_a, h_b, h_c = get_box_heights(cell)
    return torch.minimum(torch.minimum(h_a, h_b), h_c)


def get_supercell_containing_rcut(
    cell: torch.Tensor, rcut: Union[float, torch.Tensor]
) -> torch.Tensor:
    """For each unit cell in `cell`, return the minimum (n_a, n_b, n_c) such
    that every point within `rcut` of the central cell is covered by the
    supercell formed by repeating the basis vectors that many times."""

    rcut = torch.as_tensor(rcut, dtype=cell.dtype, device=cell.device)
    h_a, h_b, h_c = get_box_heights(cell)
    n_a = torch.ceil(rcut / h_a).to(torch.int64)
    n_b = torch.ceil(rcut / h_b).to(torch.int64)
    n_c = torch.ceil(rcut / h_c).to(torch.int64)

    return torch.stack((n_a, n_b, n_c), dim=-1)


def wrap_positions(cells: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Wrap `positions` into periodically-repeated unit cells."""

    inv_cells = torch.linalg.inv(cells)
    frac = torch.matmul(positions, inv_cells)
    shifts = torch.floor(frac)
    frac_wrapped = frac - shifts
    wrapped = torch.matmul(frac_wrapped, cells)

    return wrapped, shifts


def generate_supercell_positions(
    cell: torch.Tensor, positions: torch.Tensor, repetitions: torch.Tensor
) -> torch.Tensor:
    """Generate supercell atomic positions by repeating a single unit cell in
    all directions."""

    reps = repetitions.int()
    shifts = (
        torch.stack(
            torch.meshgrid(
                torch.arange(-reps[0], reps[0] + 1),
                torch.arange(-reps[1], reps[1] + 1),
                torch.arange(-reps[2], reps[2] + 1),
                indexing="ij",
            ),
            dim=-1,
        )
        .reshape(-1, 3)
        .to(dtype=cell.dtype, device=cell.device)
    )
    displacements = shifts @ cell
    base = positions.unsqueeze(0).expand(len(displacements), -1, -1)
    supercell_positions = base + displacements.unsqueeze(1)
    M = positions.shape[0]
    atom_indices = (
        torch.arange(M, device=positions.device, dtype=torch.long)
        .unsqueeze(0)
        .expand(len(displacements), -1)
        .reshape(-1)
    )
    shifts_all = shifts.repeat_interleave(M, dim=0)

    return supercell_positions.reshape(-1, 3), atom_indices, shifts_all


def get_neighbors_brute_force(
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a neighbor list under periodic boundary conditions."""

    wrapped_positions, init_shifts = wrap_positions(cell, positions)
    repetitions = get_supercell_containing_rcut(cell, cutoff)
    supercell_pos, atom_indices, shifts = generate_supercell_positions(
        cell, wrapped_positions, repetitions
    )
    M = wrapped_positions.shape[0]

    # get pairwise displacements
    disp = supercell_pos.unsqueeze(0) - wrapped_positions.unsqueeze(1)
    dist2 = (disp**2).sum(dim=-1)
    within_cutoff = dist2 <= cutoff * cutoff

    # eliminate self-interaction at zero shift
    same_atom = atom_indices.unsqueeze(0) == torch.arange(
        M, device=wrapped_positions.device
    ).unsqueeze(1)
    zero_distance = dist2 == 0.0
    within_cutoff &= ~(same_atom & zero_distance)

    # extract edges
    pairs = torch.nonzero(within_cutoff, as_tuple=False)
    first_index = pairs[:, 0]
    img_index_j = pairs[:, 1]
    second_index = atom_indices[img_index_j]
    shift_vec = (
        shifts[img_index_j] - init_shifts[second_index] + init_shifts[first_index]
    )

    return first_index, second_index, shift_vec


def get_neighbors_minimum_image(
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Neighbour list via minimum-image convention."""

    rcut2 = cutoff * cutoff
    cell_inv = torch.linalg.inv(cell)
    frac = positions @ cell_inv
    d_frac = frac.unsqueeze(1) - frac.unsqueeze(0)
    shift = torch.round(d_frac).to(torch.int64)
    d_frac -= shift
    d_cart = torch.matmul(d_frac, cell)
    dist2 = (d_cart**2).sum(-1)
    mask = dist2 <= rcut2
    mask.fill_diagonal_(False)
    first, second = torch.nonzero(mask, as_tuple=True)
    shift_vec = shift[first, second]

    return first, second, shift_vec


@torch.no_grad()
def mic_safe(cell: torch.Tensor, r_cut: float, margin: float = 1e-8) -> bool:
    """Check if the minimum image convention can be safely applied."""

    h_min = get_min_box_height(cell)

    return r_cut < 0.5 * h_min - margin


def get_neighbors_auto(
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
):
    """Uses the minimum-image algorithm when safe, otherwise falls back on
    brute force supercell generator."""

    if mic_safe(cell, cutoff):
        return get_neighbors_minimum_image(cell, positions, cutoff)
    else:
        return get_neighbors_brute_force(cell, positions, cutoff)


if __name__ == "__main__":
    from ase.build import bulk

    np.random.seed(1)
    torch.manual_seed(1)

    # NaCl crystal (1000 atoms):
    crystal = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    supercell = crystal.repeat((5, 5, 5))
    cell = torch.tensor(supercell.cell[:])
    positions = torch.tensor(supercell.positions)
    cutoff = 6.0

    # # Random cell:
    # n_frames = 1
    # n_atoms = 100
    # cell = 5 * torch.randn(3, 3)
    # random_positions = torch.randn(n_atoms, 3)
    # positions, _ = wrap_positions(cell, random_positions)
    # cutoff = 1.0

    print("timing torch brute force implementation:")
    for n in range(10):
        time0 = time.time()
        first_index, second_index, shift_vec = get_neighbors_brute_force(
            cell, positions, cutoff
        )
        time1 = time.time()
        print(time1 - time0)

    print(len(first_index))

    print("timing torch minimum image implementation:")
    for n in range(10):
        time0 = time.time()
        first_index, second_index, shift_vec = get_neighbors_minimum_image(
            cell, positions, cutoff
        )
        time1 = time.time()
        print(time1 - time0)
    print(len(first_index))

    print("timing torch auto implementation:")
    for n in range(10):
        time0 = time.time()
        first_index, second_index, shift_vec = get_neighbors_auto(
            cell, positions, cutoff
        )
        time1 = time.time()
        print(time1 - time0)
    print(len(first_index))

    torch_first_index = sorted([int(ind) for ind in first_index])
    torch_shifts = sorted(
        [tuple(int(shift) for shift in tshifts) for tshifts in shift_vec]
    )

    print("timing ase implementation:")
    for n in range(1):
        time0 = time.time()
        first_index, second_index, shifts = get_neighbors_ase(
            np.array(positions), np.array(cell), cutoff
        )
        time1 = time.time()
        print(time1 - time0)

    print(len(first_index))

    ase_first_index = sorted(list(first_index))
    ase_shifts = sorted([tuple(int(shift) for shift in tshifts) for tshifts in shifts])

    # TODO: Check that second indices and shift vectors match
    # TODO: Switch torch implementations to numpy - don't need gradients
    assert torch_first_index == ase_first_index
    assert torch_shifts == ase_shifts
