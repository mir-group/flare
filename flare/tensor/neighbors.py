import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list
from ase.geometry import complete_cell
import time
from typing import Tuple
from itertools import product


def get_neighbor_list(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    pbc: bool = True,
):
    """
    Compute the neighbor list for a set of atoms using periodic boundary conditions.

    Parameters:
        positions (np.ndarray): Array of shape (N_atoms, 3) with Cartesian coordinates.
        cell (np.ndarray): Array of shape (3, 3) representing the unit cell vectors.
        cutoff (float): Distance cutoff for neighbor searching.
        pbc (bool or tuple): Periodic boundary conditions. Can be a bool or a tuple of 3 bools.

    Returns:
        first_index (np.ndarray): Indices of source atoms in each neighbor pair.
        second_index (np.ndarray): Indices of target atoms in each neighbor pair.
        shifts (np.ndarray): Lattice shift vectors for each neighbor interaction.
    """
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

    # remove self interactions
    bad_edge = first_index == second_index
    bad_edge &= np.all(shifts == 0, axis=1)
    keep_edge = ~bad_edge
    first_index = first_index[keep_edge]
    second_index = second_index[keep_edge]
    shifts = shifts[keep_edge]

    return first_index, second_index, shifts


def get_supercell_containing_rcut(cell: torch.Tensor, rcut: float) -> torch.Tensor:
    """
    Given a (possibly triclinic) unit cell, generate a supercell that contains
    all atoms within rcut of the central cell.

    Args:
        cell (torch.Tensor): Unit cells of shape (3, 3).
        rcut (float): Cutoff radius.

    Returns:
        torch.Tensor: Tensor of shape (N, 3), number of repeats along each cell vector.
    """
    a = cell[0]
    b = cell[1]
    c = cell[2]

    # Compute orthogonal thicknesses (height of unit cell along each direction)
    def height(v, w1, w2):
        cross = torch.cross(w1, w2, dim=0)
        det = torch.sum(v * cross)
        volume = torch.abs(det)  # scalar triple product
        base_area = torch.norm(cross)

        return volume / base_area

    h_a = height(a, b, c)
    h_b = height(b, a, c)
    h_c = height(c, a, b)

    # Compute minimum number of cells needed in each direction
    n_a = torch.ceil(rcut / h_a).int()
    n_b = torch.ceil(rcut / h_b).int()
    n_c = torch.ceil(rcut / h_c).int()
    repetitions = torch.stack([n_a, n_b, n_c])

    return repetitions


def wrap_positions(cells: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Wrap positions into the unit cell using periodic boundary conditions.

    Args:
        cells (torch.Tensor): Unit cells of shape (N, 3, 3).
        positions (torch.Tensor): Atomic positions of shape (N, M, 3).

    Returns:
        torch.Tensor: Wrapped positions of shape (N, M, 3).
    """

    inv_cells = torch.inverse(cells)
    frac = torch.einsum("nij,njk->nik", positions, inv_cells)
    frac_wrapped = frac % 1.0
    wrapped_positions = torch.einsum("nij,njk->nik", frac_wrapped, cells)

    return wrapped_positions


def generate_supercell_positions(
    cell: torch.Tensor, positions: torch.Tensor, repetitions: torch.Tensor
) -> torch.Tensor:
    """
    Generate supercell atomic positions by repeating a single unit cell in both
    directions.

    Args:
        positions (torch.Tensor): Atomic positions in Cartesian coordinates,
        shape (M, 3).
        cell (torch.Tensor): Unit cell matrix, shape (3, 3).
        repetitions (torch.Tensor): Number of repetitions in each direction
        (±n), shape (3,).

    Returns:
        torch.Tensor: Supercell positions, shape (M * R, 3),
                      where R = ∏(2 * repetitions + 1).
    """
    reps = repetitions.int()

    # Generate shift indices: [-n, ..., 0, ..., n]
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
    )  # shape: (R, 3)

    # Convert to Cartesian displacements
    displacements = shifts @ cell

    # Repeat positions for each shift
    base = positions.unsqueeze(0).expand(len(displacements), -1, -1)  # (R, M, 3)
    supercell_positions = base + displacements.unsqueeze(1)  # (R, M, 3)

    # Generate corresponding atom indices.
    M = positions.shape[0]
    atom_indices = (
        torch.arange(M, device=positions.device, dtype=torch.long)
        .unsqueeze(0)  # (1, M)
        .expand(len(displacements), -1)  # (R, M)
        .reshape(-1)  # (M × R,)
    )

    # Flatten to (M * R, 3)
    return supercell_positions.reshape(-1, 3), atom_indices


def get_neighbors_brute_force(
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a neighbor list under periodic boundary conditions.

    Parameters
    ----------
    cell : (3, 3) torch.Tensor
        Lattice vectors as rows (Cartesian Å or Bohr are fine as long as
        `positions` use the same units).
    positions : (M, 3) torch.Tensor
        Atomic positions of the *reference* unit cell.
    cutoff : float
        Pairwise cutoff distance.

    Returns
    -------
    first_index  : (E,) torch.LongTensor
        Index (0 … M-1) of the central atom of each edge.
    second_index : (E,) torch.LongTensor
        Index (0 … M-1) of the neighbour atom (in any periodic image).
    rel_vec      : (E, 3) torch.Tensor
        Displacement vector **r_j – r_i** in Cartesian coordinates that
        satisfies ‖rel_vec‖ ≤ cutoff.
    """
    # 1) How many translations are needed so that every neighbour up to
    #    `cutoff` is present somewhere in the supercell?
    repetitions = get_supercell_containing_rcut(cell, cutoff)

    # 2) Generate all periodic images (+ map them back to their home-cell index)
    supercell_pos, atom_indices = generate_supercell_positions(
        cell, positions, repetitions
    )  # supercell_pos : (R*M, 3)

    M = positions.shape[0]
    N = supercell_pos.shape[0]

    # 3) Pairwise displacement: r_j – r_i  (broadcasting)
    disp = supercell_pos.unsqueeze(0) - positions.unsqueeze(1)  # (M, N, 3)
    dist2 = (disp**2).sum(dim=-1)  # (M, N)

    # 4) Within-cutoff mask
    within_cutoff = dist2 <= cutoff * cutoff  # bool (M, N)

    # 5) Eliminate self-interaction at zero shift
    same_atom = atom_indices.unsqueeze(0) == torch.arange(
        M, device=positions.device
    ).unsqueeze(1)
    zero_distance = dist2 == 0.0
    within_cutoff &= ~(same_atom & zero_distance)

    # 6) Extract edges
    pairs = torch.nonzero(within_cutoff, as_tuple=False)  # (E, 2)
    first_index = pairs[:, 0]  # i
    img_index_j = pairs[:, 1]  # j in the big list
    second_index = atom_indices[img_index_j]  # map to 0 … M-1
    rel_vec = disp[first_index, img_index_j, :]  # (E, 3)

    return first_index, second_index, rel_vec


def get_neighbors_minimum_image(
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
    directed: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Neighbour list via minimum-image convention (triclinic OK).

    Returns the same (first_index, second_index, rel_vec) triplet but
    *without* generating any explicit periodic images.

    Parameters
    ----------
    cell      : (3, 3) lattice matrix  (rows = a, b, c)
    positions : (M, 3) Cartesian coordinates
    cutoff    : scalar cutoff distance
    directed  : keep both i→j and j→i edges (default True)

    Notes
    -----
    * Works for any full-rank cell.
    * Uses only 𝑂(M²) memory.
    """
    device, dtype = positions.device, positions.dtype
    M = positions.shape[0]
    rcut2 = cutoff * cutoff

    # -------- 1) Cartesian → fractional coordinates -----------------------
    # Fast path for orthorhombic cells: just divide
    if torch.allclose(cell, torch.diag(torch.diagonal(cell))):
        frac = positions / torch.diagonal(cell)
    else:
        # (cell.T)⁻¹ is faster than full inverse because positions are row-vecs
        cell_inv_T = torch.linalg.inv(cell).T  # (3, 3)
        frac = positions @ cell_inv_T  # (M, 3)

    # -------- 2) Pairwise fractional displacements, minimum-image ----------
    d_frac = frac.unsqueeze(1) - frac.unsqueeze(0)  # (M, M, 3)
    d_frac -= torch.round(d_frac)  # wrap into (-0.5, 0.5]

    # -------- 3) Convert back to Cartesian and compute |r|² ----------------
    #   d_cart = d_frac @ cell   (broadcast matmul)
    d_cart = torch.matmul(d_frac, cell)  # (M, M, 3)
    dist2 = (d_cart**2).sum(-1)  # (M, M)

    # -------- 4) Mask: within cutoff & not the same atom -------------------
    mask = dist2 <= rcut2  # bool (M, M)
    mask.fill_diagonal_(False)  # drop i == j

    if not directed:
        mask = torch.triu(mask, diagonal=1)  # undirected list

    # -------- 5) Gather indices & displacement vectors ---------------------
    first, second = torch.nonzero(mask, as_tuple=True)
    rel_vec = d_cart[first, second]

    return first, second, rel_vec


def mic_safe(cell: torch.Tensor, r_cut: float, margin: float = 1e-8) -> bool:
    """
    True if the minimum-image convention is guaranteed to be exact for
    the given cutoff.

    Parameters
    ----------
    cell   : (3,3) tensor, lattice matrix (rows = a, b, c)
    r_cut  : cutoff radius
    margin : small buffer to stay away from the strict ½|h_min| limit
    """
    device, dtype = cell.device, cell.dtype

    # Generate the 26 neighbour lattice vectors
    neighbours = torch.tensor(
        [v for v in product((-1, 0, 1), repeat=3) if v != (0, 0, 0)],
        device=device,
        dtype=dtype,
    )  # (26, 3)

    h_vecs = neighbours @ cell  # (26, 3)
    h_norm = torch.linalg.norm(h_vecs, dim=1).min().item()

    return r_cut < 0.5 * h_norm - margin


def get_neighbors_auto(
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
    *,
    directed: bool = True,
):
    """
    Uses the fast minimum-image algorithm whenever it is provably exact;
    otherwise falls back to the brute-force super-cell generator.
    """
    if mic_safe(cell, cutoff):
        return get_neighbors_minimum_image(cell, positions, cutoff, directed=directed)
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
    # cells = torch.randn(n_frames, 3, 3)
    # random_positions = torch.randn(n_frames, n_atoms, 3)
    # all_positions = wrap_positions(cells.expand(n_frames, -1, -1), random_positions)
    # cell = cells[0]
    # positions = all_positions[0]
    # cutoff = 1.0

    print("timing torch brute force implementation:")
    for n in range(10):
        time0 = time.time()
        first_index, second_index, rel_vec = get_neighbors_brute_force(
            cell, positions, cutoff
        )
        time1 = time.time()
        print(time1 - time0)

    print(len(first_index))

    print("timing torch minimum image implementation:")
    for n in range(10):
        time0 = time.time()
        first_index, second_index, rel_vec = get_neighbors_minimum_image(
            cell, positions, cutoff
        )
        time1 = time.time()
        print(time1 - time0)
    print(len(first_index))

    print("timing torch auto implementation:")
    for n in range(10):
        time0 = time.time()
        first_index, second_index, rel_vec = get_neighbors_auto(cell, positions, cutoff)
        time1 = time.time()
        print(time1 - time0)
    print(len(first_index))

    print("timing ase implementation:")
    for n in range(1):
        time0 = time.time()
        first_index, second_index, shifts = get_neighbor_list(
            np.array(positions), np.array(cell), cutoff
        )
        time1 = time.time()
        print(time1 - time0)

    print(len(first_index))
