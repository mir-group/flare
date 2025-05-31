import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list
from ase.geometry import complete_cell
import time
from typing import Tuple, Union
from itertools import product


def get_neighbors_ase(
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


def get_supercell_containing_rcut(
    cell: torch.Tensor,            # (..., 3, 3)
    rcut: Union[float, torch.Tensor]
) -> torch.Tensor:                 # (..., 3)  – ints
    """
    For each unit cell in `cell`, return the minimum (n_a, n_b, n_c) such that
    every point within `rcut` of the central cell is covered by the supercell
    formed by repeating the basis vectors that many times.

    Parameters
    ----------
    cell  : (..., 3, 3) torch.Tensor
        Lattice matrices with rows a, b, c.  Any number of leading batch dims.
    rcut  : float or torch.Tensor
        Cut-off radius.  If a tensor, it must broadcast with the leading
        dimensions of `cell`.

    Returns
    -------
    torch.Tensor
        Integer tensor with shape (..., 3) — repeats along (a, b, c).
    """
    # Ensure rcut is a tensor on the same device / dtype for broadcasting
    rcut = torch.as_tensor(rcut, dtype=cell.dtype, device=cell.device)

    # Split rows of the cell: a, b, c → shape (..., 3)
    a, b, c = cell.unbind(dim=-2)

    # Helper: orthogonal height of v above the plane spanned by w1 × w2
    def height(v, w1, w2):
        cross = torch.cross(w1, w2, dim=-1)                 # (..., 3)
        volume = torch.abs((v * cross).sum(dim=-1))         # scalar triple product
        base_area = torch.linalg.norm(cross, dim=-1)        # |w1 × w2|
        return volume / base_area                           # (...,)

    h_a = height(a, b, c)                                   # height along a
    h_b = height(b, a, c)                                   # height along b
    h_c = height(c, a, b)                                   # height along c

    # Minimum number of images along each direction
    n_a = torch.ceil(rcut / h_a).to(torch.int64)
    n_b = torch.ceil(rcut / h_b).to(torch.int64)
    n_c = torch.ceil(rcut / h_c).to(torch.int64)

    # (..., 3) — leading dims preserved
    return torch.stack((n_a, n_b, n_c), dim=-1)


def wrap_positions(cells: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Wrap `positions` into periodically-repeated unit cells.

    Parameters
    ----------
    cells : (..., 3, 3) torch.Tensor
        Lattice matrices (rows = a, b, c).  The leading batch dims (if any) must
        be broadcast-compatible with those of `positions`.
    positions : (..., *, 3) torch.Tensor
        Atomic Cartesian coordinates.  `*` can be empty (single vector) or one
        or more extra axes (e.g. n_atoms).

    Returns
    -------
    torch.Tensor
        Wrapped positions with the same shape as `positions`.
    """
    # (…, 3, 3)  — batched inverse handled automatically
    inv_cells = torch.linalg.inv(cells)                   

    # Cartesian → fractional  (broadcast matmul)
    frac = torch.matmul(positions, inv_cells)             # (…, *, 3)

    # Wrap into [0, 1)
    frac_wrapped = frac - torch.floor(frac)

    # Fractional → Cartesian
    wrapped = torch.matmul(frac_wrapped, cells)           # (…, *, 3)

    return wrapped


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

    shifts_all = shifts.repeat_interleave(M, dim=0)

    # Flatten to (M * R, 3)
    return supercell_positions.reshape(-1, 3), atom_indices, shifts_all


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
    shift_ijk    : (E, 3) torch.long
        Integer lattice vector **n** = (nx, ny, nz) such that
        r_j + n·cell is the neighbour chosen for the edge (i, j).
    """
    # 1) How many translations are needed so that every neighbour up to
    #    `cutoff` is present somewhere in the supercell?
    repetitions = get_supercell_containing_rcut(cell, cutoff)

    # 2) Generate all periodic images (+ map them back to their home-cell index)
    supercell_pos, atom_indices, shifts = generate_supercell_positions(
        cell, positions, repetitions
    )  # supercell_pos : (R*M, 3)

    M = positions.shape[0]

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
    shift_vec = shifts[img_index_j]

    return first_index, second_index, shift_vec


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
    shift = torch.round(d_frac).to(torch.int64)
    d_frac -= shift  # wrap into (-0.5, 0.5]

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
    shift_vec = shift[first, second]

    return first, second, shift_vec


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

    # # NaCl crystal (1000 atoms):
    # crystal = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    # supercell = crystal.repeat((5, 5, 5))
    # cell = torch.tensor(supercell.cell[:])
    # positions = torch.tensor(supercell.positions)
    # cutoff = 6.0

    # Random cell:
    n_frames = 1
    n_atoms = 100
    cell = torch.randn(3, 3)
    random_positions = torch.randn(n_atoms, 3)
    positions = wrap_positions(cell, random_positions)
    cutoff = 1.0

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
