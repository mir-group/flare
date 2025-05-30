import numpy as np
import torch
import time
from typing import Tuple
from itertools import product
from neighbors import get_neighbors_ase


def get_supercell_containing_rcut(cell: np.ndarray, rcut: float) -> np.ndarray:
    """
    Given a (possibly triclinic) unit cell, generate a supercell that contains
    all atoms within rcut of the central cell.

    Parameters
    ----------
    cell : (3, 3) np.ndarray
        Lattice vectors as rows.  Units must match `rcut`.
    rcut : float
        Cutoff radius.

    Returns
    -------
    np.ndarray
        Array of shape (3,) with the minimum number of repeats
        along the a-, b-, and c-vectors.
    """
    a, b, c = cell  # rows of the matrix

    # --- helper: height of the parallelepiped along vector v ---
    def height(v, w1, w2):
        cross = np.cross(w1, w2)
        volume = abs(np.dot(v, cross))  # scalar-triple product
        base_area = np.linalg.norm(cross)  # area of parallelogram spanned by w1, w2
        return volume / base_area  # height = V / A

    h_a = height(a, b, c)
    h_b = height(b, a, c)
    h_c = height(c, a, b)

    # minimum number of whole cells needed in each direction
    n_a = int(np.ceil(rcut / h_a))
    n_b = int(np.ceil(rcut / h_b))
    n_c = int(np.ceil(rcut / h_c))

    return np.array([n_a, n_b, n_c], dtype=int)


def wrap_positions(cell: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Periodically wrap Cartesian coordinates into a single unit cell.

    Parameters
    ----------
    cell : (3, 3) np.ndarray
        Lattice vectors as rows of the cell matrix.
    positions : (..., 3) np.ndarray
        Cartesian coordinates to wrap.  The leading dimensions (if any) are
        preserved; only the last dimension must be length 3.

    Returns
    -------
    np.ndarray
        Wrapped positions with the same shape and dtype as `positions`.
    """
    # Inverse cell once (3×3)
    inv_cell = np.linalg.inv(cell)

    # Cartesian → fractional
    frac = np.einsum("...j,jk->...k", positions, inv_cell)

    # Wrap to [0, 1)
    frac_wrapped = frac % 1.0

    # Fractional → Cartesian
    wrapped = np.einsum("...j,jk->...k", frac_wrapped, cell)

    return wrapped.astype(positions.dtype, copy=False)


def generate_supercell_positions(
    cell: np.ndarray,
    positions: np.ndarray,
    repetitions: np.ndarray,
):
    """
    Build a super-cell by repeating a reference unit cell in ±n a/±n b/±n c
    directions and return all atomic coordinates.

    Parameters
    ----------
    cell : (3, 3) np.ndarray
        Lattice vectors as rows (Cartesian units).
    positions : (M, 3) np.ndarray
        Cartesian coordinates inside the reference cell.
    repetitions : (3,) array-like
        Number of repeats in the +a, +b, +c directions.
        The same number is used in the − direction, so e.g. [1,1,0] produces
        three copies along a, three along b, and one along c.

    Returns
    -------
    supercell_pos : (M * R, 3) np.ndarray
        Coordinates of all atoms in the super-cell, where
        R = ∏(2*repetitions + 1).
    atom_indices : (M * R,) np.ndarray
        Index (0 … M-1) of the parent atom for each row in `supercell_pos`.
    """
    # Ensure integer repetition counts
    reps = np.asarray(repetitions, dtype=int)

    # -- 1. Build all integer shift triples -------------------------------
    ranges = [np.arange(-n, n + 1, dtype=int) for n in reps]  # three 1-D arrays
    shift_grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1)
    shifts = shift_grid.reshape(-1, 3).astype(cell.dtype, copy=False)  # (R, 3)

    # -- 2. Convert those fractional shifts to Cartesian displacements ----
    displacements = shifts @ cell  # (R, 3)

    # -- 3. Translate every reference-cell atom by every displacement ----
    base = positions[np.newaxis, :, :]  # (1, M, 3)
    supercell_positions = base + displacements[:, np.newaxis, :]  # (R, M, 3)

    # -- 4. Flatten to (M*R, 3) and build parent-atom index array -------
    supercell_pos_flat = supercell_positions.reshape(-1, 3)
    M = positions.shape[0]
    atom_indices = np.tile(np.arange(M, dtype=int), len(displacements))

    return supercell_pos_flat.astype(positions.dtype, copy=False), atom_indices


def get_neighbors_brute_force(
    cell: np.ndarray,
    positions: np.ndarray,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a neighbour list under periodic boundary conditions via a brute-force
    super-cell search.

    Parameters
    ----------
    cell : (3, 3) np.ndarray
        Lattice vectors as rows (Cartesian units must match `positions`).
    positions : (M, 3) np.ndarray
        Atomic positions inside one reference cell.
    cutoff : float
        Pairwise cutoff distance.

    Returns
    -------
    first_index  : (E,) np.ndarray[int]
        Central atom index i (0 … M-1) for each edge.
    second_index : (E,) np.ndarray[int]
        Neighbour atom index j (0 … M-1) in some periodic image.
    shift_ijk    : (E, 3) np.ndarray[int]
        Integer lattice vector n = (nx, ny, nz) such that
        r_j + n·cell is the chosen neighbour of atom i.
    """
    # 1) How many translations are needed to reach `cutoff`?
    repetitions = get_supercell_containing_rcut(cell, cutoff)  # (3,)

    # 2) Generate every periodic image + remember parent indices
    supercell_pos, atom_indices = generate_supercell_positions(
        cell, positions, repetitions
    )  # (R*M, 3), (R*M,)

    M = positions.shape[0]
    N = supercell_pos.shape[0]
    cell_inv_T = np.linalg.inv(cell).T  # (3,3)

    # 3) Pairwise displacements r_j – r_i  (broadcasting)
    disp = supercell_pos[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (M,N,3)
    dist2 = np.sum(disp**2, axis=-1)  # (M,N)

    # 4) Mask pairs within cutoff
    within_cutoff = dist2 <= cutoff**2  # (M,N) bool

    # 5) Remove self-interaction at zero shift
    same_atom = atom_indices[np.newaxis, :] == np.arange(M)[:, np.newaxis]
    zero_dist = dist2 == 0.0
    within_cutoff &= ~(same_atom & zero_dist)

    # 6) Extract edge list
    pairs = np.argwhere(within_cutoff)  # (E,2)  columns: i, j_big
    first_index = pairs[:, 0].astype(int)
    img_index_j = pairs[:, 1].astype(int)
    second_index = atom_indices[img_index_j]  # map to 0…M-1

    rel_vec = disp[first_index, img_index_j, :]  # (E,3)
    frac_vec = rel_vec @ cell_inv_T
    shift_vec = np.round(frac_vec).astype(int)  # (E,3)

    return first_index, second_index, shift_vec


def get_neighbors_minimum_image(
    cell: np.ndarray,
    positions: np.ndarray,
    cutoff: float,
    directed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Neighbour list via the minimum-image convention (works for triclinic cells).

    Parameters
    ----------
    cell      : (3, 3) np.ndarray  – lattice matrix (rows = a, b, c)
    positions : (M, 3) np.ndarray  – Cartesian coordinates
    cutoff    : float              – pairwise cutoff distance
    directed  : bool               – if False, keep only i<j edges

    Returns
    -------
    first_index  : (E,) np.ndarray[int]
    second_index : (E,) np.ndarray[int]
    shift_ijk    : (E, 3) np.ndarray[int]   (# lattice translations)
    """
    M = positions.shape[0]
    rcut2 = cutoff * cutoff

    # --- 1) Cartesian → fractional coordinates ---------------------------
    # Fast orthorhombic path
    if np.allclose(cell, np.diag(np.diagonal(cell))):
        frac = positions / np.diagonal(cell)  # (M,3)
    else:
        cell_inv_T = np.linalg.inv(cell).T  # (3,3)
        frac = positions @ cell_inv_T  # (M,3)

    # --- 2) Pairwise fractional displacement, minimum-image --------------
    d_frac = frac[:, np.newaxis, :] - frac[np.newaxis, :, :]  # (M,M,3)
    shift = np.round(d_frac).astype(np.int64)  # (M,M,3)
    d_frac -= shift  # wrap into (−0.5,0.5]

    # --- 3) Back to Cartesian & distance² -------------------------------
    d_cart = d_frac @ cell  # (M,M,3)
    dist2 = np.sum(d_cart**2, axis=-1)  # (M,M)

    # --- 4) Mask: within cutoff & not self ------------------------------
    mask = dist2 <= rcut2  # (M,M) bool
    np.fill_diagonal(mask, False)

    if not directed:  # undirected list
        mask = np.triu(mask, k=1)

    # --- 5) Gather edges -----------------------------------------------
    first_idx, second_idx = np.nonzero(mask)  # (E,), (E,)
    shift_vec = shift[first_idx, second_idx]  # (E,3)

    return first_idx.astype(int), second_idx.astype(int), shift_vec


def mic_safe(cell: np.ndarray, r_cut: float, margin: float = 1e-8) -> bool:
    """
    Return True iff the minimum-image convention is guaranteed to be exact
    for a given cutoff in this lattice.

    Parameters
    ----------
    cell   : (3, 3) np.ndarray       lattice vectors as rows
    r_cut  : float                   cutoff radius
    margin : float, optional         small safety buffer (<½ |h_min|)
    """
    # 26 neighbour lattice vectors (all non-zero triples of −1/0/1)
    neighbours = np.array(
        [v for v in product((-1, 0, 1), repeat=3) if v != (0, 0, 0)],
        dtype=cell.dtype,
    )  # (26, 3)

    h_vecs = neighbours @ cell  # Cartesian height vectors
    h_norm = np.linalg.norm(h_vecs, axis=1).min()  # shortest height

    return r_cut < 0.5 * h_norm - margin


def get_neighbors_auto(
    cell: np.ndarray,
    positions: np.ndarray,
    cutoff: float,
    *,
    directed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick the fastest neighbour-list algorithm automatically:

    * If `mic_safe` proves the minimum-image convention is exact at this
      cutoff, use the `get_neighbors_minimum_image` routine (𝒪(M²), no
      explicit super-cell).
    * Otherwise fall back to the brute-force super-cell generator.

    Returns the usual (first_index, second_index, shift_vec) triplet.
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
    cell = supercell.cell[:]
    positions = supercell.positions
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
    for n in range(10):
        time0 = time.time()
        first_index, second_index, shifts = get_neighbors_ase(positions, cell, cutoff)
        time1 = time.time()
        print(time1 - time0)

    print(len(first_index))

    ase_first_index = sorted(list(first_index))
    ase_shifts = sorted([tuple(int(shift) for shift in tshifts) for tshifts in shifts])

    # TODO: Check that second indices and shift vectors match
    # TODO: Switch torch implementations to numpy - don't need gradients
    assert torch_first_index == ase_first_index
    assert torch_shifts == ase_shifts
