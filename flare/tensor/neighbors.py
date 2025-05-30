import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list
from ase.geometry import complete_cell
import time


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


def get_supercell_containing_rcut(cells: torch.Tensor, rcut: float) -> torch.Tensor:
    """
    Given a (possibly triclinic) unit cell, generate a supercell that contains
    all atoms within rcut of the central cell.

    Args:
        cells (torch.Tensor): Unit cells of shape (N, 3, 3).
        rcut (float): Cutoff radius.

    Returns:
        torch.Tensor: Tensor of shape (N, 3), number of repeats along each cell vector.
    """
    a = cells[:, 0, :]  # (N, 3)
    b = cells[:, 1, :]
    c = cells[:, 2, :]

    # Compute orthogonal thicknesses (height of unit cell along each direction)
    def height(v, w1, w2):
        cross = torch.cross(w1, w2, dim=1)  # (N, 3)
        det = torch.sum(v * cross, dim=1)
        volume = torch.abs(det)  # scalar triple product
        base_area = torch.norm(cross, dim=1)

        return volume / base_area

    h_a = height(a, b, c)
    h_b = height(b, a, c)
    h_c = height(c, a, b)

    # Compute minimum number of cells needed in each direction
    n_a = torch.ceil(rcut / h_a).int()
    n_b = torch.ceil(rcut / h_b).int()
    n_c = torch.ceil(rcut / h_c).int()
    n = torch.stack([n_a, n_b, n_c], dim=1)

    return n


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

    # Flatten to (M * R, 3)
    return supercell_positions.reshape(-1, 3)

if __name__ == "__main__":
    from ase.build import bulk

    np.random.seed(1)

    crystal = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    supercell = crystal.repeat((5, 5, 5))
    cell = supercell.cell[:]
    positions = supercell.positions
    cutoff = 6.0

    ns = get_supercell_containing_rcut(torch.tensor(cell).unsqueeze(0), cutoff)
    print(ns)
