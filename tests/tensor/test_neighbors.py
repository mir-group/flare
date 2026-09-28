"""Neighbor topology parity and differentiable reconstruction checks."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")
from ase.neighborlist import primitive_neighbor_list

from flare.tensor.neighbors import (
    batched_neighbor_list,
    get_batched_edge_vectors,
    get_edge_vectors,
    neighbor_list,
)
from flare.tensor.structures import StructureBatch


def _edge_set(first, second, shifts):
    return {
        (i, j, *shift)
        for i, j, shift in zip(first.tolist(), second.tolist(), shifts.tolist())
    }


def _assert_ase_parity(positions, cell, cutoff, pbc):
    edges = neighbor_list(positions, cell, cutoff, pbc)
    expected = primitive_neighbor_list(
        "ijS", pbc, cell.cpu().numpy(), positions.cpu().numpy(), cutoff,
        self_interaction=True,
    )
    expected_set = _edge_set(*expected)
    expected_set -= {(i, i, 0, 0, 0) for i in range(len(positions))}
    assert _edge_set(*edges) == expected_set
    assert all(value.dtype == torch.long and value.device == positions.device for value in edges)
    return edges


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_skew_unwrapped_cells_match_ase(dtype):
    generator = torch.Generator().manual_seed(147)
    for _ in range(12):
        # Strongly skew, rather than reduced/orthogonal, lattice vectors.
        cell = torch.tensor([[2.6, 0, 0], [2.1, 1.5, 0], [1.6, 0.9, 1.9]], dtype=dtype)
        cell += 0.08 * torch.randn((3, 3), generator=generator, dtype=dtype)
        fractional = torch.rand((5, 3), generator=generator, dtype=dtype)
        fractional += torch.randint(-4, 5, (5, 3), generator=generator).to(dtype)
        positions = fractional @ cell
        _assert_ase_parity(positions, cell, 2.3, [True, True, True])


@pytest.mark.parametrize(
    "cell,pbc",
    [
        ([[2, 0, 0], [1.8, 1.3, 0], [0, 0, 0]], [True, True, False]),
        ([[0, 0, 0], [0.2, 2, 0.3], [0, 0, 0]], [False, True, False]),
        ([[2, 0, 0], [1.8, 1.3, 0], [0.1, 0.7, 2]], [True, False, True]),
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [False, False, False]),
    ],
)
def test_partial_pbc_and_incomplete_cells(cell, pbc):
    cell = torch.tensor(cell, dtype=torch.float64)
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [2.5, 0.7, 0.1], [-1.3, 1.1, 0.7]], dtype=torch.float64
    )
    positions = positions + torch.tensor(
        [[3., -2., 1.], [-4., 5., -1.], [2., 3., -3.]], dtype=cell.dtype
    ) @ cell
    first, second, shifts = _assert_ase_parity(positions, cell, 1.8, pbc)
    assert bool((shifts[:, ~torch.tensor(pbc)] == 0).all())
    vectors = get_edge_vectors(positions, cell, first, second, shifts)
    assert bool((torch.linalg.vector_norm(vectors, dim=-1) < 1.8).all())


@pytest.mark.parametrize(
    "pbc",
    [True, (True, True, True), torch.tensor(True), torch.tensor([True, True, True])],
)
def test_multiple_images_include_periodic_self_edges(pbc):
    positions = torch.zeros((1, 3), dtype=torch.float64)
    cell = torch.eye(3, dtype=positions.dtype)
    first, second, shifts = neighbor_list(positions, cell, 1.01, pbc)
    assert len(first) == 6
    assert bool((first == second).all())
    assert _edge_set(first, second, shifts) == {
        (0, 0, 1, 0, 0), (0, 0, -1, 0, 0),
        (0, 0, 0, 1, 0), (0, 0, 0, -1, 0),
        (0, 0, 0, 0, 1), (0, 0, 0, 0, -1),
    }


def test_strict_cutoff_and_distinct_coincident_atoms():
    positions = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]], dtype=torch.float64)
    edges = neighbor_list(positions, torch.zeros((3, 3), dtype=positions.dtype), 1., False)
    assert _edge_set(*edges) == {(0, 2, 0, 0, 0), (2, 0, 0, 0, 0)}


@pytest.mark.parametrize("count", [0, 1])
def test_empty_neighbor_outputs(count):
    positions = torch.zeros((count, 3), dtype=torch.float64)
    edges = neighbor_list(positions, torch.zeros((3, 3), dtype=positions.dtype), 1., False)
    assert edges[0].shape == edges[1].shape == (0,)
    assert edges[2].shape == (0, 3)
    vectors = get_edge_vectors(
        positions, torch.zeros((3, 3), dtype=positions.dtype), *edges
    )
    assert vectors.shape == (0, 3)


def test_edge_reconstruction_coordinate_and_cell_gradients():
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [2.2, 0.8, 0.4]], dtype=torch.float64, requires_grad=True
    )
    cell = torch.tensor(
        [[2.5, 0.1, 0], [0.8, 2.8, 0.2], [0.1, 0.3, 3.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edges = neighbor_list(positions, cell, 1.7)
    assert bool((edges[2] != 0).any())
    assert all(not edge.requires_grad for edge in edges)

    def pair_energy(coords, lattice):
        vectors = get_edge_vectors(coords, lattice, *edges)
        return (vectors.square().sum(dim=1).exp()).sum()

    assert torch.autograd.gradcheck(pair_energy, (positions, cell))
    assert torch.autograd.gradgradcheck(pair_energy, (positions, cell))


def test_batches_do_not_mix_structures_and_preserve_gradients():
    positions = torch.tensor(
        [[0., 0., 0.], [0.6, 0.1, 0.2], [0., 0., 0.], [1.5, 0.2, 0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    cells = torch.stack([
        torch.zeros((3, 3), dtype=positions.dtype),
        2 * torch.eye(3, dtype=positions.dtype),
    ]).requires_grad_()
    batch = StructureBatch(
        positions, torch.tensor([0, 2, 2, 2]), cells,
        torch.tensor([[False, False, False], [True, True, True]]),
        torch.tensor([0, 2, 4]),
    )
    edges = batched_neighbor_list(batch, 1.)
    assert bool((batch.structure_indices[edges[0]] == batch.structure_indices[edges[1]]).all())
    vectors = get_batched_edge_vectors(batch, *edges)
    for b in range(2):
        start, end = int(batch.ptr[b]), int(batch.ptr[b + 1])
        local_edges = neighbor_list(positions[start:end], cells[b], 1., batch.pbc[b])
        mask = batch.structure_indices[edges[0]] == b
        torch.testing.assert_close(
            vectors[mask], get_edge_vectors(positions[start:end], cells[b], *local_edges)
        )
    coordinate_gradient, cell_gradient = torch.autograd.grad(
        vectors.square().sum(), (positions, cells)
    )
    assert bool(torch.isfinite(coordinate_gradient).all())
    assert bool(torch.isfinite(cell_gradient).all())
    assert bool((cell_gradient[0] == 0).all())
    assert bool((cell_gradient[1] != 0).any())


@pytest.mark.parametrize("cutoff", [0., -1., float("nan"), float("inf")])
def test_invalid_cutoffs_are_rejected(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        neighbor_list(torch.zeros((1, 3)), torch.eye(3), cutoff)


def test_dependent_periodic_vectors_are_rejected():
    with pytest.raises(ValueError, match="linearly independent"):
        neighbor_list(torch.zeros((1, 3)), torch.zeros((3, 3)), 1., True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_device_and_dtype_are_preserved():
    positions = torch.tensor(
        [[0., 0., 0.], [0.5, 0., 0.]],
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    cell = torch.eye(3, dtype=positions.dtype, device=positions.device)
    edges = neighbor_list(positions, cell, 0.75)
    vectors = get_edge_vectors(positions, cell, *edges)
    assert all(value.device == positions.device for value in edges)
    assert vectors.dtype == positions.dtype and vectors.device == positions.device
    assert torch.autograd.grad(vectors.square().sum(), positions)[0].device == positions.device
