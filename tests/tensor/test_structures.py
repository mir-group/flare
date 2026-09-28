"""Batch contract, global species vocabulary, and graph preservation."""

from dataclasses import FrozenInstanceError

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")
from flare.tensor.neighbors import batched_neighbor_list, get_batched_edge_vectors
from flare.tensor.structures import StructureBatch


def test_single_structure_preserves_global_species_and_graph():
    positions = torch.tensor(
        [[0., 0., 0.], [0.5, 0.2, 0.1]], dtype=torch.float64, requires_grad=True
    )
    cell = torch.eye(3, dtype=positions.dtype, requires_grad=True)
    species = torch.tensor([3, 3])
    batch = StructureBatch.from_single(positions, species, cell, (True, True, False))
    assert batch.positions is positions and batch.species is species
    assert batch.species.tolist() == [3, 3]
    assert batch.ptr.tolist() == [0, 2]
    assert batch.structure_indices.tolist() == [0, 0]
    assert batch.pbc.tolist() == [[True, True, False]]
    position_gradient, cell_gradient = torch.autograd.grad(
        batch.positions.sum() + batch.cells.sum(), (positions, cell)
    )
    torch.testing.assert_close(position_gradient, torch.ones_like(positions))
    torch.testing.assert_close(cell_gradient, torch.ones_like(cell))
    with pytest.raises(FrozenInstanceError):
        batch.positions = positions.detach()


def test_molecule_defaults_to_nonperiodic_zero_cell():
    batch = StructureBatch.from_single(torch.zeros((1, 3)), torch.tensor([5]))
    assert not bool(batch.pbc.any())
    assert not bool(batch.cells.any())


@pytest.mark.parametrize("ptr", [[0, 0, 1, 1], [0, 1, 1, 1]])
def test_empty_structures_in_ragged_batch(ptr):
    batch = StructureBatch(
        torch.zeros((1, 3)), torch.tensor([4]), torch.zeros((3, 3, 3)),
        torch.zeros((3, 3), dtype=torch.bool), torch.tensor(ptr),
    )
    expected = 1 if ptr == [0, 0, 1, 1] else 0
    assert batch.structure_indices.tolist() == [expected]
    edges = batched_neighbor_list(batch, 1.)
    assert get_batched_edge_vectors(batch, *edges).shape == (0, 3)


def test_empty_batch():
    batch = StructureBatch(
        torch.empty((0, 3)), torch.empty(0, dtype=torch.long),
        torch.empty((0, 3, 3)), torch.empty((0, 3), dtype=torch.bool), torch.tensor([0]),
    )
    assert batch.num_structures == 0 and batch.structure_indices.shape == (0,)
    edges = batched_neighbor_list(batch, 1.)
    assert get_batched_edge_vectors(batch, *edges).shape == (0, 3)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("positions", torch.tensor([[float("nan"), 0., 0.]]), "finite"),
        ("cells", torch.full((1, 3, 3), float("inf")), "finite"),
        ("species", torch.tensor([-1]), "nonnegative"),
        ("species", torch.tensor([0.]), "int64"),
        ("ptr", torch.tensor([1, 1]), "start at zero"),
        ("ptr", torch.tensor([0, 2]), "number of atoms"),
        ("pbc", torch.zeros((1, 3)), "bool"),
        ("cells", torch.zeros((1, 3, 3), dtype=torch.float64), "same dtype"),
    ],
)
def test_invalid_batch_fields(field, value, error):
    fields = dict(
        positions=torch.zeros((1, 3)), species=torch.tensor([0]),
        cells=torch.zeros((1, 3, 3)), pbc=torch.zeros((1, 3), dtype=torch.bool),
        ptr=torch.tensor([0, 1]),
    )
    fields[field] = value
    with pytest.raises((ValueError, TypeError), match=error):
        StructureBatch(**fields)


@pytest.mark.parametrize("pbc", [[True, False], [1, 0, 0], torch.tensor([1, 0, 0]), "xyz"])
def test_invalid_pbc_is_rejected(pbc):
    with pytest.raises((ValueError, TypeError), match="pbc"):
        StructureBatch.from_single(torch.zeros((1, 3)), torch.tensor([0]), torch.eye(3), pbc)
