"""Independent angular identities and derivative checks for Torch B2."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from flare.tensor.b2 import B2
from flare.tensor.neighbors import neighbor_list
from flare.tensor.radial import chebyshev_radial
from flare.tensor.structures import StructureBatch


def sample(dtype=torch.float64, device="cpu"):
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.2, 0.4, 0.2], [0.2, 1.3, 0.6]],
        dtype=dtype, device=device,
    )
    cell = torch.eye(3, dtype=dtype, device=device) * 6
    species = torch.tensor([1, 0, 1], device=device)
    return positions, cell, species


def test_addition_theorem_and_feature_packing():
    """Direct neighbor-pair sums via Legendre P_l avoid spherical harmonics."""
    positions, cell, species = sample()
    descriptor = B2(2, n_radial=3, lmax=3, cutoff=3.2)
    actual = descriptor(positions, cell, species, pbc=False)
    expected = torch.zeros_like(actual)
    for center in range(len(positions)):
        vectors = positions - positions[center]
        radii = vectors.norm(dim=-1)
        for left_neighbor in range(len(positions)):
            if left_neighbor == center:
                continue
            for right_neighbor in range(len(positions)):
                if right_neighbor == center:
                    continue
                r, s = radii[left_neighbor], radii[right_neighbor]
                cosine = (vectors[left_neighbor] @ vectors[right_neighbor]) / (r * s)
                legendre = [1, cosine, (3 * cosine**2 - 1) / 2,
                            (5 * cosine**3 - 3 * cosine) / 2]
                # Explicit radial polynomials independent of the implementation.
                radial_left = torch.stack([r * 0 + 1, r / 3.2, 2 * (r / 3.2)**2 - 1])
                radial_right = torch.stack([s * 0 + 1, s / 3.2, 2 * (s / 3.2)**2 - 1])
                radial_left *= (3.2 - r)**2
                radial_right *= (3.2 - s)**2
                feature = 0
                for channel_a in range(6):
                    for channel_b in range(channel_a, 6):
                        for angular in range(4):
                            if (channel_a // 3 == species[left_neighbor]
                                    and channel_b // 3 == species[right_neighbor]):
                                expected[center, feature] += (
                                    radial_left[channel_a % 3]
                                    * radial_right[channel_b % 3]
                                    * (2 * angular + 1) / (4 * math.pi)
                                    * legendre[angular]
                                )
                            feature += 1
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)


def test_spatial_symmetries_and_atom_permutation():
    positions, cell, species = sample()
    descriptor = B2(2, n_radial=3, lmax=3)
    expected = descriptor(positions, cell, species, pbc=False)
    rotation, _ = torch.linalg.qr(torch.tensor(
        [[0.2, 0.7, 0.6], [-0.4, 0.2, 0.9], [0.3, -0.2, 0.4]],
        dtype=positions.dtype,
    ))
    rotated = descriptor(positions @ rotation, cell @ rotation, species, pbc=False)
    translated = descriptor(positions + 1.7, cell, species, pbc=False)
    permutation = torch.tensor([2, 0, 1])
    permuted = descriptor(positions[permutation], cell, species[permutation], pbc=False)
    torch.testing.assert_close(rotated, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(translated, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(permuted, expected[permutation], rtol=1e-12, atol=1e-12)


def test_position_and_strain_gradients():
    positions, cell, species = sample()
    positions.requires_grad_()
    strain = torch.eye(3, dtype=positions.dtype, requires_grad=True)
    descriptor = B2(2, n_radial=2, lmax=2)
    edges = neighbor_list(positions, cell, descriptor.cutoff, False)

    def evaluate(p, deformation):
        return descriptor(p @ deformation.T, cell @ deformation.T,
                          species, pbc=False, edges=edges)

    assert torch.autograd.gradcheck(evaluate, (positions, strain), fast_mode=True)
    assert torch.autograd.gradgradcheck(evaluate, (positions, strain), fast_mode=True)
    jacobian = torch.func.jacrev(lambda p: evaluate(p, strain))(positions)
    forward_jacobian = torch.func.jacfwd(lambda p: evaluate(p, strain))(positions)
    torch.testing.assert_close(jacobian, forward_jacobian, rtol=1e-11, atol=1e-11)


def test_global_vocabulary_with_missing_species():
    positions, cell, _ = sample()
    species = torch.ones(3, dtype=torch.long)
    result = B2(3, n_radial=2, lmax=1)(positions, cell, species, pbc=False)
    assert result.shape == (3, 42)
    pairs = torch.triu_indices(6, 6)
    absent = (pairs[0] // 2 != 1) | (pairs[1] // 2 != 1)
    assert torch.count_nonzero(result.reshape(3, -1, 2)[:, absent]) == 0
    with pytest.raises(ValueError, match="vocabulary"):
        B2(1)(positions, cell, species, pbc=False)


@pytest.mark.parametrize("count", [0, 1])
def test_empty_environment_has_differentiable_zeros(count):
    positions = torch.zeros((count, 3), dtype=torch.float64, requires_grad=True)
    cell = torch.eye(3, dtype=torch.float64, requires_grad=True)
    species = torch.zeros(count, dtype=torch.long)
    result = B2(1, n_radial=2, lmax=1)(positions, cell, species, pbc=False)
    assert result.shape == (count, 6)
    assert torch.count_nonzero(result) == 0
    first = torch.autograd.grad(result.sum(), (positions, cell), create_graph=True)
    second = torch.autograd.grad(sum(g.sum() for g in first), (positions, cell))
    for gradient in first + second:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == 0


def test_batch_equals_separate_structures():
    positions, cell, species = sample()
    cells = torch.stack((cell, cell * 1.2))
    second_positions = positions[:2] + 0.3
    batch = StructureBatch(
        positions=torch.cat((positions, second_positions)),
        species=torch.cat((species, species[:2])), cells=cells,
        pbc=torch.tensor([[False] * 3, [False] * 3]),
        ptr=torch.tensor([0, 3, 5]),
    )
    descriptor = B2(2, n_radial=2, lmax=2)
    expected = torch.cat((descriptor(positions, cells[0], species, pbc=False),
                          descriptor(second_positions, cells[1], species[:2], pbc=False)))
    torch.testing.assert_close(descriptor.forward_batch(batch), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_preserves_dtype_device_and_graph(dtype, device):
    positions, cell, species = sample(dtype=dtype, device=device)
    positions.requires_grad_()
    cell.requires_grad_()
    result = B2(2, n_radial=2, lmax=2)(positions, cell, species, pbc=False)
    assert result.dtype == dtype
    assert result.device == positions.device
    gradients = torch.autograd.grad(result.sum(), (positions, cell))
    assert all(torch.isfinite(g).all() for g in gradients)


def test_chebyshev_endpoints_have_finite_derivatives():
    distances = torch.tensor([0.0, 1.5, 3.0], dtype=torch.float64, requires_grad=True)
    result = chebyshev_radial(distances, 4, 3.0)
    expected = torch.tensor([[9, 0, -9, 0], [2.25, 1.125, -1.125, -2.25],
                             [0, 0, 0, 0]], dtype=torch.float64)
    torch.testing.assert_close(result, expected)
    first, = torch.autograd.grad(result.sum(), distances, create_graph=True)
    second, = torch.autograd.grad(first.sum(), distances)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


def test_coincident_atoms_are_rejected():
    positions = torch.zeros((2, 3), dtype=torch.float64)
    with pytest.raises(ValueError, match="coincident"):
        B2(1)(positions, torch.eye(3, dtype=torch.float64),
              torch.zeros(2, dtype=torch.long), pbc=False)


def test_harmonic_setup_restores_e3nn_defaults():
    from e3nn import get_optimization_defaults

    defaults = get_optimization_defaults()
    B2(1)
    assert get_optimization_defaults() == defaults
    with pytest.raises(NotImplementedError):
        B2(1, lmax=100)
    assert get_optimization_defaults() == defaults
