"""Covariance, mixed derivative, and cutoff-underflow regression checks."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from flare.tensor.b2 import B2
from flare.tensor.kernels import normalized_dot_product
from flare.tensor.neighbors import neighbor_list


def test_kernel_formula_species_gate_and_positive_semidefinite():
    values = torch.tensor([[1., 2., 3.], [2., -1., 1.], [0.5, 0.2, -0.1]],
                          dtype=torch.float64)
    species = torch.tensor([0, 1, 0])
    unit = values / values.norm(dim=-1, keepdim=True)
    expected = 1.7**2 * (unit @ unit.T)**2 * (species[:, None] == species[None, :])
    actual = normalized_dot_product(values, values, species, species, amplitude=1.7)
    torch.testing.assert_close(actual, expected)
    assert torch.linalg.eigvalsh(actual).min() >= -1e-13
    torch.testing.assert_close(actual.diag(), torch.full((3,), 1.7**2, dtype=values.dtype))


def test_kernel_first_and_second_derivatives():
    left = torch.tensor([[1., -0.2, 0.4], [0.1, 0.6, 0.2]],
                        dtype=torch.float64, requires_grad=True)
    right = torch.tensor([[0.7, 0.8, -0.2]], dtype=torch.float64, requires_grad=True)
    amplitude = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    left_species, right_species = torch.tensor([0, 1]), torch.tensor([0])

    def evaluate(a, b, sigma):
        return normalized_dot_product(a, b, left_species, right_species, sigma)

    assert torch.autograd.gradcheck(evaluate, (left, right, amplitude))
    assert torch.autograd.gradgradcheck(evaluate, (left, right, amplitude))


@pytest.mark.parametrize("distance", [2.9999, 3.0, 3.0001])
def test_float32_near_cutoff_has_finite_position_and_strain_derivatives(distance):
    positions = torch.tensor([[0., 0., 0.], [distance, 0., 0.]], requires_grad=True)
    cell = torch.eye(3) * 20
    strain = torch.eye(3, requires_grad=True)
    species = torch.zeros(2, dtype=torch.long)
    descriptor = B2(1, n_radial=3, lmax=2, cutoff=3)
    edges = neighbor_list(positions, cell, 3, False)
    values = descriptor(positions @ strain.T, cell @ strain.T, species,
                        pbc=False, edges=edges)
    reference = torch.ones((1, descriptor.n_features))
    result = normalized_dot_product(values, reference, species, species[:1])
    first = torch.autograd.grad(result.sum(), (positions, strain), create_graph=True)
    second = torch.autograd.grad(sum(g.sum() for g in first), (positions, strain))
    for value in (values, result) + first + second:
        assert torch.isfinite(value).all()
    assert torch.count_nonzero(result) == 0  # raw norms lie below the legacy threshold


def test_normalization_handles_tiny_and_large_values_without_squaring_raw_values():
    values = torch.tensor([[0., 0.], [1e-40, -1e-40], [1e30, -1e30]],
                          dtype=torch.float32, requires_grad=True)
    species = torch.zeros(3, dtype=torch.long)
    actual = normalized_dot_product(values, values, species, species)
    expected = torch.zeros((3, 3))
    expected[-1, -1] = 1
    torch.testing.assert_close(actual, expected)
    gradient, = torch.autograd.grad(actual.sum(), values)
    assert torch.isfinite(gradient).all()


def test_empty_threshold_uses_raw_norm_and_includes_equality():
    values = torch.tensor([[0., 0.], [0.6e-8, 0.6e-8], [1e-8, 0.], [1e-7, 0.]],
                          dtype=torch.float64)
    species = torch.zeros(4, dtype=torch.long)
    actual = normalized_dot_product(values, values, species, species)
    torch.testing.assert_close(actual.diag(), torch.tensor([0., 0., 1., 1.], dtype=values.dtype))


def test_empty_reference_matrix():
    left = torch.ones((2, 3), dtype=torch.float64, requires_grad=True)
    right = torch.empty((0, 3), dtype=torch.float64, requires_grad=True)
    result = normalized_dot_product(left, right, torch.zeros(2, dtype=torch.long),
                                    torch.empty(0, dtype=torch.long))
    assert result.shape == (2, 0)
    gradients = torch.autograd.grad(result.sum(), (left, right))
    assert all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["left", "right"])
def test_nonfinite_descriptors_are_rejected_instead_of_treated_as_empty(bad, side):
    left = torch.ones((1, 3), dtype=torch.float64)
    right = torch.ones_like(left)
    (left if side == "left" else right)[0, 0] = bad
    species = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="descriptors must contain only finite"):
        normalized_dot_product(left, right, species, species)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_amplitude_is_rejected(bad):
    values = torch.ones((1, 3), dtype=torch.float64)
    species = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="amplitude must be a finite scalar"):
        normalized_dot_product(values, values, species, species, amplitude=bad)


def test_mixed_geometry_reference_derivative():
    positions = torch.tensor([[0.1, 0.2, 0.3], [1.1, 0.3, 0.2], [0.2, 1.2, 0.4]],
                             dtype=torch.float64, requires_grad=True)
    cell = torch.eye(3, dtype=positions.dtype) * 6
    species = torch.zeros(3, dtype=torch.long)
    descriptor = B2(1, n_radial=2, lmax=1)
    edges = neighbor_list(positions, cell, descriptor.cutoff, False)
    reference = torch.linspace(0.3, 1.2, descriptor.n_features,
                               dtype=positions.dtype).reshape(1, -1).requires_grad_()

    def force_block(z):
        def energy_kernel(p):
            values = descriptor(p, cell, species, pbc=False, edges=edges)
            return normalized_dot_product(values, z, species, species[:1]).sum()
        return -torch.func.jacrev(energy_kernel)(positions)

    assert torch.autograd.gradcheck(force_block, (reference,), fast_mode=True)
