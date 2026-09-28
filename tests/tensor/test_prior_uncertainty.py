"""Exact E/F/stress prior diagonals from independent geometry derivatives."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from flare.tensor import (
    B2, ObservationLayout, fit_sparse_gp, predict_variance_efs,
    prior_observation_variance,
)
from flare.tensor.kernels import normalized_dot_product


@pytest.fixture(scope="module")
def reference():
    with np.load(Path(__file__).parent / "data/b2_reference.npz", allow_pickle=False) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


@pytest.mark.parametrize("case", ["molecule", "crystal", "binary", "missing_species", "isolated"])
def test_exact_prior_diagonal_matches_native_and_dtc(reference, case):
    q = reference
    descriptor = B2(2, 3, 2, 3.2)
    positions = q[case + "__positions"]
    layout = ObservationLayout.full(len(positions))
    prior = prior_observation_variance(
        descriptor, positions, q[case + "__cell"], q[case + "__species"],
        layout, pbc=q[case + "__pbc"], amplitude=q["hyperparameters"][0],
    )
    torch.testing.assert_close(prior, q[case + "__prior_efs_diag"], rtol=2e-9, atol=3e-10)
    posterior = fit_sparse_gp(q["Kzz"], q["Kzy"], q["y"], q["noise_variance"])
    torch.testing.assert_close(
        posterior.dtc_variance(q[case + "__Kz_efs"], prior),
        q[case + "__dtc_variance_efs"], rtol=2e-9, atol=3e-10,
    )


@pytest.mark.parametrize("variance_type", ["SOR", "DTC"])
@pytest.mark.parametrize("case", ["molecule", "crystal", "binary", "missing_species", "isolated"])
def test_full_geometry_prediction_matches_native_variance_efs(reference, case, variance_type):
    q = reference
    positions = q[case + "__positions"]
    posterior = fit_sparse_gp(q["Kzz"], q["Kzy"], q["y"], q["noise_variance"])
    variance = predict_variance_efs(
        posterior, B2(2, 3, 2, 3.2), q["inducing_raw_b2"],
        q["inducing_species"], positions, q[case + "__cell"],
        q[case + "__species"], pbc=q[case + "__pbc"],
        amplitude=q["hyperparameters"][0], variance_type=variance_type,
        chunk_size=2,
    )
    assert variance.shape == (1 + 3 * len(positions) + 6,)
    torch.testing.assert_close(
        variance, q[case + "__" + variance_type.lower() + "_variance_efs"],
        rtol=2e-9, atol=3e-10,
    )


def test_grouped_inference_matches_native_full_dtc(reference):
    q = reference
    posterior = fit_sparse_gp(q["Kzz"], q["Kzy"], q["y"], q["noise_variance"])
    result = predict_variance_efs(
        posterior, B2(2, 3, 2, 3.2), q["inducing_raw_b2"],
        q["inducing_species"], q["binary__positions"], q["binary__cell"],
        q["binary__species"], pbc=q["binary__pbc"],
        amplitude=q["hyperparameters"][0], variance_type="DTC",
        assembly="grouped", chunk_size=2,
    )
    torch.testing.assert_close(result, q["binary__dtc_variance_efs"],
                               rtol=2e-9, atol=3e-10)


def test_auto_assembly_preserves_geometry_gradients(reference):
    q = reference
    posterior = fit_sparse_gp(q["Kzz"], q["Kzy"], q["y"], q["noise_variance"]).detach()
    positions = q["binary__positions"].clone().requires_grad_()
    args = (posterior, B2(2, 3, 2, 3.2), q["inducing_raw_b2"],
            q["inducing_species"], positions, q["binary__cell"],
            q["binary__species"])
    options = dict(pbc=q["binary__pbc"], amplitude=q["hyperparameters"][0],
                   variance_type="SOR", layout=ObservationLayout.energy_only(len(positions)),
                   chunk_size=2)
    automatic = predict_variance_efs(*args, **options)
    explicit = predict_variance_efs(*args, **options, assembly="lambda")
    torch.testing.assert_close(automatic, explicit)
    torch.testing.assert_close(
        torch.autograd.grad(automatic.sum(), positions)[0],
        torch.autograd.grad(explicit.sum(), positions)[0],
    )


def test_masked_prior_matches_selected_full_columns(reference):
    q = reference
    positions = q["binary__positions"]
    cell, species, pbc = (q["binary__" + key] for key in ("cell", "species", "pbc"))
    descriptor = B2(2, 3, 2, 3.2)
    full = prior_observation_variance(
        descriptor, positions, cell, species, ObservationLayout.full(len(positions)),
        pbc=pbc, amplitude=q["hyperparameters"][0],
    )
    layout = ObservationLayout.from_masks(
        len(positions), energy=True,
        force_mask=[[True, False, False]] + [[False, False, False]] * (len(positions) - 1),
        stress_mask=[False, True, False, False, False, True],
    )
    selected = prior_observation_variance(
        descriptor, positions, cell, species, layout, pbc=pbc,
        amplitude=q["hyperparameters"][0],
    )
    torch.testing.assert_close(selected, full[layout.full_indices])


@pytest.mark.parametrize("power", [1, 2, 3])
def test_mixed_force_and_stress_derivative_of_independent_kernel_arguments(power):
    dtype = torch.float64
    descriptor = B2(1, 2, 1, 3.0)
    positions = torch.tensor([[0., 0., 0.], [1.1, .3, -.2]], dtype=dtype)
    cell = torch.eye(3, dtype=dtype) * 5
    species = torch.zeros(2, dtype=torch.long)
    zero = torch.zeros((), dtype=dtype)
    amplitude = torch.tensor(1.3, dtype=dtype)

    def deformed(displacement, shear):
        changed = positions.clone()
        changed[1, 0] = changed[1, 0] + displacement
        strain = torch.zeros_like(cell)
        strain[0, 1] = shear
        deformation = torch.eye(3, dtype=dtype) + strain
        return changed @ deformation.T, cell @ deformation.T

    def covariance(left, right, kind):
        left_positions, left_cell = deformed(left if kind == "force" else zero,
                                             left if kind == "stress" else zero)
        right_positions, right_cell = deformed(right if kind == "force" else zero,
                                               right if kind == "stress" else zero)
        left_raw = descriptor(left_positions, left_cell, species, pbc=False)
        right_raw = descriptor(right_positions, right_cell, species, pbc=False)
        return normalized_dot_product(left_raw, right_raw, species, species,
                                      amplitude=amplitude, power=power).sum()

    mixed_force = torch.func.jacrev(
        torch.func.jacrev(lambda x, y: covariance(x, y, "force"), argnums=0),
        argnums=1,
    )(zero, zero)
    mixed_stress = torch.func.jacrev(
        torch.func.jacrev(lambda x, y: covariance(x, y, "stress"), argnums=0),
        argnums=1,
    )(zero, zero) / torch.linalg.det(cell).square()
    layout = ObservationLayout.from_masks(
        2, force_mask=[[False, False, False], [True, False, False]],
        stress_mask=[False, True, False, False, False, False],
    )
    actual = prior_observation_variance(
        descriptor, positions, cell, species, layout, pbc=False,
        amplitude=amplitude, power=power,
    )
    torch.testing.assert_close(actual, torch.stack((mixed_force, mixed_stress)),
                               rtol=2e-9, atol=3e-10)


def test_prior_geometry_gradcheck():
    descriptor = B2(1, 2, 1, 3.0)
    positions = torch.tensor([[0., 0., 0.], [1.1, .3, -.2]],
                             dtype=torch.float64, requires_grad=True)
    cell = (torch.eye(3, dtype=torch.float64) * 5).requires_grad_()
    species = torch.zeros(2, dtype=torch.long)
    layout = ObservationLayout.from_masks(
        2, force_mask=[[False, False, False], [True, False, False]],
        stress_mask=[False, True, False, False, False, False],
    )
    assert torch.autograd.gradcheck(
        lambda p, c: prior_observation_variance(
            descriptor, p, c, species, layout, pbc=False, amplitude=1.3,
        ),
        (positions, cell), eps=1e-6, atol=2e-5, rtol=2e-4,
    )


def test_prior_requires_valid_layout_and_positive_stress_volume():
    descriptor = B2(1, 2, 1)
    positions = torch.zeros((1, 3), dtype=torch.float64)
    species = torch.zeros(1, dtype=torch.long)
    cell = torch.zeros((3, 3), dtype=torch.float64)
    energy = ObservationLayout.energy_only(1)
    torch.testing.assert_close(prior_observation_variance(
        descriptor, positions, cell, species, energy, pbc=False,
    ), torch.zeros(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="positive cell volume"):
        prior_observation_variance(
            descriptor, positions, cell, species, ObservationLayout.full(1), pbc=False,
        )
    with pytest.raises(ValueError, match="layout must describe"):
        prior_observation_variance(
            descriptor, positions, cell, species, ObservationLayout.energy_only(2),
            pbc=False,
        )


def test_empty_structure_full_uncertainty(reference):
    q = reference
    descriptor = B2(2, 3, 2, 3.2)
    positions = torch.empty((0, 3), dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * 5
    species = torch.empty(0, dtype=torch.long)
    layout = ObservationLayout.full(0)
    prior = prior_observation_variance(
        descriptor, positions, cell, species, layout, pbc=False,
    )
    torch.testing.assert_close(prior, torch.zeros(7, dtype=torch.float64))
    posterior = fit_sparse_gp(q["Kzz"], q["Kzy"], q["y"], q["noise_variance"])
    for variance_type in ("SOR", "DTC"):
        result = predict_variance_efs(
            posterior, descriptor, q["inducing_raw_b2"], q["inducing_species"],
            positions, cell, species, pbc=False, variance_type=variance_type,
        )
        torch.testing.assert_close(result, prior)


def test_float32_no_grad_and_amplitude_derivative(reference):
    q = reference
    descriptor = B2(2, 3, 2, 3.2)
    positions = q["binary__positions"].float()
    cell = q["binary__cell"].float()
    species = q["binary__species"]
    pbc = q["binary__pbc"]
    layout = ObservationLayout.from_masks(
        len(positions), energy=True,
        force_mask=[[True, False, False]] + [[False] * 3] * (len(positions) - 1),
        stress_mask=[True, False, False, False, False, False],
    )
    amplitude = torch.tensor(1.3, dtype=torch.float32, requires_grad=True)
    values = prior_observation_variance(
        descriptor, positions, cell, species, layout, pbc=pbc, amplitude=amplitude,
    )
    assert values.dtype == torch.float32 and bool(torch.isfinite(values).all())
    gradient, = torch.autograd.grad(values.sum(), amplitude)
    torch.testing.assert_close(gradient, 2 * values.detach().sum() / amplitude.detach())
    with torch.no_grad():
        detached = prior_observation_variance(
            descriptor, positions, cell, species, layout, pbc=pbc,
            amplitude=amplitude.detach(),
        )
    torch.testing.assert_close(detached, values.detach())
