"""Mean inference differentiates a scalar with the fitted posterior fixed."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from flare.tensor import B2, predict_mean_efs


@pytest.fixture(scope="module")
def reference():
    with np.load(Path(__file__).parent / "data/b2_reference.npz") as archive:
        return {key: torch.as_tensor(archive[key].copy()) for key in archive.files}


@pytest.mark.parametrize("case", ["molecule", "crystal", "binary", "missing_species", "isolated"])
def test_scalar_backward_matches_native_mean(reference, case):
    q = reference
    model = B2(2, 3, 2, 3.2)
    result = predict_mean_efs(
        model, q["inducing_raw_b2"], q["inducing_species"], q["alpha"],
        q[case + "__positions"], q[case + "__cell"], q[case + "__species"],
        pbc=q[case + "__pbc"], amplitude=q["hyperparameters"][0], stress=True,
    )
    full = torch.cat((result.energy[None], result.forces.flatten(), result.stress))
    torch.testing.assert_close(full, q[case + "__mean_efs"], atol=2e-10, rtol=2e-10)
    assert not full.requires_grad


def test_offsets_and_requested_quantities(reference):
    q = reference
    model = B2(2, 3, 2, 3.2)
    args = (model, q["inducing_raw_b2"], q["inducing_species"], q["alpha"],
            q["binary__positions"], q["binary__cell"], q["binary__species"])
    normal = predict_mean_efs(*args, pbc=False, amplitude=1.3, stress=True)
    shifted = predict_mean_efs(*args, pbc=False, amplitude=1.3, stress=True,
                               atomic_offsets=[-0.2, 1.1])
    torch.testing.assert_close(shifted.energy, normal.energy + 1.8)
    torch.testing.assert_close(shifted.forces, normal.forces)
    torch.testing.assert_close(shifted.stress, normal.stress)
    energy_only = predict_mean_efs(*args, pbc=False, amplitude=1.3, forces=False)
    assert energy_only.forces is None and energy_only.stress is None
    torch.testing.assert_close(energy_only.energy, normal.energy)
    stress_only = predict_mean_efs(*args, pbc=False, amplitude=1.3, forces=False, stress=True)
    torch.testing.assert_close(stress_only.stress, normal.stress)
    assert stress_only.forces is None


def test_geometry_derivatives_hold_fitted_state_fixed(reference):
    q = reference
    model = B2(2, 3, 2, 3.2)
    positions = q["molecule__positions"].clone().requires_grad_()
    cell = q["molecule__cell"].clone().requires_grad_()
    # Make fitted quantities depend on the input deliberately. These paths
    # must not contribute to forces or higher coordinate derivatives.
    scale = positions.sum() / positions.detach().sum()
    result = predict_mean_efs(
        model, q["inducing_raw_b2"] * scale, q["inducing_species"], q["alpha"] * scale,
        positions, cell, q["molecule__species"], pbc=False,
        amplitude=q["hyperparameters"][0] * scale, create_graph=True, stress=True,
    )
    expected_forces = q["molecule__mean_efs"][1:-6].reshape(-1, 3)
    torch.testing.assert_close(result.forces, expected_forces, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(-torch.autograd.grad(result.energy, positions, retain_graph=True)[0],
                              result.forces, atol=2e-10, rtol=2e-10)
    hessian = torch.autograd.grad(result.forces.square().sum(), positions)[0]
    assert torch.isfinite(hessian).all()


def test_no_references_or_neighbors_produces_offset_energy():
    model = B2(2, 2, 1)
    z = torch.zeros((0, model.n_features), dtype=torch.float64)
    zs = torch.empty(0, dtype=torch.long)
    positions = torch.zeros((1, 3), dtype=torch.float64)
    cell = torch.zeros((3, 3), dtype=torch.float64)
    result = predict_mean_efs(model, z, zs, torch.zeros(0, dtype=torch.float64),
                              positions, cell, torch.tensor([1]), pbc=False,
                              atomic_offsets=[0., 2.])
    assert result.energy == 2
    assert torch.count_nonzero(result.forces) == 0
    with pytest.raises(ValueError, match="volume"):
        predict_mean_efs(model, z, zs, torch.zeros(0, dtype=torch.float64),
                          positions, cell, torch.tensor([1]), pbc=False, stress=True)


@pytest.mark.parametrize("bad_code", [-1, 2])
@pytest.mark.parametrize("forces", [False, True])
def test_inducing_species_outside_global_vocabulary_are_rejected(reference, bad_code, forces):
    q = reference
    inducing_species = q["inducing_species"].clone()
    inducing_species[0] = bad_code
    with pytest.raises(ValueError, match="inducing species code outside the fixed global vocabulary"):
        predict_mean_efs(
            B2(2, 3, 2, 3.2), q["inducing_raw_b2"], inducing_species, q["alpha"],
            q["molecule__positions"], q["molecule__cell"], q["molecule__species"],
            pbc=False, forces=forces,
        )
