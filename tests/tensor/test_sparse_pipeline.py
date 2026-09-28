"""Raw structures -> masked observation kernels -> sparse fit -> scalar means."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from flare.tensor import B2, StructureBatch
from flare.tensor.kernels import normalized_dot_product
from flare.tensor.linalg import fit_sparse_gp
from flare.tensor.observations import ObservationLayout, assemble_observation_covariance
from flare.tensor.prediction import predict_mean_efs


@pytest.fixture(scope="module")
def reference():
    with np.load(Path(__file__).parent / "data/b2_reference.npz") as archive:
        return {key: torch.as_tensor(archive[key].copy()) for key in archive.files}


def assemble(reference, hyperparameters, mode="full", chunk_size=2):
    q = reference
    descriptor = B2(2, 3, 2, 3.2)
    names = ("molecule", "binary")
    sizes = [len(q[name + "__positions"]) for name in names]
    batch = StructureBatch(
        torch.cat([q[name + "__positions"] for name in names]),
        torch.cat([q[name + "__species"] for name in names]),
        torch.stack([q[name + "__cell"] for name in names]),
        torch.stack([q[name + "__pbc"] for name in names]),
        torch.tensor([0, sizes[0], sum(sizes)]),
    )
    layouts, labels, noise, indices = [], [], [], []
    offset = 0
    for name, n_atoms, relative in zip(names, sizes, ([1., 1., 1.], [1.2, .8, 1.1])):
        if mode == "full":
            layout = ObservationLayout.full(n_atoms)
        elif mode == "energy":
            layout = ObservationLayout.energy_only(n_atoms)
        else:
            force_mask = torch.zeros((n_atoms, 3), dtype=torch.bool)
            force_mask[0, 0] = True
            force_mask[-1, 2] = True
            layout = ObservationLayout.from_masks(
                n_atoms, energy=name == "molecule", force_mask=force_mask,
                stress_mask=torch.tensor([False, True, False, True, False, False]),
            )
        layouts.append(layout)
        count = 1 + 3 * n_atoms + 6
        full_labels = q["y"][offset:offset + count]
        labels.append(layout.pack_labels(
            energy=full_labels[0], forces=full_labels[1:-6].reshape(n_atoms, 3),
            stress=full_labels[-6:],
        ))
        noise.append(layout.noise_variance(hyperparameters[1:], torch.tensor(relative, dtype=torch.float64)))
        indices.append(layout.full_indices + offset)
        offset += count
    Kzz = normalized_dot_product(q["inducing_raw_b2"], q["inducing_raw_b2"],
                                 q["inducing_species"], q["inducing_species"], hyperparameters[0])
    Kzy = assemble_observation_covariance(
        descriptor, q["inducing_raw_b2"], q["inducing_species"], batch, layouts,
        amplitude=hyperparameters[0], chunk_size=chunk_size,
    )
    return descriptor, Kzz, Kzy, torch.cat(labels), torch.cat(noise), torch.cat(indices)


@pytest.mark.parametrize("mode", ["energy", "masked", "full"])
def test_fit_from_geometry_agrees_with_dense_native_blocks(reference, mode):
    q = reference
    descriptor, Kzz, Kzy, y, noise, indices = assemble(q, q["hyperparameters"], mode)
    torch.testing.assert_close(Kzz, q["Kzz"], atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(Kzy, q["Kzy"][:, indices], atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(y, q["y"][indices])
    torch.testing.assert_close(noise, q["noise_variance"][indices])
    posterior = fit_sparse_gp(Kzz, Kzy, y, noise, jitter=float(q["jitter"]))

    # Independent observation-space calculation from the saved C++ blocks.
    native_Kzy = q["Kzy"][:, indices]
    regularized = q["Kzz"] + q["jitter"] * torch.eye(len(Kzz), dtype=Kzz.dtype)
    C = native_Kzy.T @ torch.linalg.solve(regularized, native_Kzy) + noise.diag()
    beta = torch.linalg.solve(C, y)
    expected_nll = .5 * (y @ beta + torch.linalg.slogdet(C)[1] + len(y) * np.log(2 * np.pi))
    torch.testing.assert_close(posterior.negative_log_likelihood, expected_nll, atol=2e-9, rtol=2e-10)
    for case in ("molecule", "crystal", "binary", "missing_species", "isolated"):
        native_test = q[case + "__Kz_efs"]
        expected = (native_Kzy.T @ torch.linalg.solve(regularized, native_test)).T @ beta
        mean = predict_mean_efs(
            descriptor, q["inducing_raw_b2"], q["inducing_species"], posterior.alpha,
            q[case + "__positions"], q[case + "__cell"], q[case + "__species"],
            pbc=q[case + "__pbc"], amplitude=q["hyperparameters"][0], stress=True,
        )
        actual = torch.cat((mean.energy[None], mean.forces.flatten(), mean.stress))
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


def test_likelihood_gradient_through_force_and_stress_assembly(reference):
    q = reference
    hyperparameters = q["hyperparameters"].clone().requires_grad_()
    _, Kzz, Kzy, y, noise, _ = assemble(q, hyperparameters)
    posterior = fit_sparse_gp(Kzz, Kzy, y, noise, jitter=float(q["jitter"]))
    gradient = torch.autograd.grad(-posterior.negative_log_likelihood, hyperparameters)[0]
    torch.testing.assert_close(gradient, q["likelihood_gradient_dense"], atol=2e-9, rtol=2e-9)
    assert abs(gradient[0] - q["likelihood_gradient"][0]) > 5e-7
