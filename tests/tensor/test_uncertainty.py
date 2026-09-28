"""Sparse uncertainty definitions against native fixtures and dense algebra."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from flare.tensor import fit_sparse_gp, normalized_dot_product


@pytest.fixture(scope="module")
def reference():
    with np.load(Path(__file__).parent / "data/b2_reference.npz", allow_pickle=False) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


@pytest.mark.parametrize("case", ["molecule", "crystal", "binary", "missing_species", "isolated"])
def test_native_efs_and_local_variances(reference, case):
    q = reference
    posterior = fit_sparse_gp(
        q["Kzz"], q["Kzy"], q["y"], q["noise_variance"],
        jitter=float(q["jitter"]),
    )
    block = q[case + "__Kz_efs"]
    exact_prior = q[case + "__prior_efs_diag"]
    torch.testing.assert_close(
        posterior.sor_variance(block), q[case + "__sor_variance_efs"],
        rtol=2e-9, atol=3e-11,
    )
    torch.testing.assert_close(
        posterior.dtc_variance(block, exact_prior), q[case + "__dtc_variance_efs"],
        rtol=2e-9, atol=3e-11,
    )

    raw = q[case + "__raw_b2"]
    species = q[case + "__species"]
    inducing = q["inducing_raw_b2"]
    inducing_species = q["inducing_species"]
    amplitude = q["hyperparameters"][0]
    local_block = normalized_dot_product(
        inducing, raw, inducing_species, species, amplitude=amplitude,
    )
    local_prior = normalized_dot_product(
        raw, raw, species, species, amplitude=amplitude,
    ).diagonal()
    local = posterior.local_selection_variance(local_block, local_prior)
    torch.testing.assert_close(local, q[case + "__local_variance"], rtol=2e-9, atol=3e-11)
    torch.testing.assert_close(
        posterior.local_selection_score(local_block, local_prior, amplitude),
        (local / amplitude.square()).sqrt(),
    )


@pytest.mark.parametrize("method", ["qr", "cholesky"])
def test_dense_variance_and_gradients(method):
    dtype = torch.float64
    Kzz = torch.tensor([[1.5, .2], [.2, 1.3]], dtype=dtype)
    Kzy = torch.tensor([[.5, .2, -.3], [.1, .4, .2]], dtype=dtype)
    labels = torch.tensor([.2, -.4, .5], dtype=dtype)
    noise = torch.tensor([.08, .11, .09], dtype=dtype)
    posterior = fit_sparse_gp(Kzz, Kzy, labels, noise, method=method)
    block = torch.tensor([[.3, -.2], [.15, .4]], dtype=dtype, requires_grad=True)
    prior = torch.tensor([.7, .8], dtype=dtype, requires_grad=True)
    regularized = Kzz + posterior.effective_jitter * torch.eye(2, dtype=dtype)
    Qtt = block.T @ torch.linalg.solve(regularized, block)
    Qty = Kzy.T @ torch.linalg.solve(regularized, block)
    C = Kzy.T @ torch.linalg.solve(regularized, Kzy) + noise.diag()
    dense_sor = (Qtt - Qty.T @ torch.linalg.solve(C, Qty)).diagonal()
    dense_local = prior - Qtt.diagonal()
    torch.testing.assert_close(posterior.sor_variance(block), dense_sor)
    torch.testing.assert_close(posterior.local_selection_variance(block, prior), dense_local)
    torch.testing.assert_close(posterior.dtc_variance(block, prior), dense_local + dense_sor)
    assert torch.autograd.gradcheck(
        lambda cross, exact: posterior.dtc_variance(cross, exact), (block, prior),
    )


def test_local_selection_is_independent_of_labels_and_noise(reference):
    q = reference
    first = fit_sparse_gp(q["Kzz"], q["Kzy"], q["y"], q["noise_variance"])
    second = fit_sparse_gp(q["Kzz"], q["Kzy"], -q["y"], 4 * q["noise_variance"])
    block = normalized_dot_product(
        q["inducing_raw_b2"], q["binary__raw_b2"],
        q["inducing_species"], q["binary__species"],
        amplitude=q["hyperparameters"][0],
    )
    prior = normalized_dot_product(
        q["binary__raw_b2"], q["binary__raw_b2"],
        q["binary__species"], q["binary__species"],
        amplitude=q["hyperparameters"][0],
    ).diagonal()
    torch.testing.assert_close(first.local_selection_variance(block, prior),
                               second.local_selection_variance(block, prior))
    assert not torch.allclose(first.sor_variance(block), second.sor_variance(block))


def test_empty_inducing_set_and_invalid_prior():
    dtype = torch.float64
    posterior = fit_sparse_gp(
        torch.empty((0, 0), dtype=dtype), torch.empty((0, 1), dtype=dtype),
        torch.zeros(1, dtype=dtype), torch.ones(1, dtype=dtype),
    )
    block = torch.empty((0, 2), dtype=dtype)
    prior = torch.tensor([.4, .2], dtype=dtype)
    torch.testing.assert_close(posterior.sor_variance(block), torch.zeros(2, dtype=dtype))
    torch.testing.assert_close(posterior.local_selection_variance(block, prior), prior)
    torch.testing.assert_close(posterior.dtc_variance(block, prior), prior)
    with pytest.raises(ValueError, match="one entry per test output"):
        posterior.dtc_variance(block, prior[:1])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        posterior.dtc_variance(block, torch.tensor([float("nan"), .2], dtype=dtype))
    with pytest.raises(ValueError, match="finite, nonzero"):
        posterior.local_selection_score(block, prior, 0.0)


def test_inconsistent_prior_is_not_silently_clipped():
    dtype = torch.float64
    posterior = fit_sparse_gp(
        torch.ones((1, 1), dtype=dtype), torch.empty((1, 0), dtype=dtype),
        torch.empty(0, dtype=dtype), torch.empty(0, dtype=dtype),
    )
    block = torch.ones((1, 1), dtype=dtype)
    with pytest.raises(ValueError, match="negative beyond floating-point roundoff"):
        posterior.local_selection_variance(block, torch.tensor([.1], dtype=dtype))
