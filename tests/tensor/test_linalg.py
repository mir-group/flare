"""Sparse QR fitting against native fixtures and independent dense algebra."""

from dataclasses import FrozenInstanceError
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from flare.tensor.linalg import fit_sparse_gp


@pytest.fixture(scope="module")
def reference():
    with np.load(Path(__file__).parent / "data" / "b2_reference.npz") as arrays:
        return {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}


def synthetic(m=3, p=8, dtype=torch.float64):
    generator = torch.Generator().manual_seed(481)
    basis = torch.randn((m, m), generator=generator, dtype=dtype)
    Kzz = basis @ basis.T + torch.eye(m, dtype=dtype) * 0.3
    Kzy = torch.randn((m, p), generator=generator, dtype=dtype)
    y = torch.randn(p, generator=generator, dtype=dtype)
    noise = torch.linspace(0.07, 0.3, p, dtype=dtype)
    return Kzz, Kzy, y, noise


def dense_fit(Kzz, Kzy, y, noise, jitter):
    """Tiny observation-space calculation used only as an independent oracle."""
    regularized = Kzz + jitter * torch.eye(len(Kzz), dtype=Kzz.dtype, device=Kzz.device)
    projected = torch.linalg.solve(regularized, Kzy)
    covariance = Kzy.T @ projected + noise.diag()
    factor = torch.linalg.cholesky(covariance)
    white_y = torch.linalg.solve_triangular(factor, y[:, None], upper=False)
    beta = torch.linalg.solve(covariance, y)
    nll = 0.5 * (white_y.square().sum() + 2 * factor.diagonal().log().sum()
                 + len(y) * math.log(2 * math.pi))
    return projected @ beta, nll


@pytest.mark.parametrize("method", ["qr", "cholesky"])
def test_fixture_fit_means_likelihood_and_factors(reference, method):
    posterior = fit_sparse_gp(
        reference["Kzz"], reference["Kzy"], reference["y"], reference["noise_variance"],
        jitter=float(reference["jitter"]), method=method,
    )
    torch.testing.assert_close(posterior.alpha, reference["alpha"], rtol=2e-10, atol=3e-12)
    torch.testing.assert_close(posterior.negative_log_likelihood,
                               -reference["log_marginal_likelihood"], rtol=2e-12, atol=2e-12)
    for case in ("molecule", "crystal", "binary", "missing_species", "isolated"):
        actual = posterior.predict_mean(reference[case + "__Kz_efs"])
        torch.testing.assert_close(actual, reference[case + "__mean_efs"], rtol=2e-10, atol=3e-12)
    assert posterior.effective_jitter == float(reference["jitter"])
    assert posterior.method == method
    L, R = posterior.inducing_cholesky, posterior.posterior_factor
    identity = torch.eye(len(L), dtype=L.dtype)
    torch.testing.assert_close(L @ L.T, reference["Kzz"] + posterior.effective_jitter * identity)
    A = torch.linalg.solve_triangular(L, reference["Kzy"], upper=False)
    A = A / reference["noise_variance"].sqrt()
    torch.testing.assert_close(R.T @ R, identity + A @ A.T)
    torch.testing.assert_close(L.T @ posterior.alpha, posterior.whitened_mean)


@pytest.mark.parametrize("method", ["qr", "cholesky"])
def test_hyperparameter_gradient_targets_exact_jittered_likelihood(reference, method):
    hyps = reference["hyperparameters"].clone().requires_grad_()
    ratio = (hyps[0] / reference["hyperparameters"][0]).square()
    noise = (hyps[1:][reference["noise_kind"]] * reference["relative_noise"]).square()
    posterior = fit_sparse_gp(reference["Kzz"] * ratio, reference["Kzy"] * ratio,
                              reference["y"], noise, method=method)
    gradient, = torch.autograd.grad(-posterior.negative_log_likelihood, hyps)
    torch.testing.assert_close(gradient, reference["likelihood_gradient_dense"],
                               rtol=2e-10, atol=3e-10)
    # The native amplitude gradient has a known 7e-7 jitter discrepancy.
    assert abs(float(gradient[0] - reference["likelihood_gradient"][0])) > 5e-7


@pytest.mark.parametrize("m,p", [(2, 7), (5, 3), (4, 4)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("method", ["qr", "cholesky"])
def test_sparse_matches_dense_for_rectangular_designs(m, p, dtype, method):
    arguments = synthetic(m, p, dtype)
    posterior = fit_sparse_gp(*arguments, method=method, jitter=1e-5)
    alpha, nll = dense_fit(*arguments, posterior.effective_jitter)
    tolerance = 2e-10 if dtype == torch.float64 else 3e-4
    torch.testing.assert_close(posterior.alpha, alpha, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(posterior.negative_log_likelihood, nll, rtol=tolerance, atol=tolerance)
    assert posterior.alpha.dtype == dtype


@pytest.mark.parametrize("method", ["qr", "cholesky"])
def test_fit_gradcheck_all_trainable_inputs(method):
    basis = torch.tensor([[0.8, 0.3], [-0.2, 1.1]], dtype=torch.float64, requires_grad=True)
    _, cross, y, noise = synthetic(2, 4)
    cross.requires_grad_()
    y.requires_grad_()
    log_noise = noise.log().requires_grad_()

    def objective(factor, block, labels, log_variance):
        Kzz = factor @ factor.T + torch.eye(2, dtype=factor.dtype) * 0.2
        posterior = fit_sparse_gp(Kzz, block, labels, log_variance.exp(), method=method)
        return posterior.negative_log_likelihood

    assert torch.autograd.gradcheck(objective, (basis, cross, y, log_noise), fast_mode=True)
    assert torch.autograd.gradgradcheck(objective, (basis, cross, y, log_noise), fast_mode=True)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_fixed_cpu_householder_qr_matches_differentiable_qr(dtype, monkeypatch):
    arguments = synthetic(12, 96, dtype)
    original = torch.geqrf
    calls = []

    def record(matrix):
        calls.append(matrix.shape)
        return original(matrix)

    monkeypatch.setattr(torch, "geqrf", record)
    fixed = fit_sparse_gp(*arguments, method="qr")
    assert calls == [(108, 12)]
    trainable = fit_sparse_gp(arguments[0].clone().requires_grad_(), *arguments[1:], method="qr")
    assert len(calls) == 1
    tolerance = 2e-12 if dtype == torch.float64 else 3e-5
    torch.testing.assert_close(fixed.alpha, trainable.alpha, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(fixed.negative_log_likelihood,
                               trainable.negative_log_likelihood,
                               rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("method", ["qr", "cholesky"])
def test_residual_quadratic_retains_finite_prior_cost_when_noise_is_tiny(method):
    # The subtractive Woodbury quadratic t.T t - (A t).T solve(B, A t)
    # rounds to zero here, losing the finite prior penalty of approximately 1.
    one = torch.ones((1, 1), dtype=torch.float64)
    variance = torch.tensor([1e-24], dtype=one.dtype)
    posterior = fit_sparse_gp(one, one, one[:, 0], variance, jitter=0, method=method)
    exact = 0.5 * (1 / (1 + variance[0]) + torch.log1p(variance[0]) + math.log(2 * math.pi))
    torch.testing.assert_close(posterior.negative_log_likelihood, exact, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(posterior.alpha, torch.ones(1, dtype=one.dtype))


@pytest.mark.parametrize("method", ["qr", "cholesky"])
def test_duplicate_references_retry_jitter_and_match_dense(method):
    Kzz = torch.ones((2, 2), dtype=torch.float64)
    Kzy = torch.tensor([[1., -0.3, 0.7], [1., -0.3, 0.7]], dtype=Kzz.dtype)
    y = torch.tensor([0.2, 0.6, -0.1], dtype=Kzz.dtype)
    noise = torch.full((3,), 0.01, dtype=Kzz.dtype)
    posterior = fit_sparse_gp(Kzz, Kzy, y, noise, jitter=0, max_jitter=1e-6, method=method)
    assert 0 < posterior.effective_jitter <= 1e-6
    _, nll = dense_fit(Kzz, Kzy, y, noise, posterior.effective_jitter)
    torch.testing.assert_close(posterior.negative_log_likelihood, nll, rtol=2e-11, atol=2e-11)
    # Coefficients for identical references are individually ill-conditioned;
    # predictions and the likelihood are the meaningful comparison.
    alpha, _ = dense_fit(Kzz, Kzy, y, noise, posterior.effective_jitter)
    torch.testing.assert_close(posterior.predict_mean(Kzy), Kzy.T @ alpha, rtol=2e-10, atol=2e-10)


def test_jitter_retries_include_the_declared_cap_and_report_the_value():
    Kzz = torch.diag(torch.tensor([1., -2e-7], dtype=torch.float64))
    _, Kzy, y, noise = synthetic(2, 3)
    posterior = fit_sparse_gp(Kzz, Kzy, y, noise, jitter=1e-8, max_jitter=5e-7)
    assert posterior.effective_jitter == 5e-7
    torch.testing.assert_close(posterior.inducing_cholesky @ posterior.inducing_cholesky.T,
                               Kzz + 5e-7 * torch.eye(2, dtype=Kzz.dtype))


def test_failed_factorization_explains_the_bound():
    Kzz = torch.diag(torch.tensor([1., -1.], dtype=torch.float64))
    _, Kzy, y, noise = synthetic(2, 3)
    with pytest.raises(RuntimeError, match="Inducing Cholesky failed.*max_jitter=0.0001"):
        fit_sparse_gp(Kzz, Kzy, y, noise)
    with pytest.raises(RuntimeError, match="last_jitter=0, max_jitter=0"):
        fit_sparse_gp(Kzz, Kzy, y, noise, jitter=0, max_jitter=0)


def test_jitter_retry_count_is_bounded_when_multiplier_makes_tiny_progress(monkeypatch):
    Kzz = torch.diag(torch.tensor([1., -2e-7], dtype=torch.float64))
    _, Kzy, y, noise = synthetic(2, 3)
    original = torch.linalg.cholesky_ex
    attempts = []

    def record(matrix, *args, **kwargs):
        attempts.append(matrix.detach().clone())
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", record)
    posterior = fit_sparse_gp(Kzz, Kzy, y, noise, jitter=1e-8, max_jitter=5e-7,
                              jitter_multiplier=1 + torch.finfo(torch.float64).eps)
    assert len(attempts) <= 16
    assert posterior.effective_jitter == 5e-7


@pytest.mark.parametrize("m,p", [(0, 0), (0, 4), (3, 0)])
@pytest.mark.parametrize("method", ["qr", "cholesky"])
def test_empty_observation_and_reference_sets(m, p, method):
    arguments = synthetic(m, p)
    posterior = fit_sparse_gp(*arguments, method=method)
    _, _, y, noise = arguments
    expected = 0.5 * ((y.square() / noise).sum() + noise.log().sum() + p * math.log(2 * math.pi))
    torch.testing.assert_close(posterior.negative_log_likelihood, expected)
    torch.testing.assert_close(posterior.alpha, torch.zeros(m, dtype=y.dtype))
    torch.testing.assert_close(posterior.predict_mean(torch.ones((m, 2), dtype=y.dtype)),
                               torch.zeros(2, dtype=y.dtype))
    if m == 0:
        assert posterior.effective_jitter == 0


def test_inference_detach_drops_training_graph_and_container_is_frozen():
    Kzz, Kzy, y, noise = synthetic()
    Kzy.requires_grad_()
    posterior = fit_sparse_gp(Kzz, Kzy, y, noise)
    inference = posterior.detach()
    for name in ("inducing_cholesky", "posterior_factor", "whitened_mean", "alpha", "negative_log_likelihood"):
        value = getattr(inference, name)
        assert not value.requires_grad
        torch.testing.assert_close(value, getattr(posterior, name))
    with pytest.raises(FrozenInstanceError):
        posterior.effective_jitter = 1.0
    test_cross = Kzy.detach().requires_grad_()
    derivative, = torch.autograd.grad(inference.predict_mean(test_cross).sum(), test_cross)
    torch.testing.assert_close(derivative, inference.alpha[:, None].expand_as(test_cross))


@pytest.mark.parametrize("argument, replacement, message", [
    (0, torch.ones(3), "square"),
    (0, torch.ones((3, 2)), "square"),
    (1, torch.ones((2, 8)), "one row"),
    (1, torch.ones(3), "one row"),
    (2, torch.ones((8, 1)), "one entry"),
    (2, torch.ones(7), "one entry"),
    (3, torch.ones((8, 1)), "same vector shape"),
    (3, torch.zeros(8), "strictly positive"),
    (3, -torch.ones(8), "strictly positive"),
    (0, torch.full((3, 3), float("nan")), "finite"),
    (1, torch.full((3, 8), float("inf")), "finite"),
    (2, torch.full((8,), float("nan")), "finite"),
    (3, torch.full((8,), float("inf")), "finite"),
])
def test_invalid_training_inputs(argument, replacement, message):
    arguments = list(synthetic())
    arguments[argument] = replacement.to(dtype=torch.float64)
    with pytest.raises(ValueError, match=message):
        fit_sparse_gp(*arguments)


def test_invalid_symmetry_dtype_device_and_non_tensor():
    Kzz, Kzy, y, noise = synthetic()
    asymmetric = Kzz.clone()
    asymmetric[0, 1] += 1e-4
    with pytest.raises(ValueError, match="symmetric"):
        fit_sparse_gp(asymmetric, Kzy, y, noise)
    with pytest.raises(ValueError, match="same dtype and device"):
        fit_sparse_gp(Kzz, Kzy.float(), y, noise)
    with pytest.raises(ValueError, match="same dtype and device"):
        fit_sparse_gp(Kzz, Kzy.to("meta"), y, noise)
    with pytest.raises(ValueError, match="float32 or float64"):
        fit_sparse_gp(Kzz.long(), Kzy, y, noise)
    with pytest.raises(ValueError, match="must be a tensor"):
        fit_sparse_gp(Kzz.tolist(), Kzy, y, noise)


@pytest.mark.parametrize("options", [
    {"jitter": -1}, {"jitter": 1e-2}, {"jitter": float("nan")},
    {"jitter": True}, {"max_jitter": -1}, {"max_jitter": float("inf")},
    {"jitter_multiplier": 1}, {"jitter_multiplier": float("nan")},
    {"method": "inverse"},
])
def test_invalid_solver_options(options):
    with pytest.raises(ValueError):
        fit_sparse_gp(*synthetic(), **options)


def test_prediction_validation():
    posterior = fit_sparse_gp(*synthetic())
    for block, message in [
        (torch.ones(3), "matrix"),
        (torch.ones((2, 4)), "one row"),
        (torch.ones((3, 4)), "dtype and device"),
        (torch.full((3, 4), float("nan"), dtype=torch.float64), "finite"),
    ]:
        with pytest.raises(ValueError, match=message):
            posterior.predict_mean(block)


def test_training_never_requires_an_explicit_inverse(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Sparse fitting must use factors and solves")
    monkeypatch.setattr(torch, "inverse", forbidden)
    monkeypatch.setattr(torch.linalg, "inv", forbidden)
    monkeypatch.setattr(torch, "cholesky_inverse", forbidden)
    posterior = fit_sparse_gp(*synthetic())
    assert torch.isfinite(posterior.negative_log_likelihood)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_fit_matches_cpu():
    inputs = synthetic()
    cpu = fit_sparse_gp(*inputs)
    gpu = fit_sparse_gp(*(value.cuda() for value in inputs))
    torch.testing.assert_close(gpu.alpha.cpu(), cpu.alpha)
    torch.testing.assert_close(gpu.negative_log_likelihood.cpu(), cpu.negative_log_likelihood)
