"""Independent dense Gaussian algebra checks for the native reference fixture.

These tests need only NumPy and pytest: no native module or Torch import. Tiny
observation-space solves deliberately differ from FLARE's inducing-space QR.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"
CASES = ["molecule", "crystal", "binary", "missing_species", "isolated"]


@pytest.fixture(scope="module")
def reference():
    with np.load(DATA / "b2_reference.npz", allow_pickle=False) as archive:
        yield {key: archive[key] for key in archive.files}


def normalized(values):
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values), where=norm >= 1e-8)


def kernel(left, right, left_species, right_species, sigma):
    return sigma**2 * (normalized(left) @ normalized(right).T)**2 * (
        left_species[:, None] == right_species[None, :])


def dense_system(reference, hyperparameters=None):
    hyps = reference["hyperparameters"] if hyperparameters is None else hyperparameters
    ratio = (hyps[0] / reference["hyperparameters"][0])**2
    kzz = reference["Kzz"] * ratio
    kzy = reference["Kzy"] * ratio
    regularized = kzz + float(reference["jitter"]) * np.eye(len(kzz))
    q = kzy.T @ np.linalg.solve(regularized, kzy)
    variance = (hyps[1:][reference["noise_kind"]] * reference["relative_noise"])**2
    covariance = q + np.diag(variance)
    cholesky = np.linalg.cholesky(covariance)
    whitened_y = np.linalg.solve(cholesky, reference["y"])
    likelihood = -0.5 * (whitened_y @ whitened_y +
                         2 * np.log(np.diag(cholesky)).sum() +
                         len(variance) * np.log(2 * np.pi))
    return regularized, covariance, kzy, likelihood


def dense_gradient(reference):
    regularized, covariance, kzy, _ = dense_system(reference)
    inverse_y = np.linalg.solve(covariance, reference["y"])
    sensitivity = np.outer(inverse_y, inverse_y) - np.linalg.solve(covariance, np.eye(len(covariance)))
    solved = np.linalg.solve(regularized, kzy)
    q = kzy.T @ solved
    # Jitter is an absolute constant; do not differentiate it with signal_std.
    derivative = 2 / reference["hyperparameters"][0] * (
        2 * q - solved.T @ reference["Kzz"] @ solved)
    gradient = [0.5 * np.sum(sensitivity * derivative)]
    for kind in range(3):
        derivative_diag = (2 * reference["hyperparameters"][kind + 1] *
                           reference["relative_noise"]**2 * (reference["noise_kind"] == kind))
        gradient.append(0.5 * np.diag(sensitivity) @ derivative_diag)
    return np.array(gradient)


def test_fixture_provenance():
    metadata = json.loads((DATA / "b2_reference.json").read_text())
    assert metadata["baseline_revision"] == "199273867d48f3a91415fa44da6c0dfc5ead759f"
    assert metadata["npz_sha256"] == hashlib.sha256((DATA / "b2_reference.npz").read_bytes()).hexdigest()
    assert metadata["build"]["baseline_revision"] == metadata["baseline_revision"]
    assert "src/flare_pp/bffs/sparse_gp.cpp" in metadata["build"]["source_hashes"]


def test_kernel_from_raw_descriptors(reference):
    refs = reference["inducing_raw_b2"]
    species = reference["inducing_species"]
    sigma = reference["hyperparameters"][0]
    np.testing.assert_allclose(kernel(refs, refs, species, species, sigma), reference["Kzz"], atol=2e-13)
    for name in CASES:
        local = kernel(refs, reference[name + "__raw_b2"], species, reference[name + "__species"], sigma)
        np.testing.assert_allclose(local.sum(axis=1), reference[name + "__Kz_efs"][:, 0], atol=2e-13)
    stacked = np.concatenate([reference[name + "__Kz_efs"] for name in ["molecule", "binary"]], axis=1)
    np.testing.assert_allclose(stacked, reference["Kzy"], atol=2e-13)


def test_dense_low_rank_likelihood_and_noise(reference):
    _, _, _, likelihood = dense_system(reference)
    np.testing.assert_allclose(likelihood, reference["log_marginal_likelihood"], rtol=2e-12, atol=2e-12)
    sigma = reference["hyperparameters"][1:][reference["noise_kind"]]
    np.testing.assert_allclose((sigma * reference["relative_noise"])**2, reference["noise_variance"], atol=2e-15)


@pytest.mark.parametrize("name", CASES)
def test_dense_predictions_and_distinct_uncertainties(reference, name):
    regularized, covariance, kzy, _ = dense_system(reference)
    kzt = reference[name + "__Kz_efs"]
    q_train_test = kzy.T @ np.linalg.solve(regularized, kzt)
    q_test_test = kzt.T @ np.linalg.solve(regularized, kzt)
    mean = q_train_test.T @ np.linalg.solve(covariance, reference["y"])
    sor = np.diag(q_test_test - q_train_test.T @ np.linalg.solve(covariance, q_train_test))
    dtc = reference[name + "__prior_efs_diag"] - np.diag(q_test_test) + sor
    for key, actual in [("mean_efs", mean), ("sor_variance_efs", sor), ("dtc_variance_efs", dtc)]:
        np.testing.assert_allclose(actual, reference[name + "__" + key], rtol=2e-10, atol=3e-12, err_msg=name + " " + key)
    raw = reference[name + "__raw_b2"]
    species = reference[name + "__species"]
    sigma = reference["hyperparameters"][0]
    kxz = kernel(raw, reference["inducing_raw_b2"], species, reference["inducing_species"], sigma)
    self_kernel = np.diag(kernel(raw, raw, species, species, sigma))
    local = self_kernel - np.einsum("ij,ji->i", kxz, np.linalg.solve(regularized, kxz.T))
    np.testing.assert_allclose(local, reference[name + "__local_variance"], atol=2e-12)
    assert np.min(dtc) >= -3e-12
    assert np.min(local) >= -3e-12


def test_native_signal_gradient_has_documented_jitter_discrepancy(reference):
    exact = dense_gradient(reference)
    np.testing.assert_allclose(exact, reference["likelihood_gradient_dense"], rtol=2e-11, atol=3e-11)
    regularized, _, _, _ = dense_system(reference)
    derivative = 2 / reference["hyperparameters"][0] * reference["Kzz"]
    # Native sparse_gp.cpp uses Kuu^{-1} instead of (Kuu+jI)^{-1} in this
    # complexity derivative only. Preserve and explain the native result;
    # a Torch implementation should target the exact dense gradient above.
    correction = 0.5 * np.trace(np.linalg.solve(reference["Kzz"], derivative) -
                                np.linalg.solve(regularized, derivative))
    expected_native = exact.copy()
    expected_native[0] += correction
    np.testing.assert_allclose(expected_native, reference["likelihood_gradient"], rtol=2e-11, atol=3e-11)


@pytest.mark.parametrize("step", [2e-5, 1e-5])
def test_dense_likelihood_gradient_finite_difference(reference, step):
    exact = dense_gradient(reference)
    finite_difference = []
    for index in range(4):
        shift = np.zeros(4)
        shift[index] = step
        plus = dense_system(reference, reference["hyperparameters"] + shift)[-1]
        minus = dense_system(reference, reference["hyperparameters"] - shift)[-1]
        finite_difference.append((plus - minus) / (2 * step))
    np.testing.assert_allclose(exact, finite_difference, rtol=2e-7, atol=2e-7)
