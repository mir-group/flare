"""Differentiable inducing-space solves for FLARE's low-rank Gaussian model.

The observation covariance is ``Kzy.T @ solve(Kzz + jitter * I, Kzy) + D``.
No observation-space covariance or explicit inverse is formed. This module
implements the existing low-rank likelihood, without a variational correction.
"""

from dataclasses import dataclass, fields
import math
from numbers import Real

import torch


@dataclass(frozen=True)
class SparsePosterior:
    """Factors from one fit; tensors retain their autograd history.

    ``inducing_cholesky`` is lower triangular. ``posterior_factor`` is upper
    triangular, with ``R.T @ R = I + A @ A.T`` for the whitened design ``A``.
    QR may give R a negative diagonal; its sign has no probabilistic meaning.
    The frozen container prevents field replacement, not tensor mutation.
    Call :meth:`detach` to drop the fit's graph before caching inference state.
    """

    inducing_cholesky: torch.Tensor
    posterior_factor: torch.Tensor
    whitened_mean: torch.Tensor
    alpha: torch.Tensor
    negative_log_likelihood: torch.Tensor
    effective_jitter: float
    method: str

    def _validate_test_block(self, Kztest):
        if not isinstance(Kztest, torch.Tensor) or Kztest.ndim != 2:
            raise ValueError("Kztest must be a matrix")
        if Kztest.shape[0] != self.alpha.numel():
            raise ValueError("Kztest must have one row per inducing reference")
        if Kztest.dtype != self.alpha.dtype or Kztest.device != self.alpha.device:
            raise ValueError("Kztest must match the posterior dtype and device")
        if not bool(torch.isfinite(Kztest).all()):
            raise ValueError("Kztest must contain only finite values")

    def _validate_prior_variance(self, prior_variance, n_outputs):
        if not isinstance(prior_variance, torch.Tensor) or prior_variance.shape != (n_outputs,):
            raise ValueError("prior_variance must have one entry per test output")
        if prior_variance.dtype != self.alpha.dtype or prior_variance.device != self.alpha.device:
            raise ValueError("prior_variance must match the posterior dtype and device")
        if not bool(torch.isfinite(prior_variance).all()) or bool((prior_variance < 0).any()):
            raise ValueError("prior_variance must be finite and nonnegative")

    def _whitened_test_block(self, Kztest):
        self._validate_test_block(Kztest)
        return torch.linalg.solve_triangular(
            self.inducing_cholesky, Kztest, upper=False,
        )

    def predict_mean(self, Kztest):
        """Contract an ``(M, T)`` reference-to-output block with the fitted mean.

        Columns may be energies, forces, stresses, or any other linear outputs
        of the same GP. Their kernel and inducing references must match the fit.
        """
        self._validate_test_block(Kztest)
        return Kztest.T @ self.alpha

    def sor_variance(self, Kztest):
        """Return the low-rank posterior's latent variance for each test column.

        ``Kztest`` has shape ``(M, T)``. This does not include observation
        noise. The squared triangular-solve norm stays nonnegative without
        constructing a test-by-test covariance matrix.
        """
        whitened = self._whitened_test_block(Kztest)
        posterior_whitened = torch.linalg.solve_triangular(
            self.posterior_factor.T, whitened, upper=False,
        )
        return posterior_whitened.square().sum(dim=0)

    def local_selection_variance(self, Kztest, prior_variance):
        """Return the inducing-coverage residual, independent of labels/noise.

        ``prior_variance`` is the exact latent prior diagonal for the same
        ``T`` outputs. For force/stress outputs it must differentiate two
        independent kernel arguments; it is not the diagonal of ``Kztest``.
        The inducing factor includes this fit's effective jitter.
        """
        whitened = self._whitened_test_block(Kztest)
        self._validate_prior_variance(prior_variance, Kztest.shape[1])
        projected = whitened.square().sum(dim=0)
        return _nonnegative_residual(prior_variance - projected, prior_variance,
                                     projected, "local selection variance")

    def dtc_variance(self, Kztest, prior_variance):
        """Return latent DTC variance: local residual plus SOR variance."""
        whitened = self._whitened_test_block(Kztest)
        self._validate_prior_variance(prior_variance, Kztest.shape[1])
        projected = whitened.square().sum(dim=0)
        local = _nonnegative_residual(prior_variance - projected, prior_variance,
                                      projected, "local selection variance")
        posterior_whitened = torch.linalg.solve_triangular(
            self.posterior_factor.T, whitened, upper=False,
        )
        return local + posterior_whitened.square().sum(dim=0)

    def local_selection_score(self, Kztest, prior_variance, amplitude):
        """Return ``sqrt(local_variance) / abs(amplitude)`` for selection."""
        amplitude = torch.as_tensor(amplitude, dtype=self.alpha.dtype,
                                    device=self.alpha.device)
        if amplitude.ndim != 0 or not bool(torch.isfinite(amplitude).all()) or bool(amplitude == 0):
            raise ValueError("amplitude must be a finite, nonzero scalar")
        return self.local_selection_variance(Kztest, prior_variance).sqrt() / amplitude.abs()

    def detach(self):
        """Return graph-free factors sharing storage with this posterior."""
        return type(self)(**{
            field.name: value.detach() if isinstance(value, torch.Tensor) else value
            for field in fields(self)
            for value in (getattr(self, field.name),)
        })


def _nonnegative_residual(value, prior, projected, name):
    # A negative result beyond roundoff indicates an inconsistent exact prior
    # or inducing/test covariance. Never hide a substantial PSD violation.
    tolerance = 128 * torch.finfo(value.dtype).eps * torch.maximum(
        prior.abs(), projected.abs(),
    )
    if bool((value < -tolerance).any()):
        raise ValueError(name + " is negative beyond floating-point roundoff")
    return value.clamp_min(0)


def _validate_inputs(Kzz, Kzy, y, noise_variance):
    named = (("Kzz", Kzz), ("Kzy", Kzy), ("y", y),
             ("noise_variance", noise_variance))
    for name, value in named:
        if not isinstance(value, torch.Tensor):
            raise ValueError(name + " must be a tensor")
        if value.dtype not in (torch.float32, torch.float64):
            raise ValueError(name + " must use float32 or float64")
        if value.dtype != Kzz.dtype or value.device != Kzz.device:
            raise ValueError("all inputs must have the same dtype and device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(name + " must contain only finite values")
    if Kzz.ndim != 2 or Kzz.shape[0] != Kzz.shape[1]:
        raise ValueError("Kzz must be a square matrix")
    if Kzy.ndim != 2 or Kzy.shape[0] != Kzz.shape[0]:
        raise ValueError("Kzy must be a matrix with one row per inducing reference")
    if y.ndim != 1 or y.shape[0] != Kzy.shape[1]:
        raise ValueError("y must be a vector with one entry per observation")
    if noise_variance.shape != y.shape:
        raise ValueError("noise_variance must have the same vector shape as y")
    if not bool((noise_variance > 0).all()):
        raise ValueError("noise_variance must be strictly positive")
    if Kzz.numel():
        tolerance = 10 * torch.finfo(Kzz.dtype).eps
        scale = float(Kzz.detach().abs().max())
        if not torch.allclose(Kzz, Kzz.T, rtol=tolerance, atol=tolerance * scale):
            raise ValueError("Kzz must be symmetric within floating-point roundoff")


def _validate_jitter(jitter, max_jitter, jitter_multiplier):
    for name, value in (("jitter", jitter), ("max_jitter", max_jitter),
                        ("jitter_multiplier", jitter_multiplier)):
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(name + " must be a finite real constant")
    if not 0 <= jitter <= max_jitter:
        raise ValueError("require 0 <= jitter <= max_jitter")
    if jitter_multiplier <= 1:
        raise ValueError("jitter_multiplier must exceed one")


def _inducing_cholesky(Kzz, identity, jitter, max_jitter, multiplier):
    if Kzz.shape[0] == 0:
        return Kzz.clone(), 0.0
    effective = float(jitter)
    # Bound work as well as jitter: multipliers arbitrarily close to one must
    # not turn a failed factorization into an effectively unbounded loop.
    max_attempts = 16
    for attempt in range(max_attempts):
        if attempt == max_attempts - 1:
            effective = float(max_jitter)
        factor, info = torch.linalg.cholesky_ex(Kzz + effective * identity)
        if int(info) == 0 and bool(torch.isfinite(factor).all()):
            return factor, effective
        if effective >= max_jitter:
            raise RuntimeError(
                "Inducing Cholesky failed: "
                f"M={Kzz.shape[0]}, dtype={Kzz.dtype}, device={Kzz.device}, "
                f"leading_minor={int(info)}, requested_jitter={jitter:g}, "
                f"last_jitter={effective:g}, max_jitter={max_jitter:g}"
            )
        if effective == 0:
            # A documented starting scale, rather than multiplying zero forever.
            scale = max(1.0, float(Kzz.detach().diagonal().abs().max()))
            next_jitter = torch.finfo(Kzz.dtype).eps * scale
        else:
            next_jitter = effective * multiplier
        effective = (float(max_jitter) if next_jitter <= effective
                     else min(float(max_jitter), next_jitter))


def fit_sparse_gp(
    Kzz,
    Kzy,
    y,
    noise_variance,
    *,
    jitter=1e-8,
    max_jitter=1e-4,
    jitter_multiplier=10.0,
    method="qr",
):
    """Fit a zero-mean sparse GP to scalar observations using Torch only.

    Parameters are ``Kzz: (M,M)``, ``Kzy: (M,P)``, labels ``y: (P,)``, and
    positive observation **variances** ``noise_variance: (P,)``. All inputs
    must share a float32/float64 dtype and device; float64 is recommended for
    fitting. Atomic energy offsets must already have been subtracted from y.

    With ``L L.T = Kzz + effective_jitter I``, define ``G = solve(L, Kzy)``,
    ``A = G / sqrt(noise_variance)`` and ``t = y / sqrt(noise_variance)``.
    The default QR solve uses the augmented design ``[A.T; I]`` and target
    ``[t; 0]``. Fixed-input CPU fits apply Householder reflectors without
    materializing the large Q. Trainable fits retain differentiable QR.
    ``method='cholesky'`` factors ``I + A @ A.T`` instead; QR is
    preferable when the design is poorly conditioned. Both compute the NLL
    quadratic as ``||t - A.T @ mu||^2 + ||mu||^2`` to avoid cancellation from
    subtracting two large quadratic forms. No matrix inverse is constructed.

    Failed inducing Cholesky factorizations increase the absolute jitter by
    ``jitter_multiplier``, capped at ``max_jitter`` (which is also tried).
    At most 16 factorizations are attempted, using the cap on the final attempt
    if the multiplier has not reached it. A stalled increase also tries the cap.
    If zero jitter fails, the first retry uses
    ``min(max_jitter, eps * max(1, max(abs(diag(Kzz)))))``. The chosen jitter
    is reported and treated as a constant in differentiation. The adaptive
    choice is discrete: gradients are valid for that fixed chosen value.
    Asymmetric/nonfinite inputs are rejected, not silently repaired.

    Empty reference sets give a noise-only model; empty observation sets give
    an unfitted zero-mean posterior and zero NLL. With no references the
    effective jitter is zero. Cost is O(P M^2 + M^3) and storage O(P M + M^2).
    """
    _validate_inputs(Kzz, Kzy, y, noise_variance)
    _validate_jitter(jitter, max_jitter, jitter_multiplier)
    if method not in ("qr", "cholesky"):
        raise ValueError("method must be 'qr' or 'cholesky'")
    n_references, n_observations = Kzy.shape
    identity = torch.eye(n_references, dtype=Kzz.dtype, device=Kzz.device)
    L, effective_jitter = _inducing_cholesky(
        Kzz, identity, jitter, max_jitter, jitter_multiplier,
    )
    inverse_noise_std = noise_variance.rsqrt()
    t = y * inverse_noise_std
    # geqrf has no backward in the tested Torch runtime.
    fixed_cpu_qr = (
        method == "qr" and Kzz.device.type == "cpu"
        and n_references > 0 and n_observations > 0
        and (not torch.is_grad_enabled() or not any(
            value.requires_grad for value in (Kzz, Kzy, y, noise_variance)
        ))
    )
    G = torch.linalg.solve_triangular(L, Kzy, upper=False)
    A = G * inverse_noise_std[None, :]
    del G
    if not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(t).all()):
        raise RuntimeError("Whitened training data overflowed; rescale inputs or use float64")
    if fixed_cpu_qr:
        reflectors, tau = torch.geqrf(torch.cat((A.T, identity), dim=0))
        target = torch.cat((t, t.new_zeros(n_references)))[:, None]
        projected = torch.ormqr(reflectors, tau, target, left=True, transpose=True)
        R = torch.triu(reflectors[:n_references])
        rhs = projected[:n_references]
        mu = torch.linalg.solve_triangular(R, rhs, upper=True).squeeze(-1)
        del reflectors, tau, projected, target, rhs
    elif method == "qr":
        Q, R = torch.linalg.qr(torch.cat((A.T, identity), dim=0), mode="reduced")
        rhs = Q[:n_observations].T @ t
        mu = torch.linalg.solve_triangular(R, rhs[:, None], upper=True).squeeze(-1)
    else:
        lower, info = torch.linalg.cholesky_ex(identity + A @ A.T)
        if int(info) != 0 or not bool(torch.isfinite(lower).all()):
            raise RuntimeError(
                "Posterior Cholesky failed; use method='qr' for ill-conditioned "
                f"designs (M={n_references}, dtype={Kzz.dtype}, leading_minor={int(info)})"
            )
        R = lower.T
        rhs = torch.linalg.solve_triangular(lower, (A @ t)[:, None], upper=False)
        mu = torch.linalg.solve_triangular(R, rhs, upper=True).squeeze(-1)
    alpha = torch.linalg.solve_triangular(L.T, mu[:, None], upper=True).squeeze(-1)
    residual = t - A.T @ mu
    nll = 0.5 * (
        residual.square().sum() + mu.square().sum()
        + noise_variance.log().sum() + 2 * R.diagonal().abs().log().sum()
        + n_observations * math.log(2 * math.pi)
    )
    if not bool(torch.isfinite(nll)) or not bool(torch.isfinite(alpha).all()):
        raise RuntimeError("Sparse GP solve produced nonfinite values; rescale inputs or use float64")
    return SparsePosterior(L, R, mu, alpha, nll, effective_jitter, method)
