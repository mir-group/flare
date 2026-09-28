Torch sparse GP numerical conventions
=====================================

The Torch backend has a fixed global species vocabulary. Positions are
``(N, 3)`` and cell vectors are rows of ``(3, 3)``. Neighbor lists include
periodic images strictly inside the cutoff, exclude the zero-shift self-edge,
and retain nonzero self-images. The generic tensor geometry supports partial
periodicity; the ASE OTF workflow uses a full-rank periodic cell. Tensor
functions preserve input dtype, device, and differentiation history. The
native-reference fixtures and ASE adapter use float64 CPU values.

The B2 descriptor returns raw power-spectrum features. With
``C = n_species * n_radial``, its feature order is the upper triangle of channel
pairs ``a <= b``, with angular order ``l`` varying fastest. Off-diagonal pairs
have no extra ``sqrt(2)`` factor. The normalized dot product kernel normalizes
these raw vectors and is zero for different central species. A raw descriptor
norm below ``1e-8`` counts as empty and yields zero covariance. This threshold
is part of native FLARE compatibility, including its nonsmooth boundary.

Training observations are ordered per structure as total energy, atom-major
``xyz`` forces, and stress ``xx, xy, xz, yy, yz, zz``. Energy sums local
energies; configured single-atom offsets are subtracted from energy labels.
Force columns are negative position derivatives. Native stress is
``-dE/dstrain / volume`` under the deformation
``R' = R @ (I + strain).T`` and ``cell' = cell @ (I + strain).T``. The ASE
stress order and sign are converted at the calculator boundary. Energy, force,
and stress noise settings are standard deviations; each observation's noise
variance is the square of its configured standard deviation times its relative
noise multiplier.

The sparse fit uses the low-rank Gaussian likelihood without a variational
trace correction. ``SOR`` predictive variance belongs to that low-rank model;
``DTC`` adds the prior residual. ``local`` uncertainty uses the prior residual
alone for environment selection. The numerical tests compare descriptors,
derivatives, kernels, likelihoods, predictions, and uncertainties against
committed native reference fixtures and independent finite differences.

Run ``python -m pytest -q tests/tensor`` from a source checkout to validate the
backend without building the native extension. The fixtures in
``tests/tensor/data`` can be regenerated with ``dev/tensor/build_reference.py``
and ``dev/tensor/generate_reference.py``. Both scripts pin and verify the native
source revision; the generator's ``--check`` mode compares regenerated arrays
with the committed fixtures without rewriting them.
