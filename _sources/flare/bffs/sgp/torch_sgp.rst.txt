Torch sparse Gaussian process
=============================

The experimental Torch backend provides sparse GP fitting and prediction through
FLARE's ASE on-the-fly (OTF) workflow. It is opt-in and uses float64 tensors on
the CPU. From a source checkout, install its optional dependencies with
``pip install -e '.[torch]'``.

To select it, set ``flare_calc.gp`` to ``TorchSGP`` in a normal ``flare-otf``
input. For example, the ``flare_calc`` section for argon is:

.. code-block:: yaml

   flare_calc:
     gp: TorchSGP
     species: [18]
     cutoff: 3.0
     kernels:
       - {name: NormalizedDotProduct, sigma: 1.3, power: 2}
     descriptors:
       - {name: B2, nmax: 2, lmax: 1, radial_basis: chebyshev,
          cutoff_function: quadratic}
     energy_noise: 0.2
     forces_noise: 0.15
     stress_noise: 0.04
     variance_type: SOR
     use_mapping: false

Supply the usual ``supercell``, ``dft_calc``, and ``otf`` sections alongside it,
then run ``flare-otf input.yaml``. The ``species`` list contains the atomic
numbers in a fixed order and must include every element the run may encounter.
The example uses one normalized dot product kernel and one B2 descriptor with a
Chebyshev radial basis and quadratic cutoff. Those are the supported kernel and
descriptor choices. ``variance_type`` may be ``SOR``, ``DTC``, or ``local``.

The ASE OTF workflow accepts energy, force, and stress labels, selects inducing
environments, and can optimize the kernel amplitude and observation noise
values. Stress requires a finite, full-rank periodic cell. Model files and OTF
checkpoints use the usual ``write_model`` and restart settings. A saved model
can also be loaded with ``flare_calc: {gp: TorchSGP, file: model.json}``.

This backend does not provide mapped potentials or PyLAMMPS integration. The
tensor implementation's geometry, descriptor, and observation conventions are
documented in :doc:`torch_tensor_contract`.
