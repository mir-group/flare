FLARE: Active Learning Bayesian Force Fields
============================================

We have a few tutorial notebooks that you can check out and play with.

[FLARE (ACE descriptors + sparse GP) with LAMMPS](https://colab.research.google.com/github/mir-group/FLARE-Tutorials/blob/master/FLARE_Tutorial_2026.ipynb) 
This tutorial demonstrates how to construct Bayesian force field based on sparse Gaussian Process (SGP) model for aluminum. We cover both "online" (on-the-fly) active learning and "offline" training methodologies using ASE MD engine. Additionally, the later section guides users through compiling LAMMPS with FLARE pair styles to enable scalable Bayesian active learning and uncertainty-aware molecular dynamics.

`FLARE (LAMMPS active learning) <https://bit.ly/flarelmpotf>`_.
This tutorial demonstrates new functionality for running active learning all within LAMMPS, with LAMMPS running the dynamics to allow arbitrarily complex molecular dynamics workflows while maintaining a simple interface. This also demonstrates how to use the C++ API directly from Python through `pybind11`. Finally, there's a simple demonstration of phonon calculations with FLARE using `phonopy`.

.. `FLARE (ACE descriptors + sparse GP) Python API <https://colab.research.google.com/drive/18_pTcWM19AUiksaRyCgg9BCpVyw744xv>`_.
.. The tutorial shows how to do the offline and online trainings with python scripts.
.. A video walkthrough of the tutorial, including detailed discussion of expected outputs, is available `here <https://youtu.be/-FH_VqRQrso>`_.

.. `FLARE (2+3-body + GP) <https://colab.research.google.com/drive/1Q2NCCQWYQdTW9-e35v1W-mBlWTiQ4zfT>`_.
.. The tutorial shows how to use flare 2+3 body descriptors and squared exponential kernel to train a Gaussian Process force field on-the-fly.

`Compute thermal conductivity from FLARE and Boltzmann transport equations <https://phoebe.readthedocs.io/en/develop/tutorials/mlPhononTransport.html>`_.
The tutorial shows how to use FLARE (LAMMPS) potential to compute lattice thermal conductivity from Boltzmann transport equation method, 
with `Phono3py <https://phonopy.github.io/phono3py/>`_ for force constants calculations  and `Phoebe <https://mir-group.github.io/phoebe/>`_ for thermal conductivities.

`Using your own customized descriptors with FLARE <https://colab.research.google.com/drive/1VzbIPmx1z-uygKstOYTj2Nqr53AMC5NL?usp=sharing>`_. 
The tutorial shows how to attach your own descriptors with FLARE sparse GP model and do training and testing.

All the tutorials take a few minutes to run on a normal desktop computer or laptop (excluding installation time).
