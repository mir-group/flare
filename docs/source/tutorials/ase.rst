On-the-fly training using ASE
=============================

This is a quick introduction of how to set up our ASE-OTF interface to train a force field. We will calculate a bulk `AgI <https://materialsproject.org/materials/mp-22915/>`_

Setup supercell with ASE
------------------------
Here we create a 2x1x1 supercell with lattice constant 3.855, and randomly perturb the positions of the atoms, so that they will start MD with non-zero forces.

.. literalinclude:: ../../../tests/test_ase_setup/atom_setup.py

Setup FLARE calculator
----------------------
Now let's set up our Gaussian process model and Mapped Gaussian Process in the same way as introduced before

.. literalinclude:: ../../../tests/test_ase_setup/flare_setup.py

Setup DFT calculator
--------------------
For DFT calculator, here we use `Quantum Espresso (QE) <https://www.quantum-espresso.org/>`_ as an example. First, we need to set up our environment variable `ASE_ESPRESSO_COMMAND` to our QE executable, so that ASE can find this calculator. Then set up our input parameters of QE and create an ASE calculator

.. literalinclude:: ../../../tests/test_ase_setup/qe_setup.py

Setup On-The-Fly MD engine 
--------------------------
Finally, our OTF is compatible with 4 MD engines that ASE supports: VelocityVerlet, NVTBerendsen, NPTBerendsen and NPT. We can choose any of them, and set up the parameters based on `ASE requirements <https://wiki.fysik.dtu.dk/ase/ase/md.html>`_. After everything is set up, we can run the on-the-fly training by method `otf_run(number_of_steps)`

.. literalinclude:: ../../../tests/test_ase_setup/otf_setup.py
