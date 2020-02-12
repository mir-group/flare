On-the-fly training using ASE
=============================

.. toctree::
   :maxdepth: 2

This is a quick introduction of how to set up our ASE-OTF interface to train a force field. We will train a force field model for bulk `AgI <https://materialsproject.org/materials/mp-22915/>`_. To run the on-the-fly training, we will need to 

    1. Create a supercell with ASE Atoms object

    2. Set up FLARE ASE calculator, including the kernel functions, hyperparameters, cutoffs for Gaussian process, and mapping parameters (if Mapped Gaussian Process is used)

    3. Set up DFT ASE calculator. Here we will give an example of Quantum Espresso
           
    4. Set up on-the-fly training with ASE MD engine

To increase the flexibility and make it easier for testing, we suggest creating four ".py" files for the 4 steps above. And please make sure you are using the LATEST FLARE code in our master branch. 

Setup supercell with ASE
------------------------
Here we create a 2x1x1 supercell with lattice constant 3.855, and randomly perturb the positions of the atoms, so that they will start MD with non-zero forces.

.. literalinclude:: ../../../tests/ase_otf/atom_setup.py

Setup FLARE calculator
----------------------
Now let's set up our Gaussian process model in the same way as introduced before

.. literalinclude:: ../../../tests/ase_otf/flare_setup.py
   :lines: 1-27

**Optional:** if you want to use the LAMMPS interface with the trained force field, you need to construct Mapped Gaussian Process (MGP). Accelerated on-the-fly training with MGP is also enabled, but not thoroughly tested. You can set up MGP in FLARE calculator as below:

.. literalinclude:: ../../../tests/ase_otf/flare_setup.py
   :lines: 28-51

Create a ``Calculator`` object

.. literalinclude:: ../../../tests/ase_otf/flare_setup.py
   :lines: 52-53


Setup DFT calculator
--------------------
For DFT calculator, here we use `Quantum Espresso (QE) <https://www.quantum-espresso.org/>`_ as an example. 
First, we need to set up our environment variable ``ASE_ESPRESSO_COMMAND`` to our QE executable, 
so that ASE can find this calculator. Then set up our input parameters of QE and create an ASE calculator

.. literalinclude:: ../../../tests/ase_otf/qe_setup.py

Setup On-The-Fly MD engine 
--------------------------
Finally, our OTF is compatible with 4 MD engines that ASE supports: 
VelocityVerlet, NVTBerendsen, NPTBerendsen and NPT. 
We can choose any of them, and set up the parameters based on 
`ASE requirements <https://wiki.fysik.dtu.dk/ase/ase/md.html>`_. 
After everything is set up, we can run the on-the-fly training by method ``otf_run(number_of_steps)``

**Note:** Currently, only ``VelocityVerlet`` is tested on real system, ``NPT`` may have issue with pressure and stress.

.. literalinclude:: ../../../tests/ase_otf/otf_setup.py

When the OTF training is finished, there will be data files saved including:

1. A log file ``otf_run.log`` of the information in training. 
If ``data_in_logfile=True``, then the data in ``otf_data`` folder 
(described below) will also be written in this log file. 

2. A folder ``otf_data`` containing:

    * positions.xyz: the trajectory of the on-the-fly MD run
    * velocities.dat: the velocities of the frames in the trajectory
    * forces.dat: the forces of the frames in trajectory predicted by FLARE
    * uncertainties.dat: the uncertainties of the frames in trajectory predicted by FLARE 
    * dft_positions.xyz: the DFT calculated frames
    * dft_forces.dat: the DFT forces correspond to frames in dft_positions.xyz
    * added_atoms.dat: the list of atoms added to the training set of FLARE in each DFT calculated frame

3. Kernel matrix and alpha vector used in GP: ``ky_mat_inv.npy`` and ``alpha.npy``

4. If MGP is used, i.e. ``use_mapping=True``, and 3-body kernel is used and mapped, 
then there will be two files saving grid values: ``grid3_mean.npy`` and ``grid3_var.npy``.

Restart from previous training
------------------------------
We have an option for continuing from a finished training.

1. Move all the saved files mentioned above to one folder, e.g. ``restart_data``. 
(All the ``.xyz``, ``.dat``, ``.npy`` files should be in one folder)

2. Set ``otf_params['restart_from'] = 'restart_data'``

3. Run as mentioned in above sections


