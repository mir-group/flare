Prepare your data
=================

If you have collected data for training, including atomic positions, chemical 
species, cell etc., you need to convert it into a list of ``Structure`` objects. 
Below we provide a few examples.


VASP data
---------

If you have AIMD data from VASP, you can follow 
`the step 2 of this instruction <https://flare.readthedocs.io/en/latest/tutorials/gpfa.html>`_
to generate ``Structure``s with the ``vasprun.xml`` file. 


Data from Quantum Espresso, LAMMPS, etc.
----------------------------------------

If you have collected data from any 
`calculator that ASE supports <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html>`_,
or have dumped data file of `format that ASE supports <https://wiki.fysik.dtu.dk/ase/ase/io/io.html>`_,
you can convert your data into ASE ``Atoms``, then from ``Atoms`` to 
``Structure`` via ``Structure.from_ase_atoms``.

For example, if you have collected data from QE, and obtained the QE output file ``.pwo``, 
you can parse it with ASE, and convert ASE ``Atoms`` into ``Structure``.


.. code-block:: python
 
    from ase.io import read
    from flare.struc import Structure

    frames = read('data.pwo', index=':', format='espresso-out') # read the whole traj
    trajectory = []
    for atoms in frames:
        trajectory.append(Structure.from_ase_atoms(atoms))


If the data is from the LAMMPS dump file, use
.. code-block:: python
    
    # if it's text file
    frames = read('data.dump', index=':', format='lammps-dump-text')

    # if it's binary file
    frames = read('data.dump', index=':', format='lammps-dump-binary')


Then the ``trajectory`` can be used to
`train GP from AIMD data <https://flare.readthedocs.io/en/latest/tutorials/gpfa.html>`_.


Try building GP from data
-------------------------

To have a more complete and better monitored training process, please use our 
`GPFA module <https://flare.readthedocs.io/en/latest/tutorials/gpfa.html>`_. 

Here we are not going to use this module, but only provide a simple example on 
how the GP is constructed from the data.

.. code-block:: python

    from flare.gp import GaussianProcess
    from flare.parameters import Parameters
