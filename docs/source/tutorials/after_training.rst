After Training
==============

.. toctree::
   :maxdepth: 2

After the on-the-fly training is complete, we can play with the force field we obtained. 
We are going to do the following things:

1. Parse the on-the-fly training trajectory to collect training data
2. Reconstruct the GP model from the training trajectory
3. Build up Mapped GP (MGP) for accelerated force field, and save coefficient file for LAMMPS
4. Use LAMMPS to run fast simulation using MGP pair style

Parse OTF log file
------------------
After the on-the-fly training is complete, we have a log file and can use the `otf_parser` module to parse the trajectory. 

.. literalinclude:: after_training.py
   :lines: 1-6

Construct GP model from log file
--------------------------------
We can reconstruct GP model from the parsed log file (the on-the-fly training trajectory)

.. literalinclude:: after_training.py
   :lines: 11-21

The last step `write_model` is to write this GP model into a binary file, 
so next time we can directly load the model from the pickle file as

.. code-block:: python

    gp_model = pickle.load(open('AgI.gp.pickle', 'rb'))


Map the GP force field & Dump LAMMPS coefficient file
-----------------------------------------------------
To use the trained force field with accelerated version MGP, or in LAMMPS, we need to build MGP from GP model

.. literalinclude:: after_training.py
   :lines: 25-53

The coefficient file for LAMMPS mgp pair_style is automatically saved once the mapping is done. 
Saved as `lmp_file_name`. 

Run LAMMPS with MGP pair style
------------------------------
With the above coefficient file, we can run LAMMPS simulation with the mgp pair style.

1. One way to use it is running `lmp_executable < in.lammps > log.lammps` 
with the executable provided in our repository. 
When creating the input file, please note to set

.. code-block:: C

    newton off
    pair_style mgp
    pair_coeff * * <lmp_file_name> <chemical_symbols> yes/no yes/no

An example is using coefficient file `AgI_Molten_15.txt` for AgI system, 
with two-body (the 1st `yes`) together with three-body (the 2nd `yes`).

.. code-block:: C

    pair_coeff * * AgI_Molten_15.txt Ag I yes yes

2. Another way to run LAMMPS is using our LAMMPS interface, please set the
environment variable `$lmp` to the executable.

.. literalinclude:: after_training.py
    :lines: 66-106
