Compile LAMMPS with MGP Pair Style
==================================
Anders Johansson, Yu Xie


Download
--------

If you want to use MGP force field in LAMMPS, the first step is to download our MGP pair_style source code from:
https://github.com/mir-group/flare/tree/master/lammps_plugins

Then, follow the instruction below to compile the LAMMPS executable.

MPI
---

For when you can't get Kokkos+OpenMP to work.

Compiling
*********

.. code-block:: bash

    cp -r lammps_plugin /path/to/lammps/src/USER-MGP
    cd /path/to/lammps/src
    make yes-user-mgp
    make -j$(nproc) mpi


You can replace ``mpi`` with your favourite Makefile, e.g. ``intel_cpu_intelmpi``, or use the CMake build system.

Running
*******

.. code-block:: bash

    mpirun /path/to/lammps/src/lmp_mpi -in in.lammps

as usual, but your LAMMPS script ``in.lammps`` needs to specify ``newton off``.


MPI+OpenMP through Kokkos
-------------------------

For OpenMP parallelisation on your laptop or on one node, or for hybrid parallelisation on multiple nodes.

Compiling
*********

.. code-block:: bash

    cp -r lammps_plugin /path/to/lammps/src/USER-MGP
    cd /path/to/lammps/src
    make yes-user-mgp
    make yes-kokkos
    make -j$(nproc) kokkos_omp

You can change the compiler flags etc. in ``/path/to/lammps/src/MAKE/OPTIONS/Makefile.kokkos_omp``. 
As of writing, the pair style is not detected by CMake.

Running
*******

With ``newton on`` in your LAMMPS script:

.. code-block:: bash

    mpirun /path/to/lammps/src/lmp_kokkos_omp -k on t 4 -sf kk -package kokkos newton on neigh half -in in.lammps

With ``newton off`` in your LAMMPS script:

.. code-block:: bash

    mpirun /path/to/lammps/src/lmp_kokkos_omp -k on t 4 -sf kk -package kokkos newton off neigh full -in in.lammps

Replace 4 with the desired number of threads *per MPI task*. Skip ``mpirun`` when running on one machine.

If you are running on multiple nodes on a cluster, you would typically launch one MPI task per node, 
and then set the number of threads equal to the number of cores (or hyperthreads) per node. 
A sample SLURM job script for 4 nodes, each with 48 cores, may look something like this:

.. code-block:: bash

    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=48
    mpirun -np $SLURM_NTASKS /path/to/lammps/src/lmp_kokkos_omp -k on t $SLURM_CPUS_PER_TASK -sf kk -package kokkos newton off neigh full -in in.lammps

When running on Knight's Landing or other heavily hyperthreaded systems, you may want to try using more than one thread per CPU.


MPI+CUDA through Kokkos
-----------------------

For running on the GPU on your laptop, or for multiple GPUs on one or more nodes.

Compiling
*********

.. code-block:: bash

    cp -r lammps_plugin /path/to/lammps/src/USER-MGP
    cd /path/to/lammps/src
    make yes-user-mgp
    make yes-kokkos
    make -j$(nproc) KOKKOS_ARCH=Volta70 kokkos_cuda_mpi

The ``KOKKOS_ARCH`` must be changed according to your GPU model. ``Volta70`` is for V100, ``Pascal60`` is for P100, etc.

You can change the compiler flags etc. in ``/path/to/lammps/src/MAKE/OPTIONS/Makefile.kokkos_cuda_mpi``. 
As of writing, the pair style is not detected by CMake.

Running
*******

With ``newton on`` in your LAMMPS script:

.. code-block:: bash

    mpirun /path/to/lammps/src/lmp_kokkos_cuda_mpi -k on g 4 -sf kk -package kokkos newton on neigh half -in in.lammps

With ``newton off`` in your LAMMPS script:

.. code-block:: bash

    mpirun /path/to/lammps/src/lmp_kokkos_cuda_mpi -k on g 4 -sf kk -package kokkos newton off neigh full -in in.lammps

Replace 4 with the desired number of GPUs *per node*, skip ``mpirun`` if you are using 1 GPU. 
The number of MPI tasks should be set equal to the total number of GPUs.

If you are running on multiple nodes on a cluster, you would typically launch one MPI task per GPU. 
A sample SLURM job script for 4 nodes, each with 2 GPUs, may look something like this:

.. code-block:: bash

    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=2
    #SBATCH --cpus-per-task=1
    #SBATCH --gpus-per-node=2
    mpirun -np $SLURM_NTASKS /path/to/lammps/src/lmp_kokkos_cuda_mpi -k on g $SLURM_GPUS_PER_NODE -sf kk -package kokkos newton off neigh full -in in.lammps


Notes on Newton (only relevant with Kokkos)
-------------------------------------------

There are defaults which will kick in if you don't specify anything in the input 
script and/or skip the ``-package kokkos newton ... neigh ...`` flag. 
You can try these at your own risk, but it is safest to specify everything. 
See also the `documentation <https://lammps.sandia.gov/doc/Speed_kokkos.html>`_.

``newton on`` will probably be faster if you have a 2-body potential, 
otherwise the alternatives should give roughly equal performance.
