Compile LAMMPS with FLARE
=========================
Anders Johansson

Compilation with CMake
----------------------

1. Download LAMMPS, e.g.

.. code-block:: bash

    git clone --depth=1 https://github.com/lammps/lammps.git

2. From the directory ``flare/lammps_plugins``, run the install script.

.. code-block:: bash

    cd path/to/flare/lammps_plugins
    ./install.sh /path/to/lammps

This copies the necessary files and slightly updates the LAMMPS CMake file.

3. Compile LAMMPS with CMake as usual. If you're not using Kokkos, the Makefile system might also work.

Note: The pair style uses the Eigen (https://gitlab.com/libeigen/eigen) (header-only) library. 
This needs to be found by the compiler. Specifically, the ``Eigen`` folder. There are two workarounds.

1. Clone the ``eigen`` repository, and ``mv eigen/Eigen /path/to/lammps/src/`` if it is not easily available on your system.

2. If you are using LAMMPS version > 17Feb22, then you can set ``-DPKG_MACHDYN=yes -DDOWNLOAD_EIGEN3=yes`` for cmake, and Eigen3 will be downloaded and compiled automatically.


Compilation for GPU with Kokkos
*******************************

Run the above steps, follow LAMMPS's compilation instructions with Kokkos (https://docs.lammps.org/Build_extras.html#kokkos). Typically

.. code-block:: bash

    cd lammps
    mkdir build
    cd build
    cmake ../cmake -DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON [-DKokkos_ARCH_VOLTA70=on if not auto-detected]
    make -j<n_cpus>

If you really, really, really want to use the old Makefile system, you should be able to copy the files from ``kokkos/`` into ``/path/to/lammps/src``, do ``make yes-kokkos`` and otherwise follow the LAMMPS Kokkos instructions.

The ``KOKKOS_ARCH`` must be changed according to your GPU model. ``Volta70`` is for V100, ``Pascal60`` is for P100, etc.

Basic usage
-----------

For the model with B2 descriptors and NormalizedDotProduct kernel, the LAMMPS input script:

.. code-block:: bash

    newton on
    pair_style	flare
    pair_coeff	* * Si.txt

where ``Si.txt`` should be replaced by the name of your mapped model. Then run ``lmp -in in.script`` as usual.

For the model with 2+3 body descriptors, the LAMMPS non-Kokkos version uses ``newton off``, while the Kokkos version uses ``newton on``.

Running on a GPU with Kokkos
****************************

See the LAMMPS documentation (https://docs.lammps.org/Speed_kokkos.html). In general, run

.. code-block:: bash

    lmp -k on g 1 -sf kk -pk kokkos newton on neigh full -in in.script

When running with MPI, replace ``g 1`` with the number of GPUs *per node*.

In order to run large systems, the atoms will be divided into batches to reduce memory usage. The batch size is controlled by the ``MAXMEM`` environment variable (in GB). If necessary, set this to an estimate of how much memory FLARE++ can use (i.e., total GPU memory minus LAMMPS's memory for neighbor lists etc.). If you are memory-limited, you can set ``MAXMEM=1`` or similar, otherwise leave it to a larger number for more parallelism. The default is 12 GB, which should work for most systems while not affecting performance.

``MAXMEM`` is printed at the beginning of the simulation *from every MPI process*, in order to verify that the environment variable has been correctly set *on all nodes*. Look at ``mpirun -x`` if this is not the case.

For MPI run on CPU nodes, if you are running on multiple nodes on a cluster, 
you would typically launch one MPI task per node, 
and then set the number of threads equal to the number of cores (or hyperthreads) per node. 
A sample SLURM job script for 4 nodes, each with 48 cores, may look something like this:

.. code-block:: bash

    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=48
    mpirun -np $SLURM_NTASKS /path/to/lammps/src/lmp_kokkos_omp -k on t $SLURM_CPUS_PER_TASK -sf kk -package kokkos newton off neigh full -in in.lammps

When running on Knight's Landing or other heavily hyperthreaded systems, you may want to try using more than one thread per CPU.

For MPI run on GPU nodes, if you are running on multiple nodes on a cluster, 
you would typically launch one MPI task per GPU. 
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
