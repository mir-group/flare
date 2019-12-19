Installation
============

************
Requirements
************

* Python_ 
* NumPy_ 
* SciPy_ 
* Memory_profiler_
* Numba_

Optional:

* ASE_
* Pymatgen_

.. _Python: https://www.python.org/
.. _NumPy: https://docs.scipy.org/doc/numpy/reference/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _Memory_profiler: https://pypi.org/project/memory-profiler/
.. _Numba: http://numba.pydata.org/
.. _ASE: https://wiki.fysik.dtu.dk/ase/
.. _Pymatgen: https://pymatgen.org/

**********************
Installation using pip
**********************

Pip can automatically fetch the source code from PyPI_ and install.

.. code-block:: bash

  $ pip install mir-flare

For non-admin users

.. code-block:: bash

  $ pip install --upgrade --user mir-flare
    
.. _PyPI: https://pypi.org/project/mir-flare/

****************************
Manual Installation with Git
****************************

First, copy the source code from https://github.com/mir-group/flare

.. code-block:: bash

   $ git clone https://github.com/mir-group/flare.git

Then add the current path to PYTHONPATH

.. code-block:: bash

    $ cd flare; export PYTHONPATH=$(pwd):$PYTHONPATH

*****************************************
Acceleration with multiprocessing and MKL
*****************************************

If users have access to high-performance computers, we recommend 
Multiprocessing_ and MKL_ library set up to accelerate the training and prediction.
The acceleration can be significant when the GP training data is large.
This can be done in the following steps.

First, make sure the Numpy_ library is linked with MKL_ or Openblas_ and Lapack_.

.. code-block:: bash

    $ python -c "import numpy as np; print(np.__config__.show())"
    
If no libraries are linked, Numpy_ should be reinstalled. Detailed steps can be found in Conda manual_.

.. _MKL: https://software.intel.com/en-us/mkl
.. _Openblas: https://www.openblas.net/
.. _Lapack: http://www.netlib.org/lapack/
.. _manual: https://docs.anaconda.com/mkl-optimizations/
.. _Multiprocessing: https://docs.python.org/2/library/multiprocessing.html

Second, in the initialization of the GP class and OTF class, turn on the GP parallelizatition and turn off the OTF par.

.. code-block:: python

    gp_model = GaussianProcess(..., no_cpus=2)
    otf_instance = OTF(..., par=False, no_cpus=2)

Third, set the number of threads for MKL before running your python script.

.. code-block:: bash

    export OMB_NUM_THREAD=2
    python training.py

.. note::
   The "no_cpus" and OMB_NUM_THREAD should be equal or less than the number of CPUs available in the computer.
   If these numbers are larger than the actual CPUs number, it can lead to an overload of the machine.

.. note::
   If OTF.par=True and GaussianProcess.no_cpus>1, it is equivalent to run with no_cpu**2 threads
   because the MKL calls are nested in the multiprocessing code. 

The current version of FLARE can only support parallel calculations within one compute node.
Interfaces with MPI using multiple nodes are still under development.

If users encounter unusually slow FLARE training and prediction, please file us a Github Issue.

********************************
Environment variables (optional)
********************************

Flare uses a couple environmental variables in its tests for DFT and MD interfaces. These variables are not needed in the run of active learning.

.. code-block:: bash

  # the path and filename of Quantum Espresso executable
  export PWSCF_COMMAND=$(which pw.x)
  # the path and filename of CP2K executable
  export CP2K_COMMAND=$(which cp2k.popt)
  # the path and filename of LAMMPS executable
  export lmp=$(which lmp_mpi)
