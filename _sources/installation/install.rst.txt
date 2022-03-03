Installation of FLARE
=====================

************
Requirements
************

If you're installing on a compute cluster, make sure to load the following modules first:

.. code-block:: bash

    module load cmake/3.17.3-fasrc01 python/3.6.3-fasrc01 gcc/9.3.0-fasrc01

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

******************************
Developer's installation guide
******************************

After loading modules as above, we use ``cmake`` to compile the c++ code 

.. code-block:: bash

    git clone git@github.com:mir-group/flare.git
    cd flare
    mkdir build
    cd build
    cmake ..
    make -j

Then copy the c-library file into the python code folder to make it importable through python

.. code-block:: bash

    cp _C_flare*.so ../flare/bffs/sgp
    cd ..

Finally, add the path of ``flare`` to ``PYTHONPATH``, such that you can ``import flare`` in python. 

.. code-block:: bash

    export PYTHONPATH=${PYTHONPATH}:<current_dir>

An alternative way is setting ``sys.path.append(<flare path>)`` in your python script.


*****************************************
Acceleration with multiprocessing and MKL
*****************************************

Sparse Gaussian Process model
-----------------------------

We use OpenMP for parallelization, so please set

.. code-block:: bash

    export OMP_NUM_THREADS=<number of CPUs on a node>

such that the model parallelized into ``OMP_NUM_THREADS`` threads.

Full Gaussian Process model
---------------------------

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

    gp_model = GaussianProcess(..., parallel=True, per_atom_par=False, n_cpus=2)
    otf_instance = OTF(..., par, n_cpus=2)

Third, set the number of threads for MKL before running your python script.

.. code-block:: bash

    export OMP_NUM_THREADS=2
    python training.py

.. note::

   The "n_cpus" and OMP_NUM_THREADS should be equal or less than the number of CPUs available in the computer.
   If these numbers are larger than the actual CPUs number, it can lead to an overload of the machine.

.. note::

   If gp_model.per_atom_par=True and OMP_NUM_THREADS>1, it is equivalent to run with OMP_NUM_THREADS * otf.n_cpus threads
   because the MKL calls are nested in the multiprocessing code. 

The current version of FLARE can only support parallel calculations within one compute node.
Interfaces with MPI using multiple nodes are still under development.

If users encounter unusually slow FLARE training and prediction, please file us a Github Issue.
