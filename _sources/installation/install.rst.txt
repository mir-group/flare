Installation of FLARE
=====================

************
Requirements
************

1. Create a new conda environment to install flare

.. code-block:: bash

    conda create --name flare python=3.8
    conda activate flare

2. Use conda to install compilers and dependencies for flare

* Option 1: You can load modules if your machine have already installed them (with mkl)

.. code-block:: bash

    module load cmake/3.17.3 gcc/9.3.0 intel-mkl/2017.2.174

* Option 2: If you want to install flare with mkl
    
.. code-block:: bash

    conda install -y gcc gxx cmake mkl-devel mkl-service mkl_fft openmp -c conda-forge
    
* Option 3: If you want to install flare with openblas + lapacke
    
.. code-block:: bash

    conda install -y gcc gxx cmake openmp liblapacke openblas -c conda-forge

3. Download flare code from github repo and pip install 

.. code-block:: bash

    git clone -b development https://github.com/mir-group/flare.git
    cd flare
    pip install .


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


***********************
Test FLARE installation
***********************

After the installation is done, you can leave the current folder and in the python console, try

.. code-block:: ipython

    >>> import flare
    >>> flare.__file__
    '/xxx/.conda/envs/flare/lib/python3.8/site-packages/flare/__init__.py'
    >>> import flare.bffs.sgp

You can also check that the flare C++ library is linked to mkl (or openblas and lapacke) and openmp by

.. code-block:: bash

    ldd /xxx/.conda/envs/flare/lib/python3.8/site-packages/flare/bffs/sgp/_C_flare.*.so

where it is expected to show libmkl (or libopenblas), libgomp etc.

****************
Trouble shooting
****************

* If it fails to build mir-flare during pip install, check whether you have done `conda install cxx-compiler`, then it sometimes does not find the correct g++, i.e. `which gcc` gives the conda’s gcc, while `which g++` still gives the local one. In such case, try `conda uninstall cxx-compiler gxx`, and then redo `conda install gxx -c conda-forge`

* If you get the error that `mkl.h` is not found during pip install, first check `conda list` that `mkl-include` is installed in the current environment. You can also use your own mkl headers by setting environment variable `MKL_INCLUDE` to the directory

* If you manage to install flare, but get warning when `import flare.bffs.sgp`, then please check

    * `which pip` should show that `pip` is in the `.conda/envs/flare/bin/` directory, instead of others

    * the `ldd` command above should show the linked libraries in the `.conda/envs/flare` directory

* If you encounter `Intel MKL FATAL ERROR` when running flare (after the compilation has done), this is likely a static library linkage issue. You can set up the environmental variable

.. code-block:: bash

    export LD_PRELOAD=${CONDA_PREFIX}/lib/libmkl_core.so:${CONDA_PREFIX}/lib/libmkl_intel_thread.so:${CONDA_PREFIX}/lib/libiomp5.so

as instructed in `this discussion <https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/mkl-fails-to-load/m-p/1155538>`_.

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
