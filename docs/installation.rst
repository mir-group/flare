Installation
============

Pip installation
----------------

The easiest way to install flare++ is with pip. Just run the following command:

.. code-block:: bash
    pip install flare_pp

This will take a few minutes on a normal desktop computer or laptop.

If you're installing on Harvard's compute cluster, make sure to load the following modules first:
.. code-block:: bash
    module load cmake/3.17.3-fasrc01 python/3.6.3-fasrc01 gcc/9.3.0-fasrc01


Developer's installation guide
------------------------------

After loading modules as above, we use ``cmake`` to compile the c++ code 

.. code-block:: bash
    git clone git@github.com:mir-group/flare_pp.git
    cd flare_pp
    mkdir build
    cd build
    cmake ..
    make -j

Then copy the c-library file into the python code folder to make it importable through python

.. code-block:: bash
    cp _C_flare*.so ../flare_pp
    cd ..

Finally, add the path of ``flare_pp`` to ``PYTHONPATH``, such that you can ``import flare_pp`` in python. 

.. code-block:: bash
    export PYTHONPATH=${PYTHONPATH}:<current_dir>

An alternative way is setting ``sys.path.append(<flare_pp path>)`` in your python script.


LAMMPS Plugin
-------------

See `lammps_plugins/README.md <https://github.com/mir-group/flare_pp/blob/master/lammps_plugins/README.md>`_.
