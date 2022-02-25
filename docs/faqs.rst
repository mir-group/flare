Trouble Shooting
================

* If you have successfully installed the code, and can ``import flare_pp`` but cannot ``import flare_pp._C_flare``, we suggest that you check your Python version and try using Python 3.6. We use ``pybind11`` to compile the C++ code into a Python importable library but have run into issues using ``pybind11`` with Python 3.9.

* For developers, if your Github Action build fails from an issue related to ``gh-pages`` in the step of "Publish the docs", this may be because you did not pull the latest code before push. Please try pull and push again. If it still does not work, check `this <https://gist.github.com/mandiwise/44d1edce18f2ffb14f63>`_.
