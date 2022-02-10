Trouble Shooting
================

* If you have successfully installed the code, and can ``import flare_pp``, but cannot ``import flare_pp._C_flare``, we suggest you to check your python version. 
  We use ``pybind11`` to compile the c++ code into a python importable library, however, ``pybind11`` seems to have problem with python3.9. We have tried python3.6 and it works.

* For developers, if your github action build fails from an issue related to ``gh-pages`` in the step of "Publish the docs", this may be because you did not pull the latest code before push. 
  Please try pull and push again. If it still does not work, check `this <https://gist.github.com/mandiwise/44d1edce18f2ffb14f63>`_.
