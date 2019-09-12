Code Standards
==============

Before pushing code to the development branch, please make sure your changes respect the following code standards.

PEP 8
-----
Run your code through pylint to check that you're in compliance with `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`__::

    $ pylint <file name>

Docstrings
----------
All new modules, classes, and methods should have `Sphinx style docstrings <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`__ describing what the code does and what its inputs are. These docstrings are used to automatically generate FLARE's documentation, so please make sure they're clear and descriptive.

Tests
-----
New features must be accompanied by unit and integration tests written using `pytest <https://docs.pytest.org/en/latest/>`__. This helps ensure the code works properly and makes the code as a whole easier to maintain.