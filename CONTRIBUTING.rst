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

Git Workflow
============

To contribute to the FLARE source code, please follow the guidelines in this section. If any of the git commands are unfamiliar, check out Chapters 3-5 of the `Pro Git book <https://git-scm.com/book/en/v2>`__.

General workflow
----------------

Development should follow this pattern:

1. Create an issue on Github describing what you want to contribute.
2. Create a topic branch addressing the issue. (See the sections below on how to push branches directly or from a forked repository.)
3. Merge with the development branch when finished and close the issue.

Master, development, and topic branches
---------------------------------------

The FLARE repository has a three-tiered structure: there is the master branch, which is only for battle-tested code that is both documented and unit tested; the development branch, which is used to push new features; and topic branches, which focus on specific issues and are deleted once the issue is addressed.

You can create local copies of branches from the remote repository as follows::

   $ git checkout -b <local branch name> origin/<remote branch name>


Pushing changes to the MIR repo directly
----------------------------------------

If you have write access to the MIR version of FLARE, you can make edits directly to the source code. Here are the steps you should follow:

1. Go into the development branch.
2. Create a new topic branch with a name describing what you're up to::

    $ git checkout -b <feature branch name>

3. Commit your changes periodically, and when you're done working, push the branch upstream::

    $ git push -u origin <feature branch name>

4. Create a Pull Request that gives a helpful description of what you've done. You can now merge and delete the branch.

Pushing changes from a forked repo
----------------------------------

1. Fork the `FLARE repository <https://github.com/mir-group/flare>`__.
2. Set FLARE as an upstream remote::

    $ git remote add upstream https://github.com/mir-group/flare

   Before branching off, make sure that your forked copy of the master branch is up to date::

    $ git fetch upstream
    $ git merge upstream/master

   If for some reason there were changes made on the master branch of your forked repo, you can always force a reset::

   $ git reset --hard upstream/master

3. Create a new branch with a name that describes the specific feature you want to work on::

    $ git checkout -b <feature branch name>

4. While you're working, commit your changes periodically, and when you're done, commit a final time and then push up the branch::

    $ git push -u origin <feature branch name>

5. When you go to Github, you'll now see an option to open a Pull Request for the topic branch you just pushed. Write a helpful description of the changes you made, and then create the Pull Request.
