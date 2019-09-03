Git Workflow
============

To contribute to the FLARE source code, please follow the guidelines in this section.

Pushing changes from the MIR repo
---------------------------------

If you have write access to the MIR version of FLARE, you can make edits directly to the source code.

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

    $ git checkout -b <branch name>

4. When you're done working, push up the branch::

    $ git push <branch name> upstream/master

5. When you go to Github, you'll now see an option to open a Pull Request for the topic branch you just pushed. Write a helpful description of the changes you made, and then create the Pull Request.


