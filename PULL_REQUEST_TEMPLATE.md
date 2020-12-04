Adapted from the Pymatgen PR template.

## Summary

Include a summary of major changes in bullet points:

* Feature 1
* Feature 2
* Fix 1
* Fix 2

## Additional dependencies introduced (if any)

* List all new dependencies needed and justify why. While adding dependencies that bring
significantly useful functionality is perfectly fine, adding ones that 
add trivial functionality, e.g., to use one single easily implementable
function, is frowned upon. Provide a justification why that dependency is needed.

## TODO (if any)

If this is a work-in-progress, write something about what else needs 
to be done

* Feature 1 supports A, but not B.

## Checklist

Work-in-progress pull requests are encouraged, but please put [WIP]
in the pull request title.

Before a pull request can be merged, the following items must be checked:

- [ ] Code is in the [standard Python style](https://www.python.org/dev/peps/pep-0008/). 
      Run [Black](https://pypi.org/project/black/) on your local machine.
- [ ] Docstrings have been added in the [Sphinx docstring format](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html).
- [ ] Type annotations are **highly** encouraged.
- [ ] Tests have been added for any new functionality or bug fixes.
- [ ] All existing tests pass.

Note that the CI system will run all the above checks. But it will be much more
efficient if you already fix most errors prior to submitting the PR.
