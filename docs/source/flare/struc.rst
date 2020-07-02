Structures
==========

The :class:`Structure` object is a collection of atoms in a periodic box.
The mandatory inputs are the cell vectors of the box and the chemical species
and Cartesian coordinates of the atoms.
The atoms are automatically folded back into the primary cell, so the
input coordinates don't need to lie inside the box.

.. py:class:: Structure(cell: np.array, species: List[str] or List[int], positions: np.array, mass_dict: dict = {}, prev_positions: ndarray = [], species_labels: List[str] = [])

    Contains information about a periodic structure of atoms, including the
    periodic cell boundaries, atomic species, and coordinates. Note that input
    positions are assumed to be Cartesian.

    :param cell: 3x3 array whose rows are the Bravais lattice vectors of the
        cell.
    :type cell: np.ndarray
    :param species: List of atomic species, which can be given as either
        atomic numbers (integers) or chemical symbols (string of one or two
        characters, e.g. 'H' for Helium). The species are stored in the coded_species attribute.
    :type species: List[int] or List[str]
    :param positions: Nx3 array of atomic coordinates in Angstrom.
    :type positions: np.array
    :param mass_dict: Dictionary of atomic masses used in MD simulations, with
        species as keywords (either as integers or strings) and masses in amu
        as values. The format of the species keyword should match the format 
        of the species input. For example, if the species are given as strings,
        mass_dict might take the form {'H': 1.0, 'He': 2.0}.
    :type mass_dict: dict
    :param prev_positions: (Optional) Nx3 array of previous atomic coordinates
        used in MD simulations. If not specified, prev_positions is set equal
        to the positions input.
    :type prev_positions: np.ndarray
    :param species_labels: (Optional) List of chemical symbols used in the
        output file of on-the-fly runs. If not specified, species_labels is
        set equal to the species input.
    :type species_labels: List[str]

    ..py:method:: positions()
        :property:
    ..py:method:: cell()
        :property:
    ..py:attribute:: coded_species

Python methods
--------------
The following are methods implemented in pure Python, i.e. independently of
the underlying C++ backend, and are intended to improve the quality of life of
the user.

.. automodule:: flare.struc
    :members:
