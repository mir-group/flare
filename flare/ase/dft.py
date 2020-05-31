'''
This module is to provide the same interface as the module `dft_interface`, so we can use ASE atoms and calculators to run OTF
'''

import numpy as np
from copy import deepcopy

def parse_dft_input(atoms):
    pos = atoms.positions
    spc = atoms.get_chemical_symbols()
    cell = np.array(atoms.get_cell())
    mass = atoms.get_masses()
    return pos, spc, cell, mass

def run_dft_par(atoms, structure, dft_calc, **dft_kwargs):
    '''
    Assume that the atoms have been updated
    '''
    # change from FLARE to DFT calculator
    calc = deepcopy(dft_calc)
    atoms.set_calculator(calc)

    # calculate DFT forces 
    forces = atoms.get_forces()

    return forces
