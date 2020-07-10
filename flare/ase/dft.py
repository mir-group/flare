'''
This module is to provide the same interface as the module `dft_interface`, so we can use ASE atoms and calculators to run OTF
'''

import numpy as np
from copy import deepcopy

def parse_dft_input(atoms):
    pos = np.copy(atoms.positions)
    spc = atoms.get_chemical_symbols()
    cell = np.array(atoms.get_cell())

    # build mass dict
    mass = atoms.get_masses()
    mass_dict = {}
    for i in range(len(spc)):
        spec_ind = str(spc[i])
        if spec_ind not in mass_dict.keys():
            mass_dict[spec_ind] = mass[i]
    return pos, spc, cell, mass_dict

def run_dft_par(atoms, structure, dft_calc, **dft_kwargs):
    '''
    Assume that the atoms have been updated
    '''
    # change from FLARE to DFT calculator
    calc = deepcopy(dft_calc)
    atoms.set_calculator(calc)

    # calculate DFT forces
    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()

    return forces, energy
