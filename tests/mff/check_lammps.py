import os
os.environ['LAMMPS_COMMAND'] = '/n/home08/xiey/lammps-16Mar18/src/lmp_mpi'
import numpy as np

from ase import Atoms, Atom
from ase.build import bulk, make_supercell
from ase.calculators.lammpsrun import LAMMPS
from ase.lattice.hexagonal import Graphene

symbol = 'C'
a = 2.46
c = 20.0 # vaccum
unit_cell = Graphene(symbol, latticeconstant={'a':a,'c':c})

multiplier = np.array([[10,0,0],[0,10,0],[0,0,1]])
super_cell = make_supercell(unit_cell, multiplier)
nat = len(unit_cell.positions)
cell = np.array([[ 25.51020000000000,  0.00000000000000,  0.00000000000000],
                 [-12.75509999999999, 22.09248125562179,  0.00000000000000],
                 [  0.00000000000000,  0.00000000000000, 20.00000000000000]])
super_cell.cell = cell

#a = [6.5, 6.5, 7.7]
#d = 2.3608
#NaCl = Atoms([Atom('Na', [0, 0, 0]),
#                  Atom('Cl', [0, 0, d])],
#                               cell=a, pbc=True)

pot_path = '/n/home08/xiey/lammps-16Mar18/potentials/' 
parameters = {'pair_style': 'airebo 3.0',
              'pair_coeff': ['* * '+pot_path+'CH.airebo C'],
              'mass': ['* 12.0107']}
files = [pot_path+'CH.airebo']


calc = LAMMPS(keep_tmp_files=True, tmp_dir='lmp_tmp/', parameters=parameters, files=files)
super_cell.set_calculator(calc)

print(super_cell.get_forces())
