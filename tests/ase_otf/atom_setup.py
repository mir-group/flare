import numpy as np
from ase import units
from ase.spacegroup import crystal
from ase.build import bulk

np.random.seed(12345)

a = 3.52678
super_cell = bulk('C', 'diamond', a=a, cubic=True) 
             
## jitter positions to give nonzero force on first frame
#for atom_pos in super_cell.positions:
#    for coord in range(3):
#        atom_pos[coord] += (2*np.random.random()-1) * 0.02
#
