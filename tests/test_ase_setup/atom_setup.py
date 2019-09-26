import numpy as np
from ase import units
from ase.spacegroup import crystal
from ase.build import bulk, make_supercell

# create primitive cell based on materials project
# url: https://materialsproject.org/materials/mp-22915/
a = 3.855
alpha = 90 
unit_cell = crystal(['Ag', 'I'], # Ag, I 
              basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
              size=(2, 2, 1),
              cellpar=[a, a, a, alpha, alpha, alpha])
#super_cell = make_supercell(unit_cell, multiplier)
super_cell = unit_cell

# jitter positions to give nonzero force on first frame
for atom_pos in super_cell.positions:
    for coord in range(3):
        atom_pos[coord] += (2*np.random.random()-1) * 0.5

