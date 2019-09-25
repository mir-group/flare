import numpy as np
from ase import units
from ase.spacegroup import crystal
from ase.build import bulk, make_supercell

# create primitive cell based on materials project
# url: https://materialsproject.org/materials/mp-22925/#
a = 4.696 * 1.1
alpha = 60
unit_cell = crystal(['Ag', 'I'], # Ag, I 
              [(0, 0, 0), (0.25, 0.25, 0.25)],
              cellpar=[a, a, a, alpha, alpha, alpha])
multiplier = np.array([[2,0,0], [0,2,0], [0,0,1]])
super_cell = make_supercell(unit_cell, multiplier)

# jitter positions to give nonzero force on first frame
for atom_pos in super_cell.positions:
    for coord in range(3):
        atom_pos[coord] += (2*np.random.random()-1) * 0.5
