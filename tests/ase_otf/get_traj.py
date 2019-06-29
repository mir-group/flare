import numpy as np
from ase.io.trajectory import Trajectory

traj = Trajectory('al.traj')
for atom in traj:
    print(np.max(atom.get_forces()))
