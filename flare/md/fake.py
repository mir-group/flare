from ase.md.md import MolecularDynamics
from ase.io import iread

class FakeMD(MolecularDynamics):

    def __init__(
        self, 
        atoms, 
        timestep, 
        trajectory=None, 
        filename=None,
        format=None,
        index=":",
        logfile=None,
        loginterval=1,
        append_trajectory=False,
    ):
        self.fake_trajectory = iread(filename, index, format)

        MolecularDynamics.__init__(
            self,
            atoms,
            timestep,
            trajectory,
            logfile,
            loginterval,
            append_trajectory=append_trajectory,
        )

    def step(self):
        new_atoms = next(self.fake_trajectory)
        new_atoms.calc = self.atoms.calc
        new_atoms.calc.reset()
        new_atoms.get_forces()

        self.atoms = new_atoms
