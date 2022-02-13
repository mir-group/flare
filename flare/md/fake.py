import numpy as np
from copy import deepcopy
from ase.md.md import MolecularDynamics
from ase.io import iread, read
from ase.calculators.calculator import Calculator, all_changes

class FakeMD(MolecularDynamics):

    def __init__(
        self, 
        atoms, 
        timestep, 
        trajectory=None, 
        filename=None,
        format=None,
        index=":",
        io_kwargs={},
        logfile=None,
        loginterval=1,
        append_trajectory=False,
    ):
        self.fake_trajectory = iread(filename, index=index, format=format, **io_kwargs)
        self.curr_step = 0 # TODO: This might be wrong if `index` does not start from 0

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
        # read the next frame
        new_atoms = next(self.fake_trajectory)

        # compute GP predictions
        assert self.atoms.calc.__class__.__name__ in ["FLARE_Calculator", "SGP_Calculator"]
        new_atoms.calc = self.atoms.calc
        new_atoms.calc.reset()
        new_atoms.get_forces()

        # update atoms and step
        self.atoms = new_atoms
        self.curr_step += 1
        self.atoms.info["step"] = self.curr_step


def FakeDFT(Calculator):

    def __init__(self, filename=None, format=None, index=":", io_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        self.fake_trajectory = read(filename, index=index, format=format, **io_kwargs)
        self.fake_frame = None

    def calculate(
        self, atoms=None, properties=None, system_changes=None,
    ):

        step = atoms.info.get("step", None)
        assert step is not None

        self.fake_frame = self.fake_trajectory[step] 
        assert np.allclose(atoms.positions, self.fake_frame.positions)
        assert np.allclose(atoms.get_volume(), self.fake_frame.get_volume())

        self.results = deepcopy(self.fake_frame.calc.results)
