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

        super().__init__(
            atoms,
            timestep,
            trajectory,
            logfile,
            loginterval,
            append_trajectory=append_trajectory,
        )

        # skip the first frame
        next(self.fake_trajectory)

    def step(self):
        # read the next frame
        new_atoms = next(self.fake_trajectory)

        # update atoms and step
        array_keys = list(self.atoms.arrays.keys())
        new_array_keys = list(new_atoms.arrays.keys())
        for key in array_keys: # first remove the original positions, numbers, etc.
            self.atoms.set_array(key, None)
        for key in new_array_keys: # then set new positions, numbers, etc.
            self.atoms.set_array(key, new_atoms.get_array(key))

        for key in self.atoms.info:
            self.atoms.info[key] = new_atoms.info.get(key, None)

        self.atoms.set_cell(new_atoms.cell)
        
        # compute GP predictions
        assert self.atoms.calc.__class__.__name__ in ["FLARE_Calculator", "SGP_Calculator"]

        self.atoms.calc.reset()
        self.atoms.get_forces()

        self.curr_step += 1
        self.atoms.info["step"] = self.curr_step


class FakeDFT(Calculator):

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, filename=None, format=None, index=":", io_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.index = index
        self.format = format
        self.io_kwargs = io_kwargs

    def calculate(
        self, atoms=None, properties=None, system_changes=None,
    ):

        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        if properties is None:
            properties = self.implemented_properties

        step = atoms.info.get("step", 0)

        fake_trajectory = read(self.filename, index=self.index, format=self.format, **self.io_kwargs)
        fake_frame = fake_trajectory[step] 
        assert np.allclose(atoms.positions, fake_frame.positions), (atoms.positions[0], fake_frame.positions[0])
        assert np.allclose(atoms.get_volume(), fake_frame.get_volume())

        self.results = deepcopy(fake_frame.calc.results)
