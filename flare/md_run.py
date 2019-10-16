from flare.gp import GaussianProcess
from flare.struc import Structure
from flare.env import AtomicEnvironment
import numpy as np
import time
import datetime

from flare import md
from flare.output import Output
import flare.predict as predict

class MD:
    """Generates NVE dynamics from a GP model."""

    def __init__(self, dt: float, number_of_steps: int, gp: GaussianProcess,
                 pos_init: np.ndarray, species, cell, masses,
                 prev_pos_init: np.ndarray=None, par: bool=False, skip: int=0,
                 output_name='otf_run.out'):

        self.dt = dt
        self.Nsteps = number_of_steps
        self.gp = gp

        self.structure = Structure(cell=cell, species=species,
                                   positions=pos_init,
                                   mass_dict=masses,
                                   prev_positions=prev_pos_init)

        self.noa = self.structure.positions.shape[0]
        self.atom_list = list(range(self.noa))
        self.curr_step = 0

        # choose prediction function
        if par is True:
            self.pred_func = predict.predict_on_structure_par_en
        else:
            self.pred_func = predict.predict_on_structure_en

        # initialize local energies
        self.local_energies = np.zeros(self.noa)

        self.pes = []
        self.kes = []

        self.output = Output(output_name)

    def run(self):
        self.output.write_header(self.gp.cutoffs, self.gp.kernel_name, self.gp.hyps,
                                 self.gp.algo, self.dt, self.Nsteps, self.structure)
        self.start_time = time.time()

        while self.curr_step < self.Nsteps:
            # verlet algorithm follows Frenkel p. 70
            self.gp.check_L_alpha()
            self.pred_func()
            new_pos = md.update_positions(self.dt, self.noa, self.structure)
            self.update_temperature(new_pos)
            self.record_state()
            self.update_positions(new_pos)
            self.curr_step += 1

        self.output.conclude_run()

    def update_positions(self, new_pos):
        self.structure.prev_positions = self.structure.positions
        self.structure.positions = new_pos
        self.structure.wrap_positions()

    def update_temperature(self, new_pos):
        KE, temperature = \
                md.calculate_temperature(new_pos, self.structure, self.dt,
                                         self.noa)
        self.KE = KE
        self.temperature = temperature

    def record_state(self):
        self.pes.append(np.sum(self.local_energies))
        self.kes.append(self.KE)
        self.output.write_md_config(self.dt, self.curr_step, self.structure,
                                    self.temperature, self.KE, self.local_energies,
                                    self.start_time)
        self.output.write_xyz_config(self.curr_step, self.structure,
                                     self.dft_step)
