import numpy as np
import datetime
import time
import os

from ase.md import MDLogger
from ase import units

from flare.ase.calculator import FLARE_Calculator
from flare.output import Output
from flare.struc import Structure

class OTFLogger(MDLogger):

    def __init__(self, dyn, atoms, logfile, header=False, stress=False,
            peratom=False, mode="w", data_folder='otf_data',
            data_in_logfile=False):

        super().__init__(dyn, atoms, logfile, header, stress,
                         peratom, mode)

        self.natoms = self.atoms.get_number_of_atoms()
        self.start_time = time.time()
        if data_folder not in os.listdir():
            os.mkdir(data_folder)

        self.positions_xyz = open(data_folder+'/positions.xyz', mode=mode)
        self.velocities_dat = open(data_folder+'/velocities.dat', mode=mode)
        self.forces_dat = open(data_folder+'/forces.dat', mode=mode)
        self.uncertainties_dat = open(data_folder+'/uncertainties.dat',
                                      mode=mode)
        self.dft_positions_xyz = open(data_folder+'/dft_positions.xyz',
                                      mode=mode)
        self.dft_forces_dat = open(data_folder+'/dft_forces.dat', mode=mode)
        self.added_atoms_dat = open(data_folder+'/added_atoms.dat', mode=mode)

        self.traj_files = [self.positions_xyz, self.velocities_dat,
                self.forces_dat, self.uncertainties_dat]
        self.dft_data_files = [self.dft_positions_xyz, self.dft_forces_dat]
        self.data_in_logfile = data_in_logfile
        if data_in_logfile: # replace original logfile in MDLogger by Output
            self.logfile = Output(logfile, always_flush=True)

    def write_header_info(self):
        gp = self.atoms.calc.gp_model
        self.structure = self.get_prev_positions()
        self.dt = self.dyn.dt/1000
        self.logfile.write_header(gp.cutoffs, gp.kernel_name,
                                  gp.hyps, gp.opt_algorithm,
                                  self.dt, 0, # Nstep set to 0
                                  self.structure, 
                                  self.dyn.std_tolerance_factor)

    def write_hyps(self):
        gp = self.atoms.calc.gp_model
        self.logfile.write_hyps(gp.hyp_labels, gp.hyps, self.start_time,
                                gp.like, gp.like_grad,
                                hyps_mask=gp.hyps_mask)


    def get_prev_positions(self):
        structure = Structure.from_ase_atoms(self.atoms)
        v = self.atoms.get_velocities()
        pos = self.atoms.get_positions()
        dt = self.dyn.dt
        prev_pos = pos - v * dt
        structure.prev_positions = prev_pos
        return structure

    def __call__(self):
        self.write_logfile()
        self.write_datafiles()

    def write_datafiles(self):
        template = '{} {:9f} {:9f} {:9f}'
        steps = self.dyn.nsteps
        t = steps / 1000

        species = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        forces = self.atoms.get_forces()
        if type(self.atoms.calc) == FLARE_Calculator:
            velocities = self.atoms.get_velocities()
            stds = self.atoms.get_uncertainties(self.atoms)
            data_files = self.traj_files
            data = [positions, velocities, forces, stds]
        else:
            data_files = self.dft_data_files
            data = [positions, forces]
            self.added_atoms_dat.write('Frame '+str(steps)+'\n')

        for ind, f in enumerate(data_files):
            f.write(str(self.natoms))
            f.write('\nFrame '+str(steps)+'\n')
            for atom in range(self.natoms):
                dat = template.format(species[atom],
                                      data[ind][atom][0],
                                      data[ind][atom][1],
                                      data[ind][atom][2])
                f.write(dat+'\n')
            f.flush()


    def write_logfile(self):
        dft_step = False

        if self.dyn is not None:
            steps = self.dyn.nsteps
            t = steps / 1000
            if type(self.atoms.calc) != FLARE_Calculator:
                dft_step = True

        # get energy, temperature info
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * self.natoms)
        local_energies = self.atoms.calc.results['local_energies']
        if self.peratom:
            epot /= self.natoms
            ekin /= self.natoms

        global_only = False if self.data_in_logfile else True
        self.logfile.write_md_config(self.dt, steps, self.structure,
                                     temp, ekin, local_energies, 
                                     self.start_time, dft_step,
                                     self.atoms.get_velocities())

    def write_mgp_train(self, mgp_model, train_time):
        train_size = len(mgp_model.GP.training_data)
        self.logfile.write('\ntraining set size: {}\n'.format(train_size))
        self.logfile.write('lower bound: {}\n'.format(mgp_model.l_bound))
        self.logfile.write('mgp l_bound: {}\n'.format(mgp_model.grid_params
                                                      ['bounds_2'][0, 0]))
        self.logfile.write('building mapping time: {}'.format(train_time))

    def run_complete(self):
        self.logfile.conclude_run()
        for f in self.traj_files:
            f.close()
        for f in self.dft_data_files:
            f.close()
        self.added_atoms_dat.close()
