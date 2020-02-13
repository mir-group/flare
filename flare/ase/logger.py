import numpy as np
import datetime
import time
import os

from ase.md import MDLogger
from ase import units

from flare.ase.calculator import FLARE_Calculator


class OTFLogger(MDLogger):

    def __init__(self, dyn, atoms, logfile, header=False, stress=False,
            peratom=False, mode="w", data_folder='otf_data', 
            data_in_logfile=False):

        super().__init__(dyn, atoms, logfile, header, stress,
                         peratom, mode)

        self.natoms = self.atoms.get_number_of_atoms()
        self.write_header_info()
        self.start_time = time.time()
        if data_folder not in os.listdir():
            os.system('mkdir '+data_folder)

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

    def write_header_info(self):
        gp_model = self.atoms.calc.gp_model
        self.logfile.write(str(datetime.datetime.now()))
        self.logfile.write('\nnumber of cpu cores: ')  # TODO
        self.logfile.write('\ncutoffs: '+str(gp_model.cutoffs))
        self.logfile.write('\nkernel: '+gp_model.kernel.__name__)
        self.logfile.write('\nnumber of hyperparameters: '+str(len(gp_model.hyps)))
        self.logfile.write('\nhyperparameters: '+str(gp_model.hyps))
        self.logfile.write('\nhyperparameter optimization algorithm: ' +
                           gp_model.algo)
        self.logfile.write('\nuncertainty tolerance: {} times noise'.format(
                           str(self.dyn.std_tolerance_factor)))
        self.logfile.write('\ntimestep (ps): {}'.format(self.dyn.dt/1000))
        self.logfile.write('\nnumber of frames: {}'.format(0))
        self.logfile.write('\nnumber of atoms: {}'.format(
                           len(self.atoms.positions)))
        self.logfile.write('\nsystem species: {}'.format(
                           self.atoms.get_chemical_symbols()))
        self.logfile.write('\nperiodic cell:\n'+str(np.array(self.atoms.cell)))
        self.logfile.write('\n')
        self.write_prev_positions()


    def write_hyps(self, hyp_labels, hyps, like, like_grad):
        self.logfile.write('\n\nGP hyperparameters: \n')
        for i, label in enumerate(hyp_labels):
            self.logfile.write('Hyp{} : {} = {}\n'.format(i, label, hyps[i]))

        self.logfile.write('likelihood: '+str(like)+'\n')
        self.logfile.write('likelihood gradient: '+str(like_grad))

    def write_wall_time(self):
        self.logfile.write('\nwall time from start: %.2f s \n'
                           % (time.time()-self.start_time))

    def write_prev_positions(self):
        v = self.atoms.get_velocities()
        pos = self.atoms.get_positions()
        dt = self.dyn.dt
        prev_pos = pos - v * dt
        species = self.atoms.get_chemical_symbols()
        nat = len(pos)
        self.logfile.write('previous positions (A):\n')
        template = '{} {:9f} {:9f} {:9f}\n'
        for n in range(nat):
            self.logfile.write(template.format(species[n], 
                prev_pos[n,0], prev_pos[n,1], prev_pos[n,2]))

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
        self.logfile.write(50*'-')
        if self.dyn is not None:
            steps = self.dyn.nsteps
            t = steps / 1000
            if type(self.atoms.calc) != FLARE_Calculator: 
                self.logfile.write('\n*-Frame: '+str(steps))
            else:
                self.logfile.write('\n-Frame: '+str(steps))
            self.logfile.write('\nSimulation time: '+str(t)+' ps')
        self.logfile.write('\n')

        if self.data_in_logfile:
            self.write_data_to_logfile()

        # write energy, temperature info
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * self.natoms)
        if self.peratom:
            epot /= self.natoms
            ekin /= self.natoms
#        self.logfile.write('\ntotal energy: '+str(epot+ekin))
        self.logfile.write('\ntemperature: '+str(temp)+' K')
        self.logfile.write('\nkinetic energy: '+str(ekin)+' eV')
        self.write_wall_time()

        self.logfile.flush()
 
    def write_data_to_logfile(self):
        # add positions, forces and stds to be written
        species = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        forces = self.atoms.get_forces()
        velocities = self.atoms.get_velocities()
        if type(self.atoms.calc) == FLARE_Calculator: 
            stds = self.atoms.get_uncertainties(self.atoms)
            force_str = 'GP  Forces'
        else:
            stds = np.zeros(positions.shape)
            force_str = 'DFT Forces'
        self.logfile.write('Type'+(7*' ') + 'Positions'+(23*' ') +
                           force_str+(22*' ') + 'Uncertainties'+(22*' ') +
                           'Velocities\n')

        template = '{} {:9f} {:9f} {:9f}   {:9f} {:9f} {:9f}   '\
                   '{:9f} {:9f} {:9f}   {:9f} {:9f} {:9f}'
        for atom in range(len(positions)):
            dat = \
                template.format(species[atom], positions[atom][0], 
                                positions[atom][1], positions[atom][2], 
                                forces[atom][0], forces[atom][1], 
                                forces[atom][2], stds[atom][0],
                                stds[atom][1], stds[atom][2],
                                velocities[atom][0], velocities[atom][1],
                                velocities[atom][2])
            self.logfile.write(dat+'\n')

    def add_atom_info(self, target_atom, uncertainty):
        if not isinstance(target_atom, list):
            target_atom = [target_atom]
        self.logfile.write('\nAdding atom {} to the training set.\
                            \nUncertainty: {}.'.format(target_atom, uncertainty))
        self.added_atoms_dat.write(str(target_atom[0])+' ') # temporarily support 1 atom

    def write_mgp_train(self, mgp_model, train_time):
        train_size = len(mgp_model.GP.training_data)
        self.logfile.write('\ntraining set size: {}\n'.format(train_size))
        self.logfile.write('lower bound: {}\n'.format(mgp_model.l_bound))
        self.logfile.write('mgp l_bound: {}\n'.format(mgp_model.grid_params
                                                      ['bounds_2'][0, 0]))
        self.logfile.write('building mapping time: {}'.format(train_time))

    def run_complete(self):
        self.logfile.write('Run complete.')
        for f in self.traj_files:
            f.close()
        for f in self.dft_data_files:
            f.close()
