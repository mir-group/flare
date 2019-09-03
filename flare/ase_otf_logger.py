import numpy as np
import datetime
import time

from ase.md import MDLogger
from ase import units

from flare.modules.gp_calculator import GPCalculator

class OTFLogger(MDLogger):

    def __init__(self, dyn, atoms, logfile, header=False, stress=False,
            peratom=False, mode="w"):

        super().__init__(dyn, atoms, logfile, header, stress,
                peratom, mode)

        self.write_header_info()
        self.start_time = time.time()


    def write_header_info(self):
        gp_model = self.atoms.calc.gp_model
        self.logfile.write(str(datetime.datetime.now()))
        self.logfile.write('\nnumber of cpu cores: ') # TODO
        self.logfile.write('\ncutoffs: '+str(gp_model.cutoffs))
        self.logfile.write('\nkernel: '+gp_model.kernel.__name__)
        self.logfile.write('\nnumber of parameters: '+str(len(gp_model.hyps)))
        self.logfile.write('\nhyperparameters: '+str(gp_model.hyps))
        self.logfile.write('\nhyperparameter optimization algorithm: '+gp_model.algo)
        self.logfile.write('\nuncertainty tolerance: '+str(self.dyn.std_tolerance))
        self.logfile.write('\nperiodic cell:\n'+str(self.atoms.cell))
        self.logfile.write('\n')


    def __call__(self):
        self.logfile.write('\n'+50*'-')
        if self.dyn is not None:
            steps = self.dyn.get_time()
            t = steps / (1000*units.fs)
            if type(self.atoms.calc) != GPCalculator: # TODO
                self.logfile.write('\n-*Frame: '+str(steps))
            else:
                self.logfile.write('\n-Frame: '+str(steps))
            self.logfile.write('\nSimulation time: '+str(t))
        self.logfile.write('\n')

        # add positions, forces and stds to be written
        positions = self.atoms.get_positions()
        forces = self.atoms.get_forces()
        velocities = self.atoms.get_velocities()
        if type(self.atoms.calc) == GPCalculator: # TODO: make it compatible with mff
            stds = self.atoms.get_uncertainties()
            force_str = 'GP Forces'
        else:
            stds = np.zeros(positions.shape)
            force_str = 'DFT Forces'
        self.logfile.write((10*' ')+'Positions'+(11*' '+'|'+11*' ')+
                                    force_str+(11*' '+'|'+11*' ')+
                                    'Velocities'+(10*' '+'|'+11*' ')+
                                    'Uncertainties\n')

        template = '{:9f} {:9f} {:9f} | {:9f} {:9f} {:9f} | {:9f} {:9f} {:9f} | {:9f} {:9f} {:9f}'
        for atom in range(len(positions)):
            dat = template.format(\
                    positions[atom][0], positions[atom][1], positions[atom][2],
                    forces[atom][0],    forces[atom][1],    forces[atom][2],
                    velocities[atom][0],velocities[atom][1],velocities[atom][2],
                    stds[atom][0],      stds[atom][1],      stds[atom][2])
            self.logfile.write(dat+'\n')

        # write energy, temperature info
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * self.natoms)
        if self.peratom:
            epot /= self.natoms
            ekin /= self.natoms
        self.logfile.write('\ntotal energy: '+str(epot+ekin))
        self.logfile.write('\nkinetic energy: '+str(ekin))
        self.logfile.write('\ntemperature: '+str(temp))
        self.logfile.write('\nwall time from start: '+str(time.time()-self.start_time))

        self.logfile.write('\n')
        self.logfile.flush()
        

    def add_atom_info(self, target_atom, uncertainty):
        self.logfile.write('\nAdding atom '+str(target_atom)+' to the training set with uncertainty: '+str(uncertainty))
