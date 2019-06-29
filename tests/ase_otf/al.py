import numpy as np
import sys
sys.path.append('../..')

from flare import kernels
from flare.gp import GaussianProcess
from flare.modules.gp_calculator import GPCalculator
from flare.ase_otf import UQ_NPT
from flare.ase_otf_logger import OTFLogger

from ase import Atoms
from ase.spacegroup import crystal
from ase.build import bulk, make_supercell, add_adsorbate
from ase.calculators.espresso import Espresso
from ase.lattice.hexagonal import Graphene
from ase.md import MDLogger
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                                         Stationary, ZeroRotation)
from ase import units
from ase.io.trajectory import Trajectory

# -------------- build atomic structure -------------------
symbol = 'Al'
a = 4.05  # Angstrom lattice spacing
unit_cell = crystal('Al', [(0,0,0)], spacegroup=225,
                     cellpar=[a, a, a, 90, 90, 90])
#unit_cell = bulk('Al', 'fcc', a=a)
multiplier = np.array([[1,0,0],[0,1,0],[0,0,1]])
super_cell = make_supercell(unit_cell, multiplier)
nat = len(super_cell.positions)
print('number of atoms:', nat)

# -------------- set up gp calculatro ---------------------
kernel = kernels.two_plus_three_body
kernel_grad = kernels.two_plus_three_body_grad
hyps = np.array([0.2, 1., 0.0001, 1, 0.005])
cutoffs = np.array([4.5, 4.5])
hyp_labels = ['sig2', 'ls2', 'sig3', 'ls3', 'noise']
opt_algorithm = 'BFGS'
gp_model = GaussianProcess(kernel, kernel_grad, hyps, cutoffs, 
        hyp_labels, opt_algorithm, par=True)

calc = GPCalculator(gp_model)
super_cell.set_calculator(calc)

# -------------- set up uq npt md --------------------
timestep = 1 # fs
temperature = 100
externalstress = 0
ttime = 25
pfactor = 3375
logfile = 'al.log'
dft_input = {'pseudopotentials': 'Al99.eam.alloy'}

# intialize velocity
MaxwellBoltzmannDistribution(super_cell, 200 * units.kB)
Stationary(super_cell)  # zero linear momentum
ZeroRotation(super_cell)  # zero angular momentum

test_uq_npt = UQ_NPT(super_cell, timestep, temperature, 
            externalstress, ttime, pfactor, mask=None, 
            logfile=logfile, loginterval=1,
            # on-the-fly parameters
            std_tolerance_factor=1, max_atoms_added=nat,
            freeze_hyps=True, dft_input=dft_input)

#trajectory = Trajectory('al.traj', 'w', super_cell)
#test_uq_npt.attach(trajectory.write, interval=1)
test_uq_npt.attach(OTFLogger(test_uq_npt, super_cell, logfile, mode="w"), interval=1)

test_uq_npt.run(5)

