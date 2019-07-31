import numpy as np
import sys
sys.path.append('../..')

from flare import kernels
from flare.gp import GaussianProcess
from flare.mff.mff_new import MappedForceField
from flare.modules.ase_calculator import FLARE_Calculator
from flare.modules.ase_otf_md import OTF_NPT
from flare.modules.ase_otf_logger import OTFLogger

from ase import Atoms
from ase import units
from ase.spacegroup import crystal
from ase.build import bulk, make_supercell
from ase.calculators.espresso import Espresso
from ase.calculators.eam import EAM
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)

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

# -------------- set up gp calculator ---------------------
kernel = kernels.two_plus_three_body
kernel_grad = kernels.two_plus_three_body_grad
hyps = np.array([0.2, 1., 0.0001, 1, 0.005])
cutoffs = np.array([4.5, 4.5])
hyp_labels = ['sig2', 'ls2', 'sig3', 'ls3', 'noise']
opt_algorithm = 'BFGS'
gp_model = GaussianProcess(kernel, kernel_grad, hyps, cutoffs, 
        hyp_labels, opt_algorithm, par=True)

# -------------- set up mff calculator ---------------------
grid_params = {'bounds_2': np.array([[3.5], [4.5]]),
               'bounds_3': np.array([[3.5, 3.5, 0.0], [4.5, 4.5, np.pi]]), 
               'grid_num_2': 8, 
               'grid_num_3': [8, 8, 8], 
               'svd_rank_2': 0, 
               'svd_rank_3': 0,
               'bodies': '2+3', 
               'load_grid': None, 
               'load_svd': None,
               'update': False}

struc_params = {'species': 'Al', 
               'cube_lat': 100*np.eye(3), #super_cell.cell, 
               'mass_dict': {'Al': 0.000103642695727*27}}

mff_model = MappedForceField(gp_model, grid_params, struc_params)
calc = FLARE_Calculator(gp_model, mff_model=mff_model, use_mapping=True)
super_cell.set_calculator(calc)

# -------------- set up dft calculator ----------------
dft_input = {'label': 'al',
             'pseudopotentials': 'Al99.eam.alloy'}
dft_calc = EAM(potential=dft_input['pseudopotentials'])        
#pw_loc = "/n/home08/xiey/q-e/bin/pw.x"
#no_cpus = 1
#npool = 1
#pwi_file = dft_input['label'] + '.pwi'
#pwo_file = dft_input['label'] + '.pwo'
#os.environ['ASE_ESPRESSO_COMMAND'] = 'srun -n {0} --mpi=pmi2 {1} -npool {2} < {3} > {4}'.format(no_cpus, pw_loc, npool, pwi_file, pwo_file)
#input_data = dft_input['input_data']
#dft_calc = Espresso(pseudopotentials=dft_input['pseudopotentials'], label=dft_input['label'], 
#                tstress=True, tprnfor=True, nosym=True, 
#                input_data=input_data, kpts=dft_input['kpts']) 

# -------------- set up otf npt md --------------------
timestep = 1 # fs
temperature = 100
externalstress = 0
ttime = 25
pfactor = 3375
logfile = 'al.log'

# intialize velocity
MaxwellBoltzmannDistribution(super_cell, 200 * units.kB)
Stationary(super_cell)  # zero linear momentum
ZeroRotation(super_cell)  # zero angular momentum

test_otf_npt = OTF_NPT(super_cell, timestep, temperature, 
                       externalstress, ttime, pfactor, mask=None, 
                       # on-the-fly parameters
                       dft_calc=dft_calc,
                       std_tolerance_factor=1, max_atoms_added=nat,
                       freeze_hyps=0, 
                       # mff parameters
                       use_mapping=super_cell.calc.use_mapping)

test_otf_npt.attach(OTFLogger(test_otf_npt, super_cell, logfile, mode="w"), interval=1)

test_otf_npt.otf_run(5)

