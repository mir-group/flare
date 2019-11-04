import numpy as np
from copy import deepcopy

from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)

from flare.ase.otf_md import otf_md
from flare.ase.logger import OTFLogger

import atom_setup, flare_setup, qe_setup

np.random.seed(12345)

md_engine = 'VelocityVerlet'
print(md_engine)

# ----------- set up atoms -----------------
super_cell = atom_setup.super_cell

# ----------- setup flare calculator ---------------
flare_calc = deepcopy(flare_setup.flare_calc)
super_cell.set_calculator(flare_calc)

# ----------- setup qe calculator --------------
dft_calc = qe_setup.dft_calc

# ----------- create otf object -----------
# set up OTF MD engine
md_params = {'timestep': 1 * units.fs, 
             'trajectory': None, 
             'dt': None} 

otf_params = {'dft_calc': dft_calc, 
              'init_atoms': [0],
              'std_tolerance_factor': 0.5, 
              'max_atoms_added' : len(super_cell.positions),
              'freeze_hyps': 0, 
              'restart_from': None,
              'use_mapping': super_cell.calc.use_mapping}

# intialize velocity
temperature = 600
MaxwellBoltzmannDistribution(super_cell, temperature * units.kB)
Stationary(super_cell)  # zero linear momentum
ZeroRotation(super_cell)  # zero angular momentum

test_otf = otf_md(md_engine, super_cell, md_params, otf_params)

# set up logger
test_otf.attach(OTFLogger(test_otf, super_cell, 
    logfile='otf_run.log', mode="w", data_in_logfile=True),
    interval=1)

# rescale temperature to 1200K at step 1001
rescale_temp = [1200]
rescale_steps = [1001]

# run otf
number_of_steps = 2000
test_otf.otf_run(number_of_steps, rescale_temp, rescale_steps)
