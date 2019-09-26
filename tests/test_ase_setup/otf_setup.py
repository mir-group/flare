import sys
import numpy as np
from copy import deepcopy

from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)

from flare.ase.otf_md import otf_md
from flare.ase.logger import OTFLogger

import atom_setup, flare_setup, qe_setup

np.random.seed(12345)

md_engines = ['VelocityVerlet', 'NVTBerendsen', 'NPTBerendsen', 'NPT']
for md_engine in md_engines:
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
    md_params = {'timestep': 1, 'trajectory': None, 'dt': 1, 
                 'externalstress': 0, 'ttime': 25, 'pfactor': 3375, 
                 'mask': None, 'temperature': 500, 'taut': 1, 'taup': 1,
                 'pressure': 0, 'compressibility': 0, 'fixcm': 1}
    otf_params = {'dft_calc': dft_calc, 
                  'init_atoms': [0, 2, 4, 6],
                  'std_tolerance_factor': 2, 
                  'max_atoms_added' :8,
                  'freeze_hyps': 10, 
                  'use_mapping': super_cell.calc.use_mapping}
   
    # intialize velocity
    temperature = md_params['temperature']
    MaxwellBoltzmannDistribution(super_cell, temperature * units.kB)
    Stationary(super_cell)  # zero linear momentum
    ZeroRotation(super_cell)  # zero angular momentum

    test_otf = otf_md(md_engine, super_cell, md_params, otf_params)

    # set up logger
    test_otf.attach(OTFLogger(test_otf, super_cell, 
        logfile=md_engine+'.log', mode="w"), interval=1)
    
    # run otf
    number_of_steps = 3
    test_otf.otf_run(number_of_steps)

