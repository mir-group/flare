import sys
import os
import pytest
import numpy as np

from flare.ase.otf_md import OTF_NPT, OTF_VelocityVerlet
from flare.ase.logger import OTFLogger

from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
import atom_setup, flare_setup, qe_setup

@pytest.mark.skipif(not os.environ.get('PWSCF_COMMAND',
                          False), reason='PWSCF_COMMAND not found '
                                  'in environment: Please install Quantum '
                                  'ESPRESSO and set the PWSCF_COMMAND env. '
                                  'variable to point to pw.x.')
def test_otf():
    np.random.seed(12345)
    
    # ----------- set up atoms -----------------
    super_cell = atom_setup.super_cell
    
    # ----------- setup flare calculator ---------------
    flare_calc = flare_setup.flare_calc
    super_cell.set_calculator(flare_calc)
    
    # ----------- setup qe calculator --------------
    dft_calc = qe_setup.dft_calc
    
    # ----------- create otf object -----------
   
    # intialize velocity
    temperature = 5000
    MaxwellBoltzmannDistribution(super_cell, temperature * units.kB)
    Stationary(super_cell)  # zero linear momentum
    ZeroRotation(super_cell)  # zero angular momentum
    
    # set up OTF MD engine
    #test_otf_npt = OTF_NPT(super_cell, timestep=1, temperature=temperature, 
    #                       externalstress=0, ttime=25, 
    #                       pfactor=3375, mask=None, 
    test_otf_npt = OTF_VelocityVerlet(super_cell, timestep=1, 
                           # on-the-fly parameters
                           dft_calc=dft_calc, 
                           init_atoms=[0, 2, 4, 6],
                           std_tolerance_factor=2, 
                           max_atoms_added=8,
                           freeze_hyps=10, 
                           # mgp parameters
                           use_mapping=super_cell.calc.use_mapping)
    
    # set up logger
    test_otf_npt.attach(OTFLogger(test_otf_npt, super_cell, logfile='agi.log',
                        mode="w"), interval=1)
    
    # run otf
    number_of_steps = 3
    test_otf_npt.otf_run(number_of_steps)
