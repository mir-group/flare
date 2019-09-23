import sys
import os
import pytest
import numpy as np

from flare.ase.otf_md import OTF_NPT, OTF_VelocityVerlet
from flare.ase.logger import OTFLogger


@pytest.mark.skipif(not os.environ.get('PWSCF_COMMAND',
                          False), reason='PWSCF_COMMAND not found '
                                  'in environment: Please install Quantum '
                                  'ESPRESSO and set the PWSCF_COMMAND env. '
                                  'variable to point to pw.x.')
def test_otf():
    from ase import units
    from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                             Stationary, ZeroRotation)
    import atom_setup, flare_setup, qe_setup
    np.random.seed(12345)
    
    # ----------- set up atoms -----------------
    super_cell = atom_setup.super_cell
    
    # ----------- setup flare calculator ---------------
    flare_calc = flare_setup.flare_calc
    super_cell.set_calculator(flare_calc)
    
    # ----------- setup qe calculator --------------
    dft_calc = qe_setup.dft_calc
    
    # ----------- create otf object -----------
    timestep = 1 # fs
    temperature = 5000  # in kelvin
    externalstress = 0
    ttime = 25
    pfactor = 3375
    logfile = 'agi.log'
    
    # intialize velocity
    MaxwellBoltzmannDistribution(super_cell, temperature * units.kB)
    Stationary(super_cell)  # zero linear momentum
    ZeroRotation(super_cell)  # zero angular momentum
    
    # otf parameters
    std_tolerance_factor = 2
    init_atoms = [0, 5, 10, 15]
    no_cpus = 32
    max_atoms_added = 10
    freeze_hyps = 10
    
    # set up OTF MD engine
    #test_otf_npt = OTF_NPT(super_cell, timestep, temperature, 
    #                       externalstress, ttime, pfactor, mask=None, 
    test_otf_npt = OTF_VelocityVerlet(super_cell, timestep, 
                           # on-the-fly parameters
                           dft_calc=dft_calc, init_atoms=init_atoms,
                           std_tolerance_factor=std_tolerance_factor, 
                           max_atoms_added=max_atoms_added,
                           freeze_hyps=freeze_hyps, 
                           # mgp parameters
                           use_mapping=super_cell.calc.use_mapping)
    
    # set up logger
    test_otf_npt.attach(OTFLogger(test_otf_npt, super_cell, logfile, mode="w"), 
                        interval=1)
    
    # run otf
    number_of_steps = 5  # 10 ps
    test_otf_npt.otf_run(number_of_steps)
