import sys
import os
from copy import deepcopy
import pytest
import numpy as np

from flare.ase.otf_md import otf_md
from flare.ase.logger import OTFLogger

from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)


def otf_md_test(md_engine):
    import atom_setup, flare_setup, qe_setup
    np.random.seed(12345)
    
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
    md_params = {'timestep': 1 * units.fs, 'trajectory': None, 'dt': None, 
                 'externalstress': 0, 'ttime': 25, 'pfactor': 3375, 
                 'mask': None, 'temperature': 500, 'taut': 1, 'taup': 1,
                 'pressure': 0, 'compressibility': 0, 'fixcm': 1, 
                 'friction': 0.02}
    otf_params = {'dft_calc': dft_calc, 
                  'init_atoms': [0, 1, 2, 3],
                  'std_tolerance_factor': 2, 
                  'max_atoms_added' : len(super_cell.positions),
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
        logfile=md_engine+'.log', mode="w", data_in_logfile=True), 
        interval=1)
    
    # run otf
    number_of_steps = 3
    test_otf.otf_run(number_of_steps)

    os.system('rm {}.log'.format(md_engine)) 
    os.system('rm AgI.pw*')
    os.system('rm -r out')
    os.system('rm -r __pycache__')
    os.system('rm -r kv3')
    os.system('rm lmp.mgp')
    os.system('rm -r otf_data')
    os.system('rm *.npy')


@pytest.mark.skipif(not os.environ.get('PWSCF_COMMAND',
                          False), reason='PWSCF_COMMAND not found '
                                  'in environment: Please install Quantum '
                                  'ESPRESSO and set the PWSCF_COMMAND env. '
                                  'variable to point to pw.x.')
def test_VelocityVerlet():
    otf_md_test('VelocityVerlet')

def test_NVTBerendsen():
    otf_md_test('NVTBerendsen')

def test_NPTBerendsen():
    otf_md_test('NPTBerendsen')

def test_NPT():
    otf_md_test('NPT')

def test_Langevin():
    otf_md_test('Langevin')

