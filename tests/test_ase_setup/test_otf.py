import os, shutil, glob
from copy import deepcopy
import pytest
import numpy as np

from flare.ase.otf_md import otf_md
from flare.ase.logger import OTFLogger

from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,\
                                         Stationary, ZeroRotation


def otf_md_test(md_engine):
    import atom_setup as atom_setup
    import flare_setup as flare_setup
    import qe_setup as qe_setup
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
    md_params = {'timestep': 1 * units.fs, 'trajectory': None, 'dt': 1*
                                                                     units.fs,
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
<<<<<<< HEAD
    # set up logger
    test_otf.attach(OTFLogger(test_otf, super_cell,
        logfile=md_engine+'.log', mode="w", data_in_logfile=True),
        interval=1)

=======

    print(test_otf.observers)

    # set up logger
    otf_logger = OTFLogger(test_otf, super_cell,
        logfile=md_engine+'.log', mode="w", data_in_logfile=True)
    test_otf.attach(otf_logger, interval=1)

>>>>>>> 5a18c1042a5767ebb8dc20ac59a17df6f1bcad77
    # run otf
    number_of_steps = 3
    test_otf.otf_run(number_of_steps)

    for f in glob.glob("AgI.pw*"):
        os.remove(f)
    for f in glob.glob("*.npy"):
        os.remove(f)
    for f in glob.glob("kv3*"):
        shutil.rmtree(f)

    for f in os.listdir("./"):
        if f in [f'{md_engine}.log', 'lmp.mgp']:
            os.remove(f)
        if f in ['out', 'otf_data']:
            shutil.rmtree(f)



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

