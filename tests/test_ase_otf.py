import os, shutil, glob
from copy import deepcopy
import pytest
import numpy as np

from flare import otf, kernels
from flare.gp import GaussianProcess
from flare.mgp import MappedGaussianProcess
from flare.ase.calculator import FLARE_Calculator
from flare.ase.otf import ASE_OTF
# from flare.ase.logger import OTFLogger

from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)


md_list = ['VelocityVerlet', 'NVTBerendsen', 'NPTBerendsen', 'NPT', 'Langevin']

@pytest.fixture(scope='module')
def md_params():
    
    md_dict = {'temperature': 500}
    for md_engine in md_list:
        if md_engine == 'VelocityVerlet':
            md_dict[md_engine] = {}
        else:
            md_dict[md_engine] = {'temperature': md_dict['temperature']}

    md_dict['NVTBerendsen'].update({'taut': 0.5e3 * units.fs})
    md_dict['NPT'].update({'externalstress': 0, 'ttime': 25, 'pfactor': 3375})
    md_dict['Langevin'].update({'friction': 0.02})

    yield md_dict
    del md_dict


@pytest.fixture(scope='module')
def super_cell():

    from ase.spacegroup import crystal
    a = 3.855
    alpha = 90
    atoms = crystal(['H', 'He'], 
                    basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
                    size=(2, 1, 1),
                    cellpar=[a, a, a, alpha, alpha, alpha])

    # jitter positions to give nonzero force on first frame
    for atom_pos in atoms.positions:
        for coord in range(3):
            atom_pos[coord] += (2*np.random.random()-1) * 0.5

    yield atoms
    del atoms


@pytest.fixture(scope='module')
def flare_calc():
    flare_calc_dict = {}
    for md_engine in md_list:

        # ---------- create gaussian process model -------------------
        gp_model = GaussianProcess(kernel_name='2+3_mc',
                                   hyps=[0.1, 1., 0.001, 1, 0.06],
                                   cutoffs=(5.0, 5.0),
                                   hyp_labels=['sig2', 'ls2', 'sig3',
                                               'ls3', 'noise'],
                                   opt_algorithm='BFGS',
                                   par=False)

        # ----------- create mapped gaussian process ------------------
        grid_num_2 = 64 
        grid_num_3 = 16
        lower_cut = 0.1

        grid_params = {'load_grid': None,
                       'update': False}
 
        grid_params['twobody'] = {'lower_bound': [lower_cut],
                                  'grid_num': [grid_num_2],
                                  'svd_rank': 'auto'}

        grid_params['threebody'] = {'lower_bound': [lower_cut for d in range(3)],
                                    'grid_num': [grid_num_3 for d in range(3)],
                                    'svd_rank': 'auto'}
    
        species_list = [1, 2]
    
        mgp_model = MappedGaussianProcess(grid_params, species_list, n_cpus=1,
             map_force=False, mean_only=False)

        # ------------ create ASE's flare calculator -----------------------
        flare_calculator = FLARE_Calculator(gp_model, mgp_model=mgp_model,
                                            par=True, use_mapping=True)

        flare_calc_dict[md_engine] = flare_calculator
        print(md_engine)
    yield flare_calc_dict
    del flare_calc_dict


@pytest.fixture(scope='module')
def qe_calc():

    from ase.calculators.lj import LennardJones
    dft_calculator = LennardJones() 

    yield dft_calculator
    del dft_calculator


@pytest.mark.parametrize('md_engine', md_list)
def test_otf_md(md_engine, md_params, super_cell, flare_calc, qe_calc):
    np.random.seed(12345)

    flare_calculator = flare_calc[md_engine]
    # set up OTF MD engine
    otf_params = {'init_atoms': [0, 1, 2, 3],
                  'output_name': md_engine,
                  'std_tolerance_factor': 2,
                  'max_atoms_added' : len(super_cell.positions),
                  'freeze_hyps': 10}
#                  'use_mapping': flare_calculator.use_mapping}

    md_kwargs = md_params[md_engine]

    # intialize velocity
    temperature = md_params['temperature']
    MaxwellBoltzmannDistribution(super_cell, temperature * units.kB)
    Stationary(super_cell)  # zero linear momentum
    ZeroRotation(super_cell)  # zero angular momentum

    super_cell.set_calculator(flare_calculator)
    test_otf = ASE_OTF(super_cell, 
                       timestep = 1 * units.fs,
                       number_of_steps = 3,
                       dft_calc = qe_calc,
                       md_engine = md_engine,
                       md_kwargs = md_kwargs,
                       **otf_params)

    # TODO: test if mgp matches gp
    # TODO: see if there's difference between MD timestep & OTF timestep

    # set up logger
#    otf_logger = OTFLogger(test_otf, super_cell,
#        logfile=md_engine+'.log', mode="w", data_in_logfile=True)
#    test_otf.attach(otf_logger, interval=1)

    test_otf.run()

    for f in glob.glob("scf.pw*"):
        os.remove(f)
    for f in glob.glob("*.npy"):
        os.remove(f)
    for f in glob.glob("kv3*"):
        shutil.rmtree(f)

    for f in os.listdir("./"):
        if f in [f'{md_engine}.out', f'{md_engine}-hyps.dat', 'lmp.mgp']:
            os.remove(f)
        if f in ['out', 'otf_data']:
            shutil.rmtree(f)
