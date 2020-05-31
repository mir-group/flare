import os, shutil, glob
from copy import deepcopy
import pytest
import numpy as np

from flare import otf, kernels
from flare.gp import GaussianProcess
from flare.mgp.mgp import MappedGaussianProcess
from flare.ase.calculator import FLARE_Calculator
from flare.ase.otf import ASE_OTF
from flare.ase.logger import OTFLogger

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
#        struc_params = {'species': [1, 2],
#                        'cube_lat': np.eye(3) * 100,
#                        'mass_dict': {'0': 2, '1': 4}}
#
#        # grid parameters
#        lower_cut = 2.5
#        two_cut, three_cut = gp_model.cutoffs
#        grid_num_2 = 8
#        grid_num_3 = 8
#        grid_params = {'bounds_2': [[lower_cut], [two_cut]],
#                       'bounds_3': [[lower_cut, lower_cut, -1],
#                                    [three_cut, three_cut,  1]],
#                       'grid_num_2': grid_num_2,
#                       'grid_num_3': [grid_num_3, grid_num_3, grid_num_3],
#                       'svd_rank_2': 0,
#                       'svd_rank_3': 0,
#                       'bodies': [2, 3],
#                       'load_grid': None,
#                       'update': False}
#
#        mgp_model = MappedGaussianProcess(grid_params,
#                                          struc_params,
#                                          map_force=True,
#                                          GP=gp_model,
#                                          mean_only=False,
#                                          container_only=False,
#                                          lmp_file_name='lmp.mgp',
#                                          n_cpus=1)

        # ------------ create ASE's flare calculator -----------------------
        flare_calculator = FLARE_Calculator(gp_model, mgp_model=None,
                                            par=True, use_mapping=False)


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

    # set up logger
#    otf_logger = OTFLogger(test_otf, super_cell,
#        logfile=md_engine+'.log', mode="w", data_in_logfile=True)
#    test_otf.attach(otf_logger, interval=1)

    test_otf.run()

#    for f in glob.glob("scf.pw*"):
#        os.remove(f)
#    for f in glob.glob("*.npy"):
#        os.remove(f)
#    for f in glob.glob("kv3*"):
#        shutil.rmtree(f)
#
#    for f in os.listdir("./"):
#        if f in [f'{md_engine}.log', 'lmp.mgp']:
#            os.remove(f)
#        if f in ['out', 'otf_data']:
#            shutil.rmtree(f)
