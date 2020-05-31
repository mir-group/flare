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


md_list = ['VelocityVerlet'] #, 'NVTBerendsen', 'NPTBerendsen', 'NPT', 'Langevin']


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
        flare_calculator = FLARE_Calculator(gp_model, mgp_model,
                                            par=True, use_mapping=False)


        flare_calc_dict[md_engine] = flare_calculator
        print(md_engine)
    yield flare_calc_dict
    del flare_calc_dict


@pytest.mark.skipif(not os.environ.get('PWSCF_COMMAND',
                          False), reason='PWSCF_COMMAND not found '
                                  'in environment: Please install Quantum '
                                  'ESPRESSO and set the PWSCF_COMMAND env. '
                                  'variable to point to pw.x.')
@pytest.fixture(scope='module')
def qe_calc():

    from ase.calculators.espresso import Espresso
    # set up executable
    label = 'scf'
    input_file = label+'.pwi'
    output_file = label+'.pwo'
    no_cpus = 1
    pw = os.environ.get('PWSCF_COMMAND')
    os.environ['ASE_ESPRESSO_COMMAND'] = f'{pw} < {input_file} > {output_file}'

    # set up input parameters
    input_data = {'control':   {'prefix': label,
                                'pseudo_dir': 'test_files/pseudos/',
                                'outdir': './out',
                                'calculation': 'scf'},
                  'system':    {'ibrav': 0,
                                'ecutwfc': 20,
                                'ecutrho': 40,
                                'smearing': 'gauss',
                                'degauss': 0.02,
                                'occupations': 'smearing'},
                  'electrons': {'conv_thr': 1.0e-02,
                                'electron_maxstep': 100,
                                'mixing_beta': 0.7}}

    # pseudo-potentials
    ion_pseudo = {'H': 'H.pbe-kjpaw.UPF',
                  'He': 'He.pbe-kjpaw_psl.1.0.0.UPF'}

    # create ASE calculator
    dft_calculator = Espresso(pseudopotentials=ion_pseudo, label=label,
                              tstress=True, tprnfor=True, nosym=True,
                              input_data=input_data, kpts=(1,1,1))

    yield dft_calculator
    del dft_calculator


@pytest.mark.skipif(not os.environ.get('PWSCF_COMMAND',
                          False), reason='PWSCF_COMMAND not found '
                                  'in environment: Please install Quantum '
                                  'ESPRESSO and set the PWSCF_COMMAND env. '
                                  'variable to point to pw.x.')
@pytest.mark.parametrize('md_engine', md_list)
def test_otf_md(md_engine, super_cell, flare_calc, qe_calc):
    np.random.seed(12345)

    flare_calculator = flare_calc[md_engine]
    # set up OTF MD engine
    md_params = {'externalstress': 0, 'ttime': 25, 'pfactor': 3375,
                 'mask': None, 'temperature': 500, 'taut': 1, 'taup': 1,
                 'pressure': 0, 'compressibility': 0, 'fixcm': 1,
                 'friction': 0.02}

    otf_params = {'init_atoms': [0, 1, 2, 3],
                  'std_tolerance_factor': 2,
                  'max_atoms_added' : len(super_cell.positions),
                  'freeze_hyps': 10}
#                  'use_mapping': flare_calculator.use_mapping}

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
                       md_kwargs = {},
                       **otf_params)

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
        if f in [f'{md_engine}.log', 'lmp.mgp']:
            os.remove(f)
        if f in ['out', 'otf_data']:
            shutil.rmtree(f)
