import os
import sys
import numpy as np
from flare.otf_parser import OtfAnalysis
from flare.kernels import two_plus_three_body, two_plus_three_body_grad
from flare.mgp.mgp import MappedGaussianProcess
from flare.ase.calculator import FLARE_Calculator
from flare.mc_simple import two_plus_three_body_mc, two_plus_three_body_mc_grad
from flare.env import AtomicEnvironment
from flare.predict import predict_on_structure

from ase.spacegroup import crystal

def test_stress_with_lammps():
    """
    Based on gp_test_al.out, ensures that given hyperparameters and DFT calls
    a GP model can be reproduced and correctly re-predict forces and
    uncertainties
    :return:
    """

    # build up GP from a previous trajectory
    parsed = OtfAnalysis('test_files/VelocityVerlet.log')

    positions = parsed.position_list
    forces = parsed.force_list

    gp_model = parsed.make_gp(kernel=two_plus_three_body_mc,
                              kernel_grad=two_plus_three_body_mc_grad)

    # build up MGP from GP
    struc_params = {'species': [47, 53],
                    'cube_lat': np.eye(3) * 100,
                    'mass_dict': {'0': 27, '1': 16}}
    
    # grid parameters
    lower_cut = 2.5
    grid_num_2 = 64
    grid_num_3 = 32
    two_cut = 5.0
    three_cut = 5.0
    grid_params = {'bounds_2': [[lower_cut], [two_cut]],
                   'bounds_3': [[lower_cut, lower_cut, -1],
                                [three_cut, three_cut,  1]],
                   'grid_num_2': grid_num_2,
                   'grid_num_3': [grid_num_3, grid_num_3, grid_num_3],
                   'svd_rank_2': 0,
                   'svd_rank_3': 0,
                   'bodies': [2, 3],
                   'load_grid': None,
                   'update': True}
    
    mgp_model = MappedGaussianProcess(gp_model.hyps, gp_model.cutoffs,
                grid_params, struc_params, mean_only=True, container_only=False,
                GP=gp_model, lmp_file_name='lmp.mgp')

    # ------------ create ASE's flare calculator -----------------------
    flare_calc = FLARE_Calculator(gp_model, mgp_model, par=True, use_mapping=True)

    a = 3.855
    alpha = 90 
    super_cell = crystal(['Ag', 'I'], # Ag, I 
                 basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
                 size=(2, 1, 1),
                 cellpar=[a, a, a, alpha, alpha, alpha])
    super_cell.positions = positions[-1]
    super_cell.set_calculator(flare_calc)
    super_cell.get_forces()
    stresses = super_cell.calc.results['stresses']

    # parse lammps stress 
    lmp_file = open('test_files/stress.lammps')
    lines = lmp_file.readlines()[9:]
    for ind, line in enumerate(lines):
        line = line.split()
        strs = np.array([float(l) for l in line[1:]]) / 1.60217662e6
        assert np.isclose(stresses[ind], strs, rtol=1e-3).all()

    os.system('rm -r __pycache__')
    os.system('rm grid3*')
    os.system('rm -r kv3')
    os.system('rm lmp.mgp')




