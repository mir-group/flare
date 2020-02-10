import numpy as np
import time
import pytest
from flare import struc, env, gp
from flare import otf_parser
from flare.mgp.mgp import MappedGaussianProcess
from flare import mc_simple
from flare.lammps import lammps_calculator
import pickle
import os


# ASSUMPTION: You have a Lammps executable with the mgp pair style with $lmp
# as the corresponding environment variable.
@pytest.mark.skipif(not os.environ.get('lmp',
                          False), reason='lmp not found '
                                  'in environment: Please install LAMMPS '
                                  'and set the $lmp env. '
                                  'variable to point to the executatble.')

def construct_gp(bodies, params):

    unique_species, hyps, cutoffs, noa, nenv, cell = params

    # construct a 2b gp model
    if bodies==2:
        kernel = mc_simple.two_body_mc #plus_three_body_mc
        kernel_grad = mc_simple.two_body_mc_grad # plus_three_body_mc_grad
    if bodies == 3:
        kernel = mc_simple.three_body_mc #plus_three_body_mc
        kernel_grad = mc_simple.three_body_mc_grad # plus_three_body_mc_grad

    gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps,
                           cutoffs, par=True, ncpus=2)
    gp_model.hyp_labels = ['sig2', 'ls2', 'noise']

    np.random.seed(0)

    # create training_set
    gp_model.training_data = []
    gp_model.training_labels_np = []
    for idenv in range(nenv):
        positions = np.random.uniform(-1, 1, [noa,3])
        forces = np.random.uniform(-1, 1, [noa,3])
        species = [unique_species[i] for i in \
                np.random.randint(0, len(unique_species), noa)]
        structure = struc.Structure(cell, species, positions, forces=forces)
        gp_model.update_db(structure, forces, [1])
    test_envi = env.AtomicEnvironment(structure, 1, cutoffs)
    struc_test = structure
    gp_model.set_L_alpha()
    return gp_model, test_envi, struc_test


@pytest.mark.parametrize('bodies', [ 2, 3])
def test_parse_header(bodies):

    unique_species = [2, 1]
    hyps = [1, 1, 1, 1, 0.01]
    cutoffs = [0.8, 0.8]
    noa = 5
    nenv=10
    cell = np.eye(3)

    params = (unique_species, hyps, cutoffs, noa, nenv, cell)

    gp_model, test_envi, struc_test = construct_gp(bodies, params)

    grid_num_2 = 128
    grid_num_3 = 15
    lower_cut = 0.01
    two_cut = cutoffs[0]
    three_cut = cutoffs[1]
    lammps_location = 'AgI_Molten_15.txt'

    # set struc params. cell and masses arbitrary?
    mapped_cell = np.eye(3) * 2
    struc_params = {'species': unique_species,
                    'cube_lat': mapped_cell,
                    'mass_dict': {'0': 27, '1': 16}}

    # grid parameters
    grid_params = {'bounds_2': [[lower_cut], [two_cut]],
                   'bounds_3': [[lower_cut, lower_cut, 0],
                                [three_cut, three_cut, np.pi]],
                   'grid_num_2': grid_num_2,
                   'grid_num_3': [grid_num_3, grid_num_3, grid_num_3],
                   'svd_rank_2': 64,
                   'svd_rank_3': 90,
                   'bodies': [bodies],
                   'load_grid': None,
                   'update': True}

    mgp_model = MappedGaussianProcess(gp_model.hyps, gp_model.cutoffs,
            grid_params, struc_params, mean_only=True, container_only=False,
            GP=gp_model, lmp_file_name=lammps_location, ncpus=4)

    # -------------------------------------------------------------------------
    #                          test the mapped potential
    # -------------------------------------------------------------------------

    gp_pred_x = gp_model.predict(test_envi, 1)
    mgp_pred = mgp_model.predict(test_envi, mean_only=True)

    # check mgp is within 1 meV/A of the gp
    assert(np.abs(mgp_pred[0][0] - gp_pred_x[0]) < 1e-3), f"{bodies} body mapping is wrong"
    # -------------------------------------------------------------------------
    #                           check lammps potential
    # -------------------------------------------------------------------------

    # mgp_model.write_lmp_file(lammps_location)
    # lmp file is automatically written now every time MGP is constructed

    # create test structure

    atom_types = [1, 2]
    atom_masses = [108, 127]
    atom_species = struc_test.coded_species
    # create data file
    data_file_name = 'tmp.data'
    data_text = lammps_calculator.lammps_dat(struc_test, atom_types,
                                             atom_masses, atom_species)
    lammps_calculator.write_text(data_file_name, data_text)

    # create lammps input
    style_string = 'mgp'
    coeff_string = '* * {} 1 2 yes yes'.format(lammps_location)
    lammps_executable = os.environ.get('lmp')
    dump_file_name = 'tmp.dump'
    input_file_name = 'tmp.in'
    output_file_name = 'tmp.out'
    input_text = \
        lammps_calculator.generic_lammps_input(data_file_name, style_string,
                                               coeff_string, dump_file_name)
    lammps_calculator.write_text(input_file_name, input_text)

    lammps_calculator.run_lammps(lammps_executable, input_file_name,
                                 output_file_name)

    lammps_forces = lammps_calculator.lammps_parser(dump_file_name)
    mgp_forces = mgp_model.predict(test_envi, mean_only=True)

    # check that lammps agrees with gp to within 1 meV/A
    assert(np.abs(lammps_forces[1, 1] - mgp_forces[0][1]) < 1e-3)

    os.system('rm tmp.in tmp.out tmp.dump tmp.data'
              ' log.lammps')
    os.system('rm '+lammps_location)
    os.system('rm grid3*.npy')
    # os.system(f'rm -r {mgp_model.kv3name}')
