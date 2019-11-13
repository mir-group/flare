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
def test_parse_header():
    # -------------------------------------------------------------------------
    #                  reconstruct gp model from otf snippet
    # -------------------------------------------------------------------------

    unique_species = [2, 1]
    hyps = [1, 1, 0.01]
    cutoffs = [0.8, 0.8]
    noa = 5
    nenv=10
    cell = np.eye(3)

    # construct a 2b gp model
    kernel = mc_simple.two_body_mc #plus_three_body_mc
    kernel_grad = mc_simple.two_body_mc_grad # plus_three_body_mc_grad
    gp_2b = gp.GaussianProcess(kernel, kernel_grad, hyps,
                           cutoffs, par=True, no_cpus=2)
    gp_2b.hyp_labels = ['sig2', 'ls2', 'noise']

    # create training_set
    gp_2b.training_data = []
    gp_2b.training_labels_np = []
    for idenv in range(nenv):
        positions = np.random.uniform(-1, 1, [noa,3])
        forces = np.random.uniform(-1, 1, [noa,3])
        species = [unique_species[i] for i in \
                np.random.randint(0, len(unique_species), noa)]
        structure = struc.Structure(cell, species, positions, forces=forces)
        gp_2b.update_db(structure, forces, [1])
    test2b_envi = env.AtomicEnvironment(structure, 1, cutoffs)
    structure_2b = structure
    gp_2b.set_L_alpha()

    # construct a 3b gp model
    kernel = mc_simple.three_body_mc #plus_three_body_mc
    kernel_grad = mc_simple.three_body_mc_grad # plus_three_body_mc_grad
    gp_3b = gp.GaussianProcess(kernel, kernel_grad, hyps,
                           cutoffs, par=True, no_cpus=2)
    gp_3b.hyp_labels = ['sig2', 'ls2', 'noise']

    # create training_set
    gp_3b.training_data = []
    gp_3b.training_labels_np = []
    for idenv in range(nenv):
        positions = np.random.uniform(-1, 1, [noa,3])
        forces = np.random.uniform(-1, 1, [noa,3])
        species = [unique_species[i] for i in \
                np.random.randint(0, len(unique_species), noa)]
        structure = struc.Structure(cell, species, positions)
        gp_3b.update_db(structure, forces, [1])
    test3b_envi = env.AtomicEnvironment(structure, 1, cutoffs)
    structure_3b = structure
    gp_3b.set_L_alpha()
    # -------------------------------------------------------------------------
    #                              map the potential
    # -------------------------------------------------------------------------

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
                   'bodies': [2],
                   'load_grid': None,
                   'update': True}

    mgp_2b = MappedGaussianProcess(gp_2b.hyps, gp_2b.cutoffs,
            grid_params, struc_params, mean_only=True, container_only=False,
            GP=gp_2b, lmp_file_name=lammps_location)

    # -------------------------------------------------------------------------
    #                          test the mapped potential
    # -------------------------------------------------------------------------

    gp_pred_x = gp_2b.predict(test2b_envi, 1)
    mgp_pred = mgp_2b.predict(test2b_envi, mean_only=True)

    # check mgp is within 1 meV/A of the gp
    assert(np.abs(mgp_pred[0][0] - gp_pred_x[0]) < 1e-3)
    # -------------------------------------------------------------------------
    #                           check lammps potential
    # -------------------------------------------------------------------------

    # mgp_2b.write_lmp_file(lammps_location)
    # lmp file is automatically written now every time MGP is constructed

    # create test structure

    atom_types = [1, 2]
    atom_masses = [108, 127]
    atom_species = structure_2b.coded_species
    # create data file
    data_file_name = 'tmp.data'
    data_text = lammps_calculator.lammps_dat(structure_2b, atom_types,
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
    mgp_forces = mgp_2b.predict(test2b_envi, mean_only=True)

    # check that lammps agrees with gp to within 1 meV/A
    assert(np.abs(lammps_forces[1, 1] - mgp_forces[0][1]) < 1e-3)

    os.system('rm tmp.in tmp.out tmp.dump tmp.data'
              ' log.lammps')
    os.system('rm '+lammps_location)
    os.system('rm grid3*.npy')
    # os.system(f'rm -r {mgp_2b.kv3name}')
