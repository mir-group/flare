import numpy as np
import sys
import crystals
import copy
import eam
sys.path.append('../../otf/otf_engine')
import gp, struc, qe_parsers, env


# given a gp model, predict vacancy diffusion activation profile
def vac_diff_fcc(gp_model, gp_cell, fcc_cell, species, cutoff, nop):

    # create 2x2x2 fcc supercell with atom 0 removed
    alat = fcc_cell[0, 0]
    fcc_unit = crystals.fcc_positions(alat)
    fcc_super = crystals.get_supercell_positions(2, fcc_cell, fcc_unit)
    vac_super = copy.deepcopy(fcc_super)
    vac_super.pop(0)
    vac_super = np.array(vac_super)

    # create list of positions for the migrating atom
    start_pos = vac_super[0]
    end_pos = np.array([0, 0, 0])
    diff_vec = end_pos - start_pos
    test_list = []
    step = diff_vec / (nop - 1)
    for n in range(nop):
        test_list.append(start_pos + n*step)

    # predict force on migrating atom
    vac_copy = copy.deepcopy(vac_super)
    store_res = np.zeros((6, nop))

    cutoffs = np.array([cutoff, cutoff])

    for count, pos_test in enumerate(test_list):
        vac_copy[0] = pos_test

        struc_curr = struc.Structure(gp_cell, species, vac_copy)
        env_curr = env.AtomicEnvironment(struc_curr, 0, cutoffs)

        fc_comp_x, vr_x = gp_model.predict(env_curr, 1)
        fc_comp_y, vr_y = gp_model.predict(env_curr, 2)
        fc_comp_z, vr_z = gp_model.predict(env_curr, 3)

        store_res[0, count] = fc_comp_x
        store_res[1, count] = fc_comp_y
        store_res[2, count] = fc_comp_z
        store_res[3, count] = vr_x
        store_res[4, count] = vr_y
        store_res[5, count] = vr_z

    return store_res


# given a gp model, predict vacancy diffusion activation profile
def vac_diff_lammps(style_string, coeff_string, lammps_folder,
                    lammps_executable, gp_cell, fcc_cell, species,
                    nop):

    # create 2x2x2 fcc supercell with atom 0 removed
    alat = fcc_cell[0, 0]
    fcc_unit = crystals.fcc_positions(alat)
    fcc_super = crystals.get_supercell_positions(2, fcc_cell, fcc_unit)
    vac_super = copy.deepcopy(fcc_super)
    vac_super.pop(0)
    vac_super = np.array(vac_super)

    # create list of positions for the migrating atom
    start_pos = vac_super[0]
    end_pos = np.array([0, 0, 0])
    diff_vec = end_pos - start_pos
    test_list = []
    step = diff_vec / (nop - 1)
    for n in range(nop):
        test_list.append(start_pos + n*step)

    # predict force on migrating atom
    vac_copy = copy.deepcopy(vac_super)
    store_res = np.zeros((3, nop))

    for count, pos_test in enumerate(test_list):
        vac_copy[0] = pos_test

        struc_curr = struc.Structure(gp_cell, species, vac_copy)

        lammps_calculator = \
            eam.EAM_Force_Calculator(struc_curr, style_string, coeff_string,
                                     lammps_folder, lammps_executable)

        lammps_forces = lammps_calculator.get_forces()

        store_res[0, count] = lammps_forces[0, 0]
        store_res[1, count] = lammps_forces[0, 1]
        store_res[2, count] = lammps_forces[0, 2]

    return store_res
