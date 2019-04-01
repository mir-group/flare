import numpy as np
import sys
import crystals
import qe_input
import qe_util
import struc

pert_percs = [0.01, 0.05]  # percentage of lattice parameter
reps = 10  # number of repetitions
npool = 4

for pert_perc in pert_percs:
    for n in range(reps):

        # --------------------------------------------------------------------
        #                         1. set initial positions
        # --------------------------------------------------------------------

        # 32 atom bulk al
        symbol = 'Al'
        alat = 8.092 / 2
        unit_cell = np.eye(3) * alat
        sc_size = 2
        fcc_positions = crystals.fcc_positions(alat)
        positions = crystals.get_supercell_positions(sc_size, unit_cell,
                                                     fcc_positions)
        cell = unit_cell * 2

        # jitter the positions to give nonzero force on first frame
        for atom_pos in positions:
            for coord in range(3):
                atom_pos[coord] += (2*np.random.random()-1) * alat * pert_perc

        # --------------------------------------------------------------------
        #                 2. make qe input file and run qe
        # --------------------------------------------------------------------

        # setup qe file
        calculation = 'scf'
        input_file_name = 'repeat_'+str(pert_perc)+'_'+str(n)+'.in'
        output_file_name = 'repeat_'+str(pert_perc)+'_'+str(n)+'.out'
        pw_loc = '/n/home03/jonpvandermause/qe-6.2.1/bin/pw.x'
        pseudo_dir = '/n/home03/jonpvandermause/qe-6.2.1/pseudo'
        outdir = './output'
        nat = len(positions)
        ntyp = 1
        species = ['Al'] * nat
        ion_names = ['Al']
        mass = 27  # in amu
        ion_masses = [mass]
        ion_pseudo = ['Al.pbe-n-kjpaw_psl.1.0.0.UPF']
        kvec = np.array([7, 7, 7])
        ecutwfc = 29  # minimum recommended
        ecutrho = 143  # minimum recommended

        scf_inputs = dict(pseudo_dir=pseudo_dir,
                          outdir=outdir,
                          nat=nat,
                          ntyp=ntyp,
                          ecutwfc=ecutwfc,
                          ecutrho=ecutrho,
                          cell=cell,
                          species=species,
                          positions=positions,
                          kvec=kvec,
                          ion_names=ion_names,
                          ion_masses=ion_masses,
                          ion_pseudo=ion_pseudo)

        # make input file
        calc = qe_input.QEInput(input_file_name, output_file_name, pw_loc,
                                calculation, scf_inputs, metal=True)

        qe_forces = \
            qe_util.run_espresso_command(input_file_name, output_file_name,
                                         pw_loc, npool)

        force_file = 'forces_'+str(pert_perc)+'_'+str(n)
        position_file = 'positions_'+str(pert_perc)+'_'+str(n)
        np.save(force_file, qe_forces)
        np.save(position_file, positions)
