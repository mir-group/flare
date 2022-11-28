import numpy as np
from copy import deepcopy
import os


def parse_outfiles(output_dir, ase_output_reader):
    "Return data from DFT output files as np arrays."

    # Get list of output files.
    output_files = os.listdir(output_dir)

    # Remove extraneous files (e.g. .DS_Store on Mac)
    n_frames = len(output_files)
    for n in range(n_frames):
        check_ind = n_frames - 1 - n
        if output_files[check_ind].startswith("."):
            output_files.pop(check_ind)

    output_files.sort()

    # Count the number of atoms.
    traj = list(ase_output_reader(output_dir + output_files[0], index=slice(None)))
    n_atoms = len(traj[0])

    # Initialize numpy arrays.
    dft_data = {}
    dft_data["positions"] = np.zeros((n_frames, n_atoms, 3))
    dft_data["cells"] = np.zeros((n_frames, 3, 3))
    dft_data["atomic_numbers"] = np.zeros((n_frames, n_atoms))
    dft_data["energies"] = np.zeros(n_frames)
    dft_data["forces"] = np.zeros((n_frames, n_atoms, 3))
    dft_data["stresses"] = np.zeros((n_frames, 3, 3))

    # Parse with ASE.
    for n, file_name in enumerate(output_files):
        current_file = output_dir + file_name
        traj = list(ase_output_reader(current_file, index=slice(None)))

        dft_data["positions"][n] = traj[0].get_positions()  # Angstrom
        dft_data["cells"][n] = traj[0].get_cell()  # Angstrom
        dft_data["atomic_numbers"][n] = traj[0].numbers
        dft_data["energies"][n] = traj[0].get_potential_energy()  # eV
        dft_data["forces"][n] = traj[0].get_forces()  # eV/A
        dft_data["stresses"][n] = -traj[0].get_stress(voigt=False)  # eV/A^3

    return dft_data


def parse_otf(filename):
    """Parse an ASE OTF output file."""

    # Labels of data to be parsed.
    data_labels = [
        "positions",
        "forces",
        "uncertainties",
        "frames",
        "times",
        "temperatures",
        "energies",
        "calc_types",
        "cells",
        "stresses",
        "pressures",
        "store_atoms",
        "store_natoms",
        "store_hyps",
    ]

    # Initialize data lists.
    all_data = {}
    for data_label in data_labels:
        all_data[data_label] = []

    header = 0
    n_hyps = 0

    with open(filename, "r") as f:
        lines = f.readlines()

    for index, line in enumerate(lines):
        if line.startswith("Number of atoms"):
            at_line = line.split()
            noa = int(at_line[-1])

        if line.startswith("Number of hyperparameters:"):
            n_hyps = int(lines[index].split()[3])

        if line.startswith("*-Frame") or line.startswith("-Frame"):
            parse_snapshot(lines, index, noa, all_data)

        if line.startswith("Simulation Time:"):
            all_data["times"].append(float(line.split()[2]))

        if line.startswith("Temperature"):
            all_data["temperatures"].append(float(lines[index].split()[2]))

        if line.startswith("Potential energy"):
            all_data["energies"].append(float(lines[index].split()[3]))

        if line.startswith("Periodic cell"):
            # Skip the initial cell.
            if header == 0:
                header = 1
            else:
                vectors = []
                for cell_line in lines[index + 1 : index + 4]:
                    cell_line = cell_line.strip().replace("[", "").replace("]", "")
                    vec = cell_line.split()
                    vector = [float(vec[0]), float(vec[1]), float(vec[2])]
                    vectors.append(vector)
                all_data["cells"].append(np.array(vectors))

        # Record stress.
        if line.startswith("Stress tensor (GPa)"):
            stress_line = lines[index + 2].split()
            stress_vec = []
            for stress_comp in stress_line:
                stress_vec.append(float(stress_comp))
            stress_vec = np.array(stress_vec)
            all_data["stresses"].append(stress_vec)

        # Record pressure.
        if line.startswith("Pressure"):
            all_data["pressures"].append(float(lines[index].split()[2]))

        # Record sparse points.
        if line.startswith("Adding atom"):
            line_split = line.split()
            for ind, string in enumerate(line_split):
                if string == "to":
                    end_ind = ind
            atoms1 = line_split[2:end_ind]
            for n, at in enumerate(atoms1):
                atoms1[n] = at.strip("[").strip("]").strip(",")
            atoms1 = [int(n) for n in atoms1]

            all_data["store_atoms"].append(atoms1)
            all_data["store_natoms"].append(len(atoms1))

        # Record hyperparameters.
        if line.startswith("GP hyperparameters"):
            hyps = []
            for n in range(n_hyps):
                hyps.append(float(lines[index + 1 + n].split()[-1]))
            all_data["store_hyps"].append(hyps)

    dft_data, gp_data = organize_data(all_data)

    return dft_data, gp_data


def organize_data(all_data):
    "Partition output data by calculation type (DFT or GP)."

    dft_data = deepcopy(all_data)
    gp_data = deepcopy(all_data)

    skip = ["store_hyps", "store_atoms", "store_natoms"]
    dft_index = 0
    gp_index = 0

    for n in range(len(all_data["calc_types"])):
        if all_data["calc_types"][n] == 0:
            for key in dft_data:
                if key not in skip:
                    dft_data[key].pop(dft_index)
            dft_index -= 1

        elif all_data["calc_types"][n] == 1:
            for key in gp_data:
                if key not in skip:
                    gp_data[key].pop(gp_index)
            gp_index -= 1

        dft_index += 1
        gp_index += 1

    return dft_data, gp_data


def parse_snapshot(lines, index, noa, all_data):
    """Parse positions, forces and uncertainties of an OTF frame."""

    line = lines[index]
    all_data["frames"].append(int(line.split()[1]))

    # Record calculation type (DFT of GP).
    if line.startswith("*-Frame"):
        all_data["calc_types"].append(1)
    elif line.startswith("-Frame"):
        all_data["calc_types"].append(0)

    # Initialize values.
    species = []
    positions = np.zeros((noa, 3))
    forces = np.zeros((noa, 3))
    uncertainties = np.zeros((noa, 3))
    velocities = np.zeros((noa, 3))

    # Current setting for # of lines to skip after Frame marker
    skip = 3

    for count, frame_line in enumerate(lines[(index + skip) : (index + skip + noa)]):
        # parse frame line
        spec, position, force, uncertainty, velocity = parse_frame_line(frame_line)

        # update values
        species.append(spec)
        positions[count] = position
        forces[count] = force
        uncertainties[count] = uncertainty
        velocities[count] = velocity

    all_data["positions"].append(positions)
    all_data["forces"].append(forces)
    all_data["uncertainties"].append(uncertainties)


def parse_frame_line(frame_line):
    """parse a line in otf output.
    :param frame_line: frame line to be parsed
    :type frame_line: string
    :return: species, position, force, uncertainty, and velocity of atom
    :rtype: list, np.arrays
    """

    frame_line = frame_line.split()

    spec = str(frame_line[0])
    position = np.array([float(n) for n in frame_line[1:4]])
    force = np.array([float(n) for n in frame_line[4:7]])
    uncertainty = np.array([float(n) for n in frame_line[7:10]])
    velocity = np.array([float(n) for n in frame_line[10:13]])

    return spec, position, force, uncertainty, velocity
