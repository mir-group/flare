import sys, subprocess
import numpy as np

from copy import deepcopy
from typing import List, Tuple
from flare.bffs.gp import GaussianProcess
from flare.atoms import FLARE_Atoms


class OtfAnalysis:
    """
    Parse the OTF log file to get trajectory, training data,
    thermostat, and build GP model.

    Args:
        filename (str): name of the OTF log file.
        calculate_energy (bool): if the potential energy is computed and
            needs to be parsed, then set to True. Default False.
    """

    def __init__(self, filename, calculate_energy=False):
        self.filename = filename

        self.calculate_energy = calculate_energy

        blocks = split_blocks(filename)

        self.header = parse_header_information(blocks[0])
        self.noa = self.header["atoms"]
        self.noh = self.header["n_hyps"]

        self.position_list = []
        self.cell_list = []
        self.force_list = []
        self.stress_list = []
        self.uncertainty_list = []
        self.velocity_list = []
        self.temperatures = []
        self.dft_frames = []
        self.dft_times = []
        self.times = []
        self.energies = []
        self.thermostat = {}

        self.gp_position_list = []
        self.gp_cell_list = []
        self.gp_force_list = []
        self.gp_stress_list = []
        self.gp_uncertainty_list = []
        self.gp_velocity_list = []
        self.gp_atom_list = []
        self.gp_species_list = []
        self.gp_atom_count = []
        self.gp_thermostat = {}

        self.gp_hyp_list = [self.header["hyps"]]

        self.mae_list = []
        self.mav_list = []

        self.parse_pos_otf(blocks[1:])

        if self.calculate_energy:
            self.energies = self.thermostat["potential energy"]

    def make_gp(
        self,
        cell=None,
        call_no=None,
        hyps=None,
        init_gp=None,
        hyp_no=None,
        **kwargs,
    ):
        """
        Build GP model from the training frames parsed from the log file.
        The cell, hyps and gp can be reset with customized values.

        Args:
            cell (np.ndarray): Default None to use the cell from the log file.
                A customized cell can be input as a 3x3 numpy array.
            call_no (int): Default None to use all the DFT frames as training
                data for building GP. If not None, then the frames 0 to `call_no`
                will be added to GP.
            hyps (np.ndarray): Default None to use the hyperparameters from the
                log file. Customized hyps can be input as an array.
            init_gp (GaussianProcess): Default to None to use no initial settings
                or training data. an initial GP can be used, and then the
                frames parsed in the log file will add to the initial GP. Then the
                final GP uses the hyps and kernels of `init_gp`, and consists of
                training data from `init_gp` and the data from the log file.
                **NOTE**: if a log file from restarted OTF is parsed, then an initial
                GP needs to be parsed from the prior log file as the `init_gp` of the
                restarted log file.
            hyp_no (int): Default None to use the final optimized hyperparameters to
                build GP. If not None, then use the hyps from the `hyp_no`th
                optimization step.
            kwargs: if a new GP setting is needed without inputing `init_gp`, the GP
                initial args can be input as kwargs.
        """
        if "restart" in self.header and self.header["restart"] > 0:
            assert (
                init_gp is not None
            ), "Please input the init_gp as the gp model dumppedbefore restarting otf."

        if call_no is None:
            call_no = len(self.gp_position_list)
        if hyp_no is None:
            hyp_no = len(self.gp_hyp_list)  # use the last hyps by default
        if hyps is None:
            # check out the last non-empty element from the list
            hyps = self.gp_hyp_list[hyp_no - 1]
        if cell is None:
            cell = self.header["cell"]

        if init_gp is None:
            # Use run's values as extracted from header
            # TODO Allow for kernel gradient in header

            dictionary = deepcopy(self.header)
            dictionary["hyps"] = hyps
            for k in kwargs:
                if kwargs[k] is not None:
                    dictionary[k] = kwargs[k]

            gp_model = GaussianProcess.from_dict(dictionary)
        else:
            gp_model = init_gp
            gp_model.hyps = hyps

        for positions, forces, atoms, species in zip(
            self.gp_position_list[:call_no],
            self.gp_force_list[:call_no],
            self.gp_atom_list[:call_no],
            self.gp_species_list[:call_no],
        ):

            struc_curr = FLARE_Atoms(cell=cell, symbols=species, positions=positions)

            gp_model.update_db(struc_curr, forces, custom_range=atoms)

        gp_model.set_L_alpha()

        return gp_model

    @staticmethod
    def get_gp_activation(gp_model):
        pass

    def parse_pos_otf(self, blocks):
        """
        Exclusively parses MD run information
        :param filename:
        :return:
        """
        n_steps = len(blocks) - 1

        for block in blocks:
            for index, line in enumerate(block):
                # DFT frame
                if line.startswith("*-Frame"):
                    dft_frame_line = line.split()
                    self.dft_frames.append(int(dft_frame_line[1]))
                    dft_time_line = block[index + 1].split()
                    self.dft_times.append(float(dft_time_line[-2]))

                    # TODO: generalize this to account for arbitrary starting list
                    append_atom_lists(
                        self.gp_species_list,
                        self.gp_position_list,
                        self.gp_force_list,
                        self.gp_uncertainty_list,
                        self.gp_velocity_list,
                        block,
                        index,
                        self.noa,
                        True,
                        self.noh,
                    )

                    post_frame = block[index + 3 + self.noa :]
                    extract_global_info(
                        self.gp_cell_list,
                        self.gp_stress_list,
                        self.gp_thermostat,
                        post_frame,
                    )

                    extract_gp_info(
                        post_frame,
                        self.mae_list,
                        self.mav_list,
                        self.gp_atom_list,
                        self.gp_hyp_list,
                        self.noh,
                    )

                # MD frame
                if line.startswith("-Frame"):
                    n_steps += 1
                    time_line = block[index + 1].split()
                    sim_time = float(time_line[2])
                    self.times.append(sim_time)

                    # TODO: generalize this to account for arbitrary starting list
                    append_atom_lists(
                        [],
                        self.position_list,
                        self.force_list,
                        self.uncertainty_list,
                        self.velocity_list,
                        block,
                        index,
                        self.noa,
                        False,
                        self.noh,
                    )

                    post_frame = block[index + 3 + self.noa :]
                    extract_global_info(
                        self.cell_list,
                        self.stress_list,
                        self.thermostat,
                        post_frame,
                    )

    def get_msds(self):
        msds = []
        for pos in self.position_list:
            msds.append(np.mean((pos - self.position_list[0]) ** 2))
        return msds

    def output_md_structures(self):
        """
        Returns structure objects corresponding to the MD frames of an OTF run.
        :return:
        """

        structures = []
        cell = self.header["cell"]
        species = self.header["species"]
        for i in range(len(self.position_list)):
            if not self.calculate_energy:
                energy = 0
            else:
                energy = self.energies[i]

            cur_struc = FLARE_Atoms(
                cell=cell,
                symbols=species,
                positions=self.position_list[i],
            )
            cur_struc.forces = np.array(self.force_list[i])
            cur_struc.stds = np.array(self.uncertainty_list[i])
            cur_struc.energy = energy
            # cur_struc.stress = self.stress_list[i]
            structures.append(cur_struc)
        return structures

    def to_xyz(self, xyz_file):
        """
        Convert OTF trajectory from log file to .xyz file.
        Args:
            xyz_file (str): the file name of the .xyz file to output

        Return:
            A list of `ASE Atoms` objects.
        """
        from ase.io import write

        struc_trj = self.output_md_structures()
        write(xyz_file, struc_trj, format="extxyz")
        return struc_trj


def split_blocks(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        head = 0
        blocks = []
        for index, line in enumerate(lines):
            if "*-Frame" in line or line.startswith("---"):
                blocks.append(lines[head:index])
                head = index
    return blocks


def parse_header_information(lines) -> dict:
    """
    Get information about the run from the header of the file
    :param outfile:
    :return:
    """
    header_info = {}

    cutoffs_dict = {}
    for i, line in enumerate(lines):
        line_lower = line.lower()

        # gp related
        if "cutoffs" in line_lower:
            if "{" in line_lower:
                line = line[len("Cutoffs: ") :]
                cutoffs = eval(line)
            else:
                line = line.split(":")[1].strip()
                line = line.strip("[").strip("]")
                line = line.split()
                cutoffs = []
                for val in line:
                    try:
                        cutoffs.append(float(val))
                    except:
                        cutoffs.append(float(val[:-1]))
            header_info["cutoffs"] = cutoffs

        if "number of hyperparameters" in line_lower:
            n_hyps = int(line.split(":")[1].strip())
            header_info["n_hyps"] = n_hyps

            # parse hyps
            new_line = lines[i + 1].replace("[", "")
            new_line = new_line.replace("]", "")
            assert "hyperparameter" in new_line.lower()

            hyps_array = new_line[new_line.find(":") + 1 :].split()
            extra_line = 2
            while len(hyps_array) < n_hyps:
                next_new_line = lines[i + extra_line].replace("[", "")
                next_new_line = next_new_line.replace("]", "")
                hyps_array += next_new_line.split()
                extra_line += 1
            assert len(hyps_array) == n_hyps

            hyps_array = [float(h.strip()) for h in hyps_array]
            header_info["hyps"] = np.array(hyps_array)

        if "kernel_name" in line_lower:
            header_info["kernel_name"] = line.split(":")[1].strip()
        elif "kernels" in line_lower:
            line = line.split(":")[1].strip()
            line = line.strip("[").strip("]")
            line = line.split()
            header_info["kernels"] = line
        elif "kernel" in line_lower:
            header_info["kernel_name"] = line.split(":")[1].strip()

        for kw in header_dict:
            get_header_item(line, header_info, kw)

        if "optimization algorithm" in line:
            header_info["algo"] = str(line.split(":")[1].strip()).upper()

        if "system species" in line_lower:
            line = line.split(":")[1]
            line = line.split("'")
            species = [item for item in line if item.isalpha()]
            header_info["species_set"] = set(species)
        if "periodic cell" in line_lower:
            vectors = []
            for cell_line in lines[i + 1 : i + 4]:
                cell_line = cell_line.strip().replace("[", "").replace("]", "")
                vec = cell_line.split()
                vector = [float(vec[0]), float(vec[1]), float(vec[2])]
                vectors.append(vector)
            header_info["cell"] = np.array(vectors)
        if "previous positions" in line_lower:
            struc_spec = []
            prev_positions = []
            for pos_line in lines[i + 1 : i + 1 + header_info.get("atoms", 0)]:
                pos = pos_line.split()
                struc_spec.append(pos[0])
                prev_positions.append((float(pos[1]), float(pos[2]), float(pos[3])))
            header_info["species"] = struc_spec
            header_info["prev_positions"] = np.array(prev_positions)

    return header_info


def get_header_item(line, header_info, kw):
    if not isinstance(line, str):
        return

    pattern = header_dict[kw][0]
    value_type = header_dict[kw][1]

    if header_dict[kw][2]:
        pattern = pattern.lower()
        line = line.lower()

    if pattern in line:
        header_info[kw] = value_type(line.split(":")[1].strip())


header_dict = {
    "restart": ["Restart", int, False],
    "frames": ["Frames", int, True],
    "atoms": ["Number of atoms", int, True],
    "dt": ["Timestep", float, True],
}


def append_atom_lists(
    species_list: List[str],
    position_list: List[np.ndarray],
    force_list: List[np.ndarray],
    uncertainty_list: List[np.ndarray],
    velocity_list: List[np.ndarray],
    lines: List[str],
    index: int,
    noa: int,
    dft_call: bool,
    noh: int,
) -> None:
    """Update lists containing atom information at each snapshot."""

    if lines[0].startswith("---"):
        start_index = 4
    else:
        start_index = 3

    noa = 0
    for line in lines[start_index:]:
        if line.strip():
            noa += 1
        else:
            break

    species, positions, forces, uncertainties, velocities = parse_snapshot(
        lines, index, noa, dft_call, noh
    )

    species_list.append(species)
    position_list.append(positions)
    force_list.append(forces)
    uncertainty_list.append(uncertainties)
    velocity_list.append(velocities)


def parse_snapshot(lines, index, noa, dft_call, noh):
    """Parses snapshot of otf output file."""

    # initialize values
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

    return species, positions, forces, uncertainties, velocities


def strip_and_split(line):
    """
    Helper function which saves a few lines of code elsewhere
    :param line:
    :return:
    """

    line = line.strip().split()
    stripped_line = [subline.strip() for subline in line]

    return stripped_line


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


def extract_global_info(
    cell_list,
    stress_list,
    thermostat,
    block,
):

    for ind, line in enumerate(block):
        if "cell" in line:
            vectors = []
            for cell_line in block[ind + 1 : ind + 4]:
                cell_line = cell_line.strip().replace("[", "").replace("]", "")
                vec = cell_line.split()
                vector = [float(vec[0]), float(vec[1]), float(vec[2])]
                vectors.append(vector)
            cell_list.append(vectors)
        if "Stress" in line:
            vectors = []
            stress_line = block[ind + 2].replace("-", " -").split()
            vectors = [float(s) for s in stress_line]
            stress_list.append(vectors)

        for t in [
            "Pressure",
            "Temperature",
            "Kinetic energy",
            "Potential energy",
            "Total energy",
        ]:
            get_thermostat(thermostat, t, line)


def get_thermostat(thermostat, kw, line):
    kw = kw.lower()
    line = line.lower()
    if kw in line:
        try:
            value = float(line.split()[-2])  # old style
        except:
            value = float(line.split()[-1])  # new style
        if kw in thermostat:
            thermostat[kw].append(value)
        else:
            thermostat[kw] = [value]


def extract_gp_info(block, mae_list, maf_list, atoms_list, hyps_list, noh):
    """
    Exclusively parses DFT run information
    :param filename:
    :return:
    """
    for ind, line in enumerate(block):
        if "Mean absolute errors & Mean absolute values":
            efs_mae = np.zeros(3)
            efs_mav = np.zeros(3)
        if line.startswith("energy mae"):
            efs_mae[0] = float(line.split()[2])
        if line.startswith("energy mav"):
            efs_mav[0] = float(line.split()[2])
        if line.startswith("stress mae"):
            efs_mae[1] = float(line.split()[2])
        if line.startswith("stress mav"):
            efs_mav[1] = float(line.split()[2])
        if line.startswith("forces mae"):
            efs_mae[2] = float(line.split()[2])
            mae_list.append(efs_mae)
        if line.startswith("forces mav"):
            efs_mav[2] = float(line.split()[2])
            maf_list.append(efs_mav)

        # keep track of atom number
        if line.startswith("Adding atom"):
            atoms_added = []
            line_split = line.split()
            atom_strings = line_split[2:-4]
            for n, atom_string in enumerate(atom_strings):
                if n == 0:
                    atoms_added.append(int(atom_string[1:-1]))
                else:
                    atoms_added.append(int(atom_string[0:-1]))
            atoms_list.append(atoms_added)

        # keep track of hyperparameters
        if line.startswith("GP hyperparameters:"):
            hyps = []
            for hyp_line in block[(ind + 1) : (ind + 1 + noh)]:
                hyp_line = hyp_line.split()
                hyps.append(float(hyp_line[-1]))
            hyps = np.array(hyps)
            hyps_list.append(hyps)
