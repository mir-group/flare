import os, logging, re
from copy import deepcopy
import numpy as np
from subprocess import call

from ase.io import read, write
from ase.calculators.lammps import convert
from ase.calculators import lammpsrun
from ase.calculators.singlepoint import SinglePointCalculator
from flare.utils.element_coder import _Z_to_mass, element_to_Z

from . import presets, config


class LAMMPS:
    def __init__(
        self,
        atoms,
        specorder: list, # TODO: might not be necessary
        lmp_command: str, # TODO: might remove to use os.environ.get("lmp")
        ff_preset: str = "flare_pp",
        md_preset: str = "vanilla",
        timestep: float = 0.001,
        trajectory=None,
        md_dict: dict = {},
        uncertainty_style: str = "sgp",
    ):
        """
        Args:
            atoms (str): ASE Atoms, init structure
            specorder (list): Species order in LAMMPS
        """

        self.atoms = atoms
        self.curr_atoms = atoms
        self.specorder = specorder
        self.species = " ".join(self.specorder)
        self.lmp_command = lmp_command
        self.curr_step = 0
        self.curr_iter = 0
        self.above_tol = True
        self.curr_tol = None
        self.uncertainty_style = uncertainty_style
        self.md_dict = md_dict
        self.md_dict["timestep"] = timestep

        self.ff_preset_name = ff_preset
        self.md_preset_name = md_preset

        self.thermo_dict = {}

        # load the preset settings of lammps
        # TODO: add customized option
        if ff_preset == "mgp":
            self.ff_preset = presets.preset_mgp
        elif ff_preset == "flare_pp":
            self.ff_preset = presets.preset_flare_pp
        else:
            raise NotImplementedError

        if md_preset == "vanilla":
            self.md_preset = presets.vanilla_md
            self.md_dict = md_dict
        else:
            raise NotImplementedError

    def step(self, tol, N_steps, logger):
        # run lammps until the uncertainty interrupts
        N_iter = N_steps // self.md_dict["n_steps"]
        self.run(coeff_dir=".", N_iter=N_iter, tol=tol, logger=logger)

        # save trajectory into .xyz
        self.backup()

        self.curr_atoms = read(config.LMP_XYZ, index=-1)

    def run(self, coeff_dir, N_iter, tol=None, logger=None, **kwargs):
        """
        Write input and data files for LAMMPS, and then launch the simulation.
        In active learning, the simulation will be interrupted and exit once 
        the uncertainty is above the tolerance threshold, or the total number
        of steps is finished.

        Args:
            coeff_dir (str): directory of the coefficient file.
            N_iter (int): the number of iterations to run. Each iteration the 
                MD runs `n_steps`. Therefore, the total `rest_number_of_steps =
                n_steps * N_iter`.
            tol (float): the uncertainty threshold.
            logger (Logger): a logger from `logging` module to dump log messages.
        """
        # write lammps data file
        self.write_data(self.atoms)

        # write lammps input file
        ff_dict = {
            "coeff_dir": coeff_dir,
            "species": self.species,
            "uncertainty_style": self.uncertainty_style,
        }
        if tol is None:
            ff_dict["compute_unc"] = False

        # Start from the last frame of the previous trajectory
        md_dict = deepcopy(self.md_dict)
        if config.LMP_RESTART in os.listdir():
            ff_dict["read_restart"] = config.LMP_RESTART
            md_dict["read_restart"] = config.LMP_RESTART
        md_dict["N_iter"] = N_iter
        md_dict["tol"] = tol

        self.write_input(ff_dict, md_dict)

        logger.info(
            f"\nRunning LAMMPS command {self.lmp_command} < {config.LMP_IN} > {config.LMP_LOG}"
        )

        # run lammps simulation
        self.above_tol, self.curr_tol = self.run_lammps()
        if not self.above_tol:
            self.curr_iter = N_iter
            logger.info(f"before updating curr_step={self.curr_step}")
            self.curr_step += self.md_dict["n_steps"] * N_iter
        logger.info(f"This iteration has run {self.md_dict['n_steps'] * self.curr_iter} time steps")
        logger.info(f"In total, the MD has run {self.curr_step} time steps")

        # post procesessing
        self.post_process()

    def run_lammps(self):
        # call the executable
        with open(config.LMP_IN) as fin:
            with open(config.LMP_LOG, "w") as fout:
                call(self.lmp_command.split(), stdin=fin, stdout=fout)

        # read the lammps log file
        above_tol = False
        curr_tol = None
        with open(config.LMP_LOG) as fout:
            lmplog = fout.readlines()
            for line in lmplog:
                words = line.split()
                if "ERROR" in line:  # check if any error occurs
                    raise Exception("LAMMPS run got ERROR")
                if (
                    "has uncertainty above threshold" in line
                ):  # check if uncertainty above threshold
                    if (len(words) > 0) and (words[0] == "Iteration"):
                        self.curr_step += self.md_dict["n_steps"] * int(words[1])
                        self.curr_iter = int(words[1])
                        above_tol = True
                        curr_tol = float(words[-1])
                        break
                if line.startswith("Absolute tolerance"):
                    try:
                        curr_tol = float(words[2])
                    except ValueError:
                        continue

        return above_tol, curr_tol

    def backup(self):
        """
        Backup trajectory to .xyz file. Include the 1st frame
        """
        trj = read(
            config.LMP_TRJ,
            format="lammps-dump-text",
            specorder=self.specorder,
            index=":",
        )

        steps = []
        with open(config.LMP_TRJ) as f:
            dump_file = f.readlines()
            for l, line in enumerate(dump_file):
                if "TIMESTEP" in line:
                    steps.append(int(dump_file[l + 1]))

        assert len(steps) == len(trj)
        start_index = len(self.thermo_dict["step"]) - len(trj)
        for i, frame in enumerate(trj):
            frame.pbc = True
            stress = np.array(
                [
                    -self.thermo_dict[p][start_index + i]
                    for p in ["pxx", "pyy", "pzz", "pyz", "pxz", "pxy"]
                ]
            )
            frame.calc.results["stress"] = convert(stress, "pressure", "metal", "ASE")
            frame.calc.results["energy"] = self.thermo_dict["pe"][start_index + i]
            assert self.thermo_dict["step"][start_index + i] == steps[i], (self.thermo_dict["step"][start_index + i], steps[i])
            frame.info["step"] = steps[i]
        write(config.LMP_XYZ, trj, append=True, format="extxyz")

    def write_data(self, atoms):
        """
        Write lammps data file for input, used as the initial configuration of MD.
        """
        write(
            config.LMP_DAT,
            atoms,
            format="lammps-data",
            specorder=self.specorder,
            force_skew=True,
        )

        # add masses
        with open(config.LMP_DAT, "a") as f:
            f.write("\n\nMasses\n\n")
            for s, spc in enumerate(self.specorder):
                mass = _Z_to_mass[element_to_Z(spc)]
                f.write(f"{s+1} {mass}\n")

    def write_input(self, ff_dict, md_dict):
        """
        Write lammps input file, using the preset settings.
        """
        ff_command = self.ff_preset(**ff_dict)
        md_command = self.md_preset(**md_dict)

        # write to file
        with open(config.LMP_IN, "w") as f:
            f.write(ff_command)
            f.write(md_command)

    def post_process(self):
        """
        Post-processing of lammps trajectory, plotting energy, temperature, etc.
        """
        self.thermo_dict = process_thermo_txt(config.LMP_IN, config.LMP_THERMO)

    def todict(self):
        var_dict = dict(vars(self))
        var_dict.pop("atoms")

        new_dict = deepcopy(var_dict)
        new_dict.pop("ff_preset")
        new_dict.pop("md_preset")
        if self.atoms is not None:
            new_dict["atoms"] = self.atoms.todict()
        new_dict["curr_atoms"] = self.curr_atoms.todict()
        return new_dict
    

################################################################################
#                                                                              #
#                               Util Functions                                 #
#                                                                              #
################################################################################

def check_sgp_match(atoms, sgp_calc, logger):
    """
    Check if the lammps trajectory or calculator matches the SGP predictions
    """
    # if atoms are from a lammps trajectory, then directly use the 
    # energy/forces/stress/stds from the trajectory to compare with sgp
    assert isinstance(atoms.calc, SinglePointCalculator), type(atoms.calc)
    lmp_energy = atoms.potential_energy
    lmp_forces = atoms.forces
    lmp_stress = atoms.stress
    lmp_stds = atoms.get_array("c_unc")

    # subtract the pressure from temperature: sum(m_k * v_ki * v_kj)/ V
    velocities = atoms.get_velocities()
    masses = atoms.get_masses()
    kinetic_stress = np.zeros(6)
    n = 0
    for i in range(3):
        for j in range(i, 3):
            kinetic_stress[n] += np.sum(masses * velocities[:, i] * velocities[:, j])
            n += 1
    kinetic_stress /= atoms.get_volume()
    kinetic_stress = kinetic_stress[[0, 3, 5, 4, 2, 1]]
    lmp_stress += kinetic_stress

    # compute sgp energy/forces/stress/stds
    sgp_calc.reset()
    atoms.calc = sgp_calc
    gp_energy = atoms.get_potential_energy()
    gp_forces = atoms.get_forces()
    gp_stress = atoms.get_stress()
    gp_stds = atoms.calc.results["stds"]

    # compute the difference and print to log file
    de = np.abs(lmp_energy - gp_energy)
    df = np.max(np.abs(lmp_forces - gp_forces))
    ds = np.max(np.abs(lmp_stress - gp_stress))
    du = np.max(np.abs(lmp_stds - gp_stds[:, 0]))
    logger.info("LAMMPS and SGP maximal absolute difference in prediction:")
    logger.info(f"Maximal absolute energy difference: {de}")
    logger.info(f"Maximal absolute forces difference: {df}")
    logger.info(f"Maximal absolute stress difference: {ds}")
    logger.info(f"Maximal absolute uncertainty difference: {du}")

    try:
        # allow a small discrepancy because the position loses precision during dumping
        assert np.allclose(lmp_energy, gp_energy, atol=1e-3)
        assert np.allclose(lmp_forces, gp_forces, atol=1e-3)
        assert np.allclose(lmp_stress, gp_stress, atol=1e-4)
        assert np.allclose(lmp_stds, gp_stds[0], atol=1e-4)  
    except:
        # if the trajectory does not match sgp, this is probably because the dumped
        # atomic positions in LAMMPS lose precision. Then build a new lammps calc
        # and do a static calculation and compare to SGP
        lmp_calc = get_ase_lmp_calc(
            ff_preset="flare_pp",
            specorder=["Si", "C"],
            coeff_dir=".",
            lmp_command=os.environ.get("lmp"),
        )
        atoms.calc = lmp_calc
        lmp_energy = atoms.get_potential_energy()
        lmp_forces = atoms.get_forces()
        lmp_stress = atoms.get_stress()

        assert np.allclose(lmp_energy, gp_energy)
        assert np.allclose(lmp_forces, gp_forces)
        assert np.allclose(lmp_stress, gp_stress)
        atoms.calc = sgp_calc


def get_ase_lmp_calc(
    ff_preset: str, specorder: list, coeff_dir: str, lmp_command: str = None,
    tmp_dir: str = "./tmp/",
):
    """
    Set up ASE calculator of lammps
    Args:
        ff_preset (str): force field preset, either "mgp" or "flare_pp".
        specorder (list): a list of (unique) species, for example, ["Si", "C"].
        coeff_dir (str): the directory of the coefficient file for LAMMPS.
    """
    if lmp_command is None:
        lmp_command = os.environ.get("lmp")

    if ff_preset == "mgp":
        species_str = " ".join(specorder)
        pot_file = os.path.join(coeff_dir, config.COEFF + ".mgp")
        params = {
            "command": lmp_command,
            "pair_style": "mgp",
            "pair_coeff": [f"* * {pot_file} {species_str} yes yes"],
        }
    elif ff_preset == "flare_pp":
        pot_file = os.path.join(coeff_dir, config.COEFF + ".flare")
        params = {
            "command": lmp_command,
            "pair_style": "flare",
            "pair_coeff": [f"* * {pot_file}"],
        }
    else:
        raise NotImplementedError(f"{ff_preset} preset is not implemented")

    files = [pot_file]

    # create ASE calc
    calc = lammpsrun.LAMMPS(
        label=f"tmp",
        keep_tmp_files=True,
        tmp_dir=tmp_dir,
        parameters=params,
        files=files,
        specorder=specorder,
    )
    return calc



def special_match(strg, search):
    return not bool(search(strg))


def process_log(log_name="log.lammps"):
    """
    Plot thermo info including temperature, ke, pe of the LAMMPS trajectory
    """
    with open(log_name) as f:
        lines = f.readlines()

    # extract blocks of thermal info
    keys = []
    data = []
    search = re.compile(r"[^0-9.-]").search
    for ind, line in enumerate(lines):
        # Get all the thermo quantities dumped
        if ("Step" in line) and (not keys):
            keys = line.split()
            thermo = {k: [] for k in keys}
        values = line.split()

        # Get lines with all numbers as values
        if keys and (len(values) == len(keys)):
            all_match = True
            for v in values:
                if not special_match(v, search):
                    all_match = False
                    break
            # Store the values in the thermo dict
            if all_match:
                step = eval(values[0])
                if step not in thermo["Step"]:
                    for k in range(len(keys)):
                        thermo[keys[k]].append(eval(values[k]))

    thermo = {k: np.array(thermo[k]) for k in keys}
    logging.info(config.LOG.format("Temperature: start", thermo["Temp"][0]))
    logging.info(config.LOG.format("Temperature: final", thermo["Temp"][-1]))
    # plot_thermostat(thermo)
    return thermo


def process_thermo_txt(in_file="in.lammps", thermo_file="thermo.txt"):
    # find all the dumped quantities from the input file
    input_lines = open(in_file).readlines()
    properties = []
    for line in input_lines:
        if "fix" in line and "print" in line:
            prop_str = line.split('"')[1].replace("$", "")
            prop_str = prop_str.replace("(", "").replace(")", "")
            properties = prop_str.split()
            break

    # find the last "#" line to be the start of the current trajectory
    with open(thermo_file) as f:
        lines = f.readlines()
        curr_traj_len = 0
        for l in range(len(lines)):
            if "#" in lines[len(lines) - l - 1]:
                curr_traj_len = l
                break

    thermo_data = np.loadtxt(thermo_file)
    if len(thermo_data.shape) == 1: # if there is only one line, numpy reads it as an array
        thermo_data = np.array([thermo_data])
    else:
        traj_start = thermo_data.shape[0] - curr_traj_len
        thermo_data = thermo_data[traj_start:, :]

    assert len(properties) == thermo_data.shape[1]
    thermo = {prop: [] for prop in properties}

    assert "step" in properties
    step_ind = properties.index("step")

    # remove duplicated steps
    new_thermo_data = []
    for s in range(thermo_data.shape[0]):
        if not new_thermo_data:
            new_thermo_data.append(thermo_data[s])
        elif thermo_data[s][step_ind] != new_thermo_data[-1][step_ind]:
            new_thermo_data.append(thermo_data[s])
    thermo_data = np.array(new_thermo_data)
    thermo = {prop: thermo_data[:, p] for p, prop in enumerate(properties)}
    return thermo
