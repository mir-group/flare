import os, logging, re
from copy import deepcopy
import numpy as np
from subprocess import call

from ase.io import read, write
from ase.calculators.lammps import convert
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
        for i, frame in enumerate(trj):
            frame.pbc = True
            stress = np.array(
                [
                    -self.thermo_dict[p][i]
                    for p in ["pxx", "pyy", "pzz", "pyz", "pxz", "pxy"]
                ]
            )
            frame.calc.results["stress"] = convert(stress, "pressure", "metal", "ASE")
            frame.calc.results["energy"] = self.thermo_dict["pe"][i]
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
#                                Functions                                     #
#                                                                              #
################################################################################


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
    thermo_data = np.loadtxt(thermo_file)
    if len(thermo_data.shape) == 1: # if there is only one line, numpy reads it as an array
        thermo_data = np.array([thermo_data])
    assert len(properties) == thermo_data.shape[1]
    thermo = {prop: [] for prop in properties}

    assert "step" in properties
    # NOT filter out replicated steps
    #    p_step = properties.index("step")
    #
    #    # filter out those replicated steps
    #    for i in range(thermo_data.shape[0]):
    #        line = thermo_data[i]
    #        step = line[p_step]
    #        if step not in thermo["step"]:
    #            for p, prop in enumerate(properties):
    #                thermo[prop].append(line[p])
    #    thermo = {k: np.array(v) for k, v in thermo.items()}
    thermo = {prop: thermo_data[:, p] for p, prop in enumerate(properties)}
    return thermo
