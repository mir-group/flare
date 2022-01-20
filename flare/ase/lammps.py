from math import ceil
import numpy as np
import glob, os
import warnings
from datetime import datetime
from copy import deepcopy
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.lammps import convert
from ase.md.md import MolecularDynamics
from ase.io import read, write


class LAMMPS_MOD(LAMMPS):
    """
    A modified ASE LAMMPS calculator based on ase.lammpsrun.LAMMPS,
    to allow for more flexible input parameters, including compute,
    fix/nvt, fix/npt etc.

    Supported customized commands for LAMMPS input:
    - mass (set by arg `masses`)
    - package
    - atom_style, bond_style, angle_style, dihedral_style, improper_style, kspace_style
    - units (default: metal)
    - boundary
    - neighbor
    - newton
    - kim_interactions
    - pair_style (default: lj/cut 2.5)
    - pair_coeff (default: * * 1 1)
    - *compute
    - group
    - fix
    - timestep
    - minimize
    - run

    Note:
    - additional commands needed at the beginning can be specified in the arg `model_init`
    - additional commands needed after "pair_coeff" can be specified in the arg `model_post`

    Non-customized input commands:
    - atom_modify sort 0 0.0
    - read_data
    - fix fix_nve all nve
    - dump dump_all all custom trj_file (dump_period) id type x y z vx vy vz fx fy fz
    - thermo_style custom (thermo_args)
    - thermo_modify flush yes format float %23.16g
    - thermo 1

    Customized parameters:
    - dump_period
    - thermo_args
    - specorder
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nsteps = 0
        if not os.environ.get("ASE_LAMMPSRUN_COMMAND"):
            raise Exception(
                "Please set environment variable 'ASE_LAMMPSRUN_COMMAND' to the lammps executable"
            )

    def calculate(
        self, atoms=None, properties=None, system_changes=None, set_atoms=False
    ):
        """Modify parameters"""

        self.parameters.setdefault("units", "metal")
        self.parameters.setdefault("model_post", [])
        self.parameters.setdefault(
            "timestep", str(DEFAULT_TIMESTEP[self.parameters["units"]])
        )

        # Add "compute" command after "pair_coeff", using `model_post`
        if "compute" in self.parameters:
            compute_command = ""
            for cmd in self.parameters["compute"]:
                compute_command += "compute " + cmd + "\n"
            self.parameters["model_post"] += compute_command

        # Always unfix "nve" defined in ASE
        if "fix" in self.parameters:
            self.parameters["fix"][-1] += "\nunfix fix_nve"

        if properties is None:
            properties = self.implemented_properties
        if system_changes is None:
            system_changes = all_changes
        Calculator.calculate(self, atoms, properties, system_changes)

        self.run(set_atoms)

        # Keep only the virial stress, remove the kinetic stress
        self.subtract_kinetic_stress()

    def subtract_kinetic_stress(self):
        # LAMMPS stress tensor = virial + kinetic
        # kinetic = sum(m_k * v_ki * v_kj) / V
        # We subtract the kinetic term and keep only the virial term
        velocities = self.atoms.get_velocities()
        masses = self.atoms.get_masses()
        volume = self.atoms.get_volume()
        kinetic = velocities.T @ np.diag(masses) @ velocities / volume

        # apply the Lammps rotation stuff to the stress (copied from lammpsrun.py)
        R = self.prism.rot_mat
        kinetic_atoms = np.dot(R, kinetic)
        kinetic_atoms = np.dot(kinetic_atoms, R.T)
        kinetic_atoms = kinetic_atoms[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
        self.results["stress"] += kinetic_atoms


class LAMMPS_BAL(MolecularDynamics):
    """
    Run MD with LAMMPS based on the ase.md.md.MolecularDynamics.
    It includes using LAMMPS_MOD to run multiple steps, and supports
    Bayesian active learning with flare.

    Args:
        parameters (dict): LAMMPS input commands.

    """

    def __init__(self, atoms, timestep, parameters, **kwargs):
        self.thermo_file = "thermo.txt"
        self.traj_xyz_file = "traj.xyz"
        self.potential_file = "lmp.flare"
        self.dump_cols = [
            "id",
            "type",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "fx",
            "fy",
            "fz",
            "c_unc",
        ]
        self.initial_parameters = deepcopy(parameters)

        # Set up ASE calculator parameters
        if "specorder" not in parameters:  # user must provide the unique species
            raise ValueError("Please set 'specorder' to a list of unique species")
        parameters.setdefault("write_velocities", True)

        # Set up default LAMMPS input commands if not given
        parameters.setdefault("units", "metal")
        parameters.setdefault("timestep", str(DEFAULT_TIMESTEP[parameters["units"]]))
        parameters.setdefault("dump_period", 1)
        self.parameters = parameters

        self.initial_parameters = deepcopy(parameters)

        super().__init__(atoms, timestep, **kwargs)
        self.curr_atoms = self.atoms.copy()

    def set_bayesian_active_learning_parameters(self):
        self.parameters.setdefault("pair_style", "mgp")
        self.parameters.setdefault("compute", [])

        # The flare uncertainty command should not be set by user
        for cmd in self.parameters["compute"]:
            if ("uncertainty/atom" in cmd) or ("flare/std/atom" in cmd):
                raise ValueError(f"Please remove command '{cmd}'")

        # Set up default potential and uncertainty command for mgp
        if self.parameters["pair_style"] == "mgp":
            self.parameters["newton"] = "off"
            species_str = " ".join(self.parameters["specorder"])
            self.parameters["pair_coeff"] = [
                f"* * {self.potential_file} {species_str} yes yes"
            ]

            self.uncertainty_file = "lmp.flare.std"
            self.parameters["compute"] += [
                f"unc all uncertainty/atom {self.uncertainty_file} {species_str} yes yes",
                "MaxUnc all reduce max c_unc",
            ]

        # Set up default potential and uncertainty command for flare
        elif self.parameters["pair_style"] == "flare":
            self.parameters["newton"] = "on"
            self.parameters["pair_coeff"] = [f"* * {self.potential_file}"]

            self.uncertainty_file = "L_inv_lmp.flare sparse_desc_lmp.flare"
            self.parameters["compute"] += [
                f"unc all flare/std/atom {self.uncertainty_file}",
                "MaxUnc all reduce max c_unc",
            ]

        else:
            raise ValueError("pair_style can only be 'mgp' or 'flare'")

    def step(self, std_tolerance, N_steps):
        """
        Run lammps until the uncertainty interrupts. Notice this method neither runs
        only a single MD step, nor finishes all the ``N_steps``. The MD exits only when
        1) the maximal atomic uncertainty goes beyond ``std_tolerance``, or
        2) all the ``N_steps`` are finished without uncertainty beyond ``std_tolerance``.

        Args:
            std_tolerance (float): the threshold for atomic uncertainty, above which the
                MD will be interrupted and DFT will be called.
            N_steps (int): number of MD steps left to run.
        """

        # Create a modified LAMMPS calculator,
        # assign a unique label by datetime to this calculator
        now = datetime.now()
        label = now.strftime("%Y.%m.%d:%H:%M:%S:")

        # Set up pair_style and compute commands
        self.parameters = deepcopy(self.initial_parameters)
        self.set_bayesian_active_learning_parameters()

        # Get lammps commands for running Bayesian MD to monitor uncertainty
        # Append the bayesian command after the "timestep" command
        self.parameters["timestep"] = str(self.parameters["timestep"])
        self.parameters["timestep"] += BAL_RUN_CMD.format(
            dump_freq=self.parameters["dump_period"],
            label=label,
            curr_steps=self.nsteps,
            N_steps=N_steps,
            std_tolerance=std_tolerance,
            thermo_file=self.thermo_file,
            dump_cols=" ".join(self.dump_cols),
        )

        # Run lammps with the customized parameters
        lmp_calc = LAMMPS_MOD(
            label=label,
            files=[self.potential_file] + self.uncertainty_file.split(),
            keep_tmp_files=True,
            tmp_dir="tmp",
        )
        lmp_calc.set(**self.parameters)
        atoms = deepcopy(self.curr_atoms)
        lmp_calc.calculate(atoms, set_atoms=True)

        # Read the trajectory after the current run exits
        trj = read(
            glob.glob(f"tmp/trjunc_{label}*")[0],
            format="lammps-dump-binary",
            specorder=self.parameters["specorder"],
            colnames=self.dump_cols,
            index=":",
        )

        # Update the time steps that have been run
        trj_len = len(trj)
        self.nsteps += self.parameters["dump_period"] * (trj_len - 1)

        # Get the last frame as the current structure
        self.curr_atoms = trj[-1]

        # Check the atoms are the same as the last frame of the trajectory
        assert np.allclose(trj[-1].get_forces(), lmp_calc.results["forces"])
        assert np.allclose(trj[-1].get_volume(), lmp_calc.atoms.get_volume())
        pos1 = trj[-1].get_positions(wrap=True)
        pos2 = lmp_calc.atoms.get_positions(wrap=True)
        pos_diff = (pos1 - pos2) @ np.linalg.inv(lmp_calc.atoms.cell)
        for i in np.reshape(pos_diff.round(6), -1):
            assert i.is_integer()

        # Back up the trajectory into the .xyz file
        self.backup(trj)

    def backup(self, curr_trj):
        """
        Back up the current trajectory into .xyz file. The atomic positions,
        velocities, forces and uncertainties are read from lammps trajectory.
        The step, potential energy and stress are read from thermo.txt

        Args:
            curr_trj (list[ase.Atoms]): lammps trajectory of current run read
                by ASE.
        """

        thermostat = np.loadtxt(self.thermo_file)
        with open(self.thermo_file) as f:
            n_iters = f.read().count("#")

        if self.traj_xyz_file in os.listdir():
            previous_trj = read(self.traj_xyz_file, index=":")
            assert (
                thermostat.shape[0]
                == 2 * (len(previous_trj) + len(curr_trj)) - 2 * n_iters
            )
        else:
            assert thermostat.shape[0] == 2 * len(curr_trj) - 2 * n_iters

        # Extract energy, stress and step from dumped log file and write to
        # the frames in .xyz
        curr_thermo = thermostat[-len(curr_trj) :]
        for i, frame in enumerate(curr_trj):
            frame.pbc = True
            frame.info["step"] = curr_thermo[i][0]
            frame.calc.results["energy"] = curr_thermo[i][3]

            stress = -curr_thermo[i][5:11]  # pxx pyy pzz pyz pxz pxy
            frame.calc.results["stress"] = convert(
                stress, "pressure", self.parameters["units"], "ASE"
            )

        write(self.traj_xyz_file, curr_trj, append=True, format="extxyz")

    def todict(self):
        dct = super().todict()
        dct["parameters"] = deepcopy(self.initial_parameters)
        dct["nsteps"] = self.nsteps
        return dct


BAL_RUN_CMD = """
fix thermoprint all print {dump_freq} "$(step) $(temp) $(ke) $(pe) $(etotal) $(pxx) $(pyy) $(pzz) $(pyz) $(pxz) $(pxy) $(c_MaxUnc)" append {thermo_file}
dump dump_unc all custom {dump_freq} tmp/trjunc_{label}.bin {dump_cols} 
reset_timestep {curr_steps}
run {N_steps} upto every {dump_freq} "if '$(c_MaxUnc) > {std_tolerance}' then quit"
unfix thermoprint
"""

DEFAULT_TIMESTEP = {
    "lj": 0.005,
    "real": 1.0,
    "metal": 0.001,
    "si": 1e-8,
    "cgs": 1e-8,
    "electron": 0.001,
    "micro": 2.0,
    "nano": 0.00045,
}


################################################################################
#                                                                              #
#                               Util Functions                                 #
#                                                                              #
################################################################################


def check_sgp_match(atoms, sgp_calc, logger, specorder):
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
            specorder=specorder,
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
    ff_preset: str,
    specorder: list,
    coeff_dir: str,
    lmp_command: str = None,
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
        pot_file = os.path.join(coeff_dir, "lmp.flare")
        params = {
            "command": lmp_command,
            "pair_style": "mgp",
            "pair_coeff": [f"* * {pot_file} {species_str} yes yes"],
        }
    elif ff_preset == "flare_pp":
        pot_file = os.path.join(coeff_dir, "lmp.flare")
        params = {
            "command": lmp_command,
            "pair_style": "flare",
            "pair_coeff": [f"* * {pot_file}"],
        }
    else:
        raise NotImplementedError(f"{ff_preset} preset is not implemented")

    files = [pot_file]

    # create ASE calc
    calc = LAMMPS(
        label=f"tmp",
        keep_tmp_files=True,
        tmp_dir=tmp_dir,
        parameters=params,
        files=files,
        specorder=specorder,
    )
    return calc
