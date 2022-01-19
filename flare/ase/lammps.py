from math import ceil
import glob
import warnings
from datetime import datetime
from copy import deepcopy
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.lammps import convert
from ase.md.md import MolecularDynamics


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
    - *velocity
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

    def calculate(
        self, atoms=None, properties=None, system_changes=None, set_atoms=False
    ):
        """Modify parameters"""

        self.parameters.setdefault("model_post", [])

        # Add "compute" command after "pair_coeff", using `model_post`
        if "compute" in self.parameters:
            compute_command = ""
            for cmd in self.parameters["compute"]:
                compute_command += "compute " + cmd + "\n"
            self.parameters["model_post"] += compute_command

        # Add "velocity" command after "pair_coeff", using `model_post`
        if "velocity" in self.parameters:
            velocity_command = ""
            for cmd in self.parameters["velocity"]:
                velocity_command += "velocity" + cmd + "\n"
            self.parameters["model_post"] += velocity_command

        # Always unfix "nve" defined in ASE
        if "fix" in self.parameters:
            self.parameters["fix"][-1] += "\nunfix fix_nve"

        if properties is None:
            properties = self.implemented_properties
        if system_changes is None:
            system_changes = all_changes
        Calculator.calculate(self, atoms, properties, system_changes)
        self.run(set_atoms)


class LAMMPS_BAL(MolecularDynamics):
    """
    Run MD with LAMMPS based on the ase.md.md.MolecularDynamics
    for Bayesian active learning

    Args:
        parameters (dict): LAMMPS input commands.

    """

    def __init__(self, atoms, timestep, parameters, **kwargs):
        super().__init__(atoms, timestep, **kwargs)

        # Set up ASE calculator parameters
        if "specorder" not in parameters:  # user must provide the unique species
            raise ValueError("Please set 'specorder' to a list of unique species")
        parameters.setdefault("write_velocities", True)

        # Set up default LAMMPS input commands if not given
        parameters.setdefault("units", "metal")
        parameters.setdefault("timestep", str(DEFAULT_TIMESTEP[parameters["units"]]))
        parameters.setdefault("dump_period", 1)
        parameters.setdefault("pair_style", "mgp")
        parameters.setdefault("compute", [])

        # The flare uncertainty command should not be set by user
        for cmd in parameters["compute"]:
            if ("uncertainty/atom" in cmd) or ("flare/std/atom" in cmd):
                raise ValueError(f"Please remove command '{cmd}'")

        # Set up default potential and uncertainty command for mgp
        self.potential_file = "lmp.flare"
        if parameters["pair_style"] == "mgp":
            species_str = " ".join(parameters["specorder"])
            parameters["pair_coeff"] = [
                f"* * {self.potential_file} {species_str} yes yes"
            ]

            self.uncertainty_file = "lmp.flare.std"
            parameters["compute"] += [
                "unc all uncertainty/atom {self.uncertainty_file} {species_str} yes yes"
            ]

        # Set up default potential and uncertainty command for flare
        elif parameters["pair_style"] == "flare":
            parameters["pair_coeff"] = ["* * {self.potential_file}"]

            self.uncertainty_file = "L_inv.flare sparse_desc.flare"
            parameters["compute"] += ["unc all flare/std/atom {self.uncertainty_file}"]

        else:
            raise ValueError("pair_style can only be 'mgp' or 'flare'")

        self.parameters = parameters

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

        lmp_calc = LAMMPS_MOD(
            label=label,
            files=[self.potential_file] + self.uncertainty_file.split(),
            keep_tmp_files=True,
            tmp_dir="tmp",
        )
        lmp_calc.set(**self.parameters)

        # Get lammps commands for running Bayesian MD to monitor uncertainty
        dump_freq = self.parameters["dump_period"]
        N_iter = ceil(N_steps / dump_freq)

        bayesian_run_command = BAL_RUN_CMD.format(
            std_tolerance=std_tolerance,
            dump_freq=dump_freq,
            N_iter=N_iter,
            thermo_file=THERMO_TXT,
        )

        # Append the bayesian command after the "timestep" command
        self.parameters["timestep"] += bayesian_run_command

        # Run lammps with the customized parameters
        atoms = deepcopy(self.atoms)
        lmp_calc.calculate(self.atoms, set_atoms=True)
        
        # Read the trajectory after the current run exits
        trj = read(
            glob.glob(f"tmp/trj_{label}*")[0],
            format="lammps-dump-binary", 
            specorder=self.parameters["specorder"],
            index=":",
        )
        trj_len = len(trj)

        # Check the atoms are the same as the last frame of the trajectory
        assert np.allclose(trj[-1].positions, self.atoms.positions)
        assert np.allclose(trj[-1].get_forces(), self.atoms.get_forces())
        assert np.allclose(trj[-1].cell, self.atoms.cell)

        # Back up the trajectory into the .xyz file
        self.backup(trj, TRAJ_XYZ, THERMO_TXT)

        self.curr_atoms = self.atoms
        self.atoms = atoms

        self.nsteps += dump_freq * (trj_len - 1)


    def backup(self, curr_trj, trj_file, thermo_file):
        """
        Back up the current trajectory into .xyz file. The atomic positions,
        velocities, forces and uncertainties are read from lammps trajectory.
        The step, potential energy and stress are read from thermo.txt

        Args:
            curr_trj (list[ase.Atoms]): lammps trajectory of current run read
                by ASE.
            trj_file (str): the file name of the .xyz file to write to.
            thermo_file (str): the file name of the thermostat file which has
                global properties dumped from lammps.
        """

        thermostat = np.loadtxt(thermo_file)
        previous_trj = read(trj_file, index=":")
        assert thermostat.shape[0] == len(previous_trj) + len(curr_trj)

        # Extract energy, stress and step from dumped log file and write to 
        # the frames in .xyz
        curr_thermo = thermostat[-len(curr_trj):]
        for i, frame in enumerate(curr_trj):
            frame.pbc = True
            frame.info["step"] = curr_thermo[i][0]
            frame.calc.results["energy"] = curr_thermo[i][3]

            stress = - curr_thermo[i][5:11] # pxx pyy pzz pyz pxz pxy
            frame.calc.results["stress"] = convert(
                stress, "pressure", self.parameters["units"], "ASE"
            )

        write(TRAJ_XYZ, curr_trj, append=True, format="extxyz")

        

THERMO_TXT = "thermo.txt"
TRAJ_XYZ = "traj.xyz"

BAL_RUN_CMD = """
fix thermoprint all print {dump_freq} "$(step) $(temp) $(ke) $(pe) $(etotal) $(pxx) $(pyy) $(pzz) $(pyz) $(pxz) $(pxy) $(c_MaxUnc)" append {thermo_file}
variable abstol equal {std_tolerance}
variable UncMax equal c_2 
variable a loop {N_iter}
label loopa
    run {dump_freq}
    if "${{UncMax}} > ${{abstol}}" then &
        "print 'Iteration $a has uncertainty above threshold ${{abstol}}'" &
        "jump SELF break"
    next a
jump SELF loopa
label break

write_restart restart.dat  # write to restart file for the next run
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

if __name__ == "__main__":
    import os
    from ase import Atom, Atoms
    from ase.build import bulk

    Ni = bulk("Ni", cubic=True)
    H = Atom("H", position=Ni.cell.diagonal() / 2)
    NiH = Ni + H

    os.environ[
        "ASE_LAMMPSRUN_COMMAND"
    ] = "/n/home08/xiey/lammps-stable_29Oct2020/src/lmp_mpi"
    files = ["NiAlH_jea.eam.alloy"]
    lammps = LAMMPS(files=files, keep_tmp_files=True, tmp_dir="tmp")
    lammps.set(
        pair_style="eam/alloy",
        pair_coeff=["* * NiAlH_jea.eam.alloy H Ni"],
        compute=["1 all pair/local dist", "2 all reduce max c_1"],
        velocity=["1 all parameters"],
        fix=[
            "1 all nvt temp 300 300 $(100.0*dt)",
        ],
        dump_period=dump_freq,
        timestep=f"{timestep}{otf_run_command}",
    )
    NiH.calc = lammps
    print("Energy ", NiH.get_potential_energy())
