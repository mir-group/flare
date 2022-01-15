from ase.calculators.lammpsrun import LAMMPS
from copy import deepcopy

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

    def __init__(**kwargs):
        super().__init__(**kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=None, set_atoms=False):
        """Modify parameters"""

        if "model_post" not in self.parameters:
            self.parameters["model_post"] = []

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

        # TODO: check if the per-atom compute result is read into atoms
        # TODO: we can not read global compute result though


class LAMMPS_BAL(LAMMPS_MOD):
    """
    LAMMPS for Bayesian active learning
    """

    def __init__(**kwargs):
        super().__init__(**kwargs)

    def step(self, std_tolerance, N_steps):
        """Run lammps until the uncertainty interrupts"""
        # TODO: compute commands for uncertainties
        # TODO: convert restart.dat into a data file and keep the group, velocity etc. info 
        # read the data file into the current atoms, and add the group, velocity info to
        # parameters dict, if it is not written by ase

        # Get the commands for running Bayesian MD
        dump_freq = self.parameters["dump_period"]
        N_iter = N_steps // dump_freq 

        bayesian_run_command = BAL_RUN_CMD.format(
            std_tolerance=std_tolerance,
            dump_freq=dump_freq,
            N_iter=N_iter,
        )

        # Append the bayesian command after the "timestep" command
        if "units" not in self.parameters:
            self.parameters["units"] = "metal"
        lmp_units = self.parameters["units"]

        if "timestep" not in self.parameters:
            self.parameters["timestep"] = str(DEFAULT_TIMESTEP[lmp_units])
        self.parameters["timestep"] += bayesian_run_command
        
        # Run lammps with the customized parameters
        atoms = deepcopy(self.atoms)
        self.calculate(self.atoms, set_atoms=True)

        self.backup()

        self.curr_atoms = self.atoms
        self.atoms = atoms

    def backup(self):
        pass


BAL_RUN_CMD = """
fix thermoprint all print {dump_freq} "$(step) $(temp) $(ke) $(pe) $(etotal) $(pxx) $(pyy) $(pzz) $(pyz) $(pxz) $(pxy) $(c_MaxUnc)" append thermo.txt
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
    
    Ni = bulk('Ni', cubic=True)
    H = Atom('H', position=Ni.cell.diagonal()/2)
    NiH = Ni + H
    
    os.environ["ASE_LAMMPSRUN_COMMAND"] = "/n/home08/xiey/lammps-stable_29Oct2020/src/lmp_mpi"
    files = ['NiAlH_jea.eam.alloy']
    lammps = LAMMPS(files=files, keep_tmp_files=True, tmp_dir="tmp")
    lammps.set(
        pair_style="eam/alloy",
        pair_coeff=["* * NiAlH_jea.eam.alloy H Ni"],
        compute=["1 all pair/local dist", "2 all reduce max c_1"],
        velocity=["1 all parameters"],
        fix=[
            '1 all nvt temp 300 300 $(100.0*dt)',
        ],
        dump_period=dump_freq,
        timestep=f"{timestep}{otf_run_command}",
    )
    NiH.calc = lammps
    print("Energy ", NiH.get_potential_energy())
    
    
