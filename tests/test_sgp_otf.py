import time, os, shutil, glob, subprocess
from copy import deepcopy
import pytest
flare_pp = pytest.importorskip("flare_pp")
import numpy as np

from flare import otf, kernels
from flare.io.otf_parser import OtfAnalysis
from flare.gp import GaussianProcess
from flare.bffs.mgp import MappedGaussianProcess
from flare.bffs.gp.calculator import FLARE_Calculator
from flare.ase.otf import ASE_OTF
from flare.utils.parameter_helper import ParameterHelper

from flare_pp._C_flare import NormalizedDotProduct, B2, SparseGP, Structure
from flare_pp.sparse_gp import SGP_Wrapper
from flare_pp.sparse_gp_calculator import SGP_Calculator

from ase.data import atomic_numbers, atomic_masses
from ase.constraints import FixAtoms
from ase import units
from ase.spacegroup import crystal
from ase import io
from ase.calculators import lammpsrun
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

np.random.seed(12345)

md_list = ["PyLAMMPS"]
#md_list = ["VelocityVerlet"]
number_of_steps = 30

@pytest.fixture(scope="module")
def super_cell():
    atoms = io.read("test_files/4H_tersoff_relaxed.xyz") * np.array([2, 2, 1])

    # jitter positions to give nonzero force on first frame
    for atom_pos in atoms.positions:
        for coord in range(3):
            atom_pos[coord] += (2 * np.random.random() - 1) * 0.1

    yield atoms
    del atoms


@pytest.fixture(scope="module")
def md_params(super_cell):
    # Set up LAMMPS MD parameters
    md_kwargs = {
        "specorder": ["Si", "C"],
        "dump_period": 10,
        "velocity": ["all create 300 12345 dist gaussian rot yes mom yes"],
        "timestep": 0.001,
        "pair_style": "flare",
        "fix": ["1 all nvt temp 300.0 300.0 100.0"],
        "keep_alive": False,
        #"binary_dump": False,
    }
    
    md_dict = {}
    md_dict["LAMMPS"] = {"params": md_kwargs}

    #md_dict["VelocityVerlet"] = {}
    yield md_dict
    del md_dict


@pytest.fixture(scope="module")
def flare_calc():
    flare_calc_dict = {}
    
    # Initialize GP settings
    sigma = 2.0
    power = 2
    kernel = NormalizedDotProduct(sigma, power)
    cutoff_function = "quadratic"
    cutoff = 4.0
    radial_basis = "chebyshev"
    radial_hyps = [0.0, cutoff]
    cutoff_hyps = []
    atom_types = [14, 6]
    settings = [len(atom_types), 8, 4]
    calc = B2(radial_basis, cutoff_function, radial_hyps, cutoff_hyps, settings)
 
    gp = SGP_Wrapper(
        [kernel],
        [calc],
        cutoff,
        sigma_e=0.01,
        sigma_f=0.05,
        sigma_s=0.005,
        species_map={14:0, 6:1},
        variance_type="local",
        single_atom_energies=None,
        energy_training=True,
        force_training=True,
        stress_training=True,
        max_iterations=10,
    )

    for md_engine in md_list:
        flare_calc_dict[md_engine] = SGP_Calculator(sgp_model=gp)
    yield flare_calc_dict
    del flare_calc_dict


@pytest.fixture(scope="module")
def qe_calc():
    species = ["Si", "C"]
    specie_symbol_list = " ".join(species)
    # masses = [["1 28", "2 12"]
    masses = [
        f"{i} {atomic_masses[atomic_numbers[species[i]]]}" for i in range(len(species))
    ]
    coef_name = "SiC.tersoff"
    rootdir = os.getcwd()
    parameters = {
        "command": os.environ.get("lmp"),  # set up executable for ASE
        "keep_alive": False,
        "newton": "on",
        "pair_style": "tersoff",
        "pair_coeff": [f"* * {rootdir}/tmp/{coef_name} Si C"],
        "mass": masses,
    }

    # set up input params
    label = "sic"
    files = [f"test_files/{coef_name}"]

    dft_calculator = lammpsrun.LAMMPS(
        label=label,
        keep_tmp_files=True,
        tmp_dir="./tmp/",
        files=files,
        specorder=species,
    )
    dft_calculator.set(**parameters)

    yield dft_calculator
    del dft_calculator


@pytest.mark.parametrize("md_engine", md_list)
def test_otf_md(md_engine, md_params, super_cell, flare_calc, qe_calc):
    for f in ["restart.dat", "thermo.txt", "traj.xyz"]:
        if f in os.listdir():
            os.remove(f)

    flare_calculator = flare_calc[md_engine]
    # set up OTF MD engine
    otf_params = {
        "init_atoms": np.arange(12).tolist(),
        "output_name": md_engine,
        "std_tolerance_factor": -0.02,
        "max_atoms_added": len(super_cell.positions),
        "freeze_hyps": 0,
        "write_model": 4,
        "update_style": "threshold",
        "update_threshold": 0.01,
    }

    md_kwargs = md_params[md_engine]

    MaxwellBoltzmannDistribution(super_cell, 300 * units.kB)
    Stationary(super_cell)  # zero linear momentum
    ZeroRotation(super_cell)  # zero angular momentum

    #super_cell.calc = flare_calculator
    test_otf = ASE_OTF(
        super_cell,
        timestep=0.001,
        number_of_steps=number_of_steps,
        calculator=flare_calculator,
        dft_calc=qe_calc,
        md_engine=md_engine,
        md_kwargs=md_kwargs,
        trajectory=md_engine + "_otf.traj",
        **otf_params,
    )

    # TODO: test if mgp matches gp
    # TODO: see if there's difference between MD timestep & OTF timestep

    test_otf.run()

    # Check that the GP forces change.
    output_name = f"{md_engine}.out"
    otf_traj = OtfAnalysis(output_name)
    #comp1 = otf_traj.force_list[0][1, 0]
    #comp2 = otf_traj.force_list[-1][1, 0]
    #assert (comp1 != comp2)

    for f in glob.glob("scf*.pw*"):
        os.remove(f)
    for f in glob.glob("*.npy"):
        os.remove(f)
    for f in glob.glob("kv3*"):
        shutil.rmtree(f)
    for f in glob.glob("otf_data"):
        shutil.rmtree(f, ignore_errors=True)
    for f in glob.glob("out"):
        shutil.rmtree(f, ignore_errors=True)
    for f in os.listdir("./"):
        if ".mgp" in f or ".var" in f:
            os.remove(f)
        if "slurm" in f:
            os.remove(f)


@pytest.mark.parametrize("md_engine", md_list)
def test_load_checkpoint(md_engine):
    new_otf = ASE_OTF.from_checkpoint(md_engine + "_checkpt.json")
    assert new_otf.curr_step == number_of_steps
    new_otf.number_of_steps = new_otf.number_of_steps + 30
    new_otf.run()


@pytest.mark.parametrize("md_engine", md_list)
def test_otf_parser(md_engine):
    output_name = f"{md_engine}.out"
    otf_traj = OtfAnalysis(output_name)
#    try:
#        replicated_gp = otf_traj.make_gp()
#    except:
#        init_flare, _ = SGP_Calculator.from_file(md_engine + "_flare.json")
#        replicated_gp = otf_traj.make_gp(init_gp=init_flare.gp_model)

    print("ase otf traj parsed")
    # Check that the GP forces change.
    comp1 = otf_traj.force_list[0][1, 0]
    comp2 = otf_traj.force_list[-1][1, 0]
#    assert (comp1 != comp2)

#    for f in glob.glob(md_engine + "*"): 
#        os.remove(f)
