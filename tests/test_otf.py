import time, os, shutil, glob, subprocess
from copy import deepcopy
import pytest
import numpy as np

from flare.otf_parser import OtfAnalysis
from flare.gp import GaussianProcess
from flare.mgp import MappedGaussianProcess
from flare.ase.calculator import FLARE_Calculator
from flare.otf import OTF
from flare.utils.parameter_helper import ParameterHelper

from ase.constraints import FixAtoms
from ase import units
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.spacegroup import crystal
from ase.calculators.espresso import Espresso
from ase import io


md_list = [
    "VelocityVerlet",
#    "NVTBerendsen",
#    "NPTBerendsen",
#    "NPT",
#    "Langevin",
#    "NoseHoover",
]
number_of_steps = 5

np.random.seed(12345)

@pytest.fixture(scope="module")
def md_params():

    md_dict = {"temperature": 500}
    print(md_list)

    for md_engine in md_list:
        for f in glob.glob(md_engine + "*"):
            os.remove(f)

        if md_engine == "VelocityVerlet":
            md_dict[md_engine] = {}
        else:
            md_dict[md_engine] = {"temperature": md_dict["temperature"]}

        if md_engine == "NVTBerendsen":
            md_dict[md_engine].update({"taut": 0.5e3 * units.fs})
        elif md_engine == "NPTBerendsen":
            md_dict[md_engine].update({"pressure": 0.0, "compressibility_au": 1.0})
        elif md_engine == "NPT":
            md_dict[md_engine].update(
                {"externalstress": 0, "ttime": 25, "pfactor": 1.0}
            )
        elif md_engine == "Langevin":
            md_dict[md_engine].update({"friction": 0.02})
        elif md_engine == "NoseHoover":
            md_dict[md_engine].update({"nvt_q": 334.0})

    yield md_dict
    del md_dict


@pytest.fixture(scope="module")
def super_cell():

    from ase.spacegroup import crystal

    a = 5.0
    alpha = 90
    atoms = crystal(
        ["H", "He"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
        size=(2, 1, 1),
        cellpar=[a, a, a, alpha, alpha, alpha],
    )

    # jitter positions to give nonzero force on first frame
    for atom_pos in atoms.positions:
        for coord in range(3):
            atom_pos[coord] += (2 * np.random.random() - 1) * 0.5

    atoms.set_constraint(FixAtoms(indices=[0]))

    yield atoms
    del atoms


@pytest.fixture(scope="module")
def flare_calc():
    flare_calc_dict = {}
    for md_engine in md_list:

        # ---------- create gaussian process model -------------------

        # set up GP hyperparameters
        kernels = ["twobody", "threebody"]  # use 2+3 body kernel
        parameters = {"cutoff_twobody": 10.0, "cutoff_threebody": 6.0}
        pm = ParameterHelper(kernels=kernels, random=True, parameters=parameters)

        hm = pm.as_dict()
        hyps = hm["hyps"]
        cut = hm["cutoffs"]
        print("hyps", hyps)

        gp_model = GaussianProcess(
            kernels=kernels,
            component="mc",  # multi-component. For single-comp, use 'sc'
            hyps=hyps,
            cutoffs=cut,
            hyp_labels=["sig2", "ls2", "sig3", "ls3", "noise"],
            opt_algorithm="L-BFGS-B",
            n_cpus=1,
        )

        # ----------- create mapped gaussian process ------------------
        grid_params = {
            "twobody": {"grid_num": [64]},
            "threebody": {"grid_num": [16, 16, 16]},
        }

        mgp_model = MappedGaussianProcess(
            grid_params=grid_params, unique_species=[1, 2], n_cpus=1, var_map="pca"
        )

        # ------------ create ASE's flare calculator -----------------------
        flare_calculator = FLARE_Calculator(
            gp_model, mgp_model=mgp_model, par=True, use_mapping=True
        )

        flare_calc_dict[md_engine] = flare_calculator
        print(md_engine)
    yield flare_calc_dict
    del flare_calc_dict


@pytest.fixture(scope="module")
def qe_calc():
    from ase.calculators.lj import LennardJones

    dft_calculator = LennardJones()
    dft_calculator.parameters.sigma = 3.0
    dft_calculator.parameters.rc = 3 * dft_calculator.parameters.sigma

    yield dft_calculator
    del dft_calculator


@pytest.mark.parametrize("md_engine", md_list)
@pytest.mark.parametrize("write_model", [1, 2, 3])
def test_otf_md(md_engine, md_params, super_cell, flare_calc, qe_calc, write_model):
    flare_calculator = flare_calc[md_engine]
    output_name = f"{md_engine}_{write_model}"

    # set up OTF MD engine
    otf_params = {
        "init_atoms": [0, 1, 2, 3],
        "output_name": output_name,
        "std_tolerance_factor": 1.0,
        "max_atoms_added": len(super_cell.positions),
        "freeze_hyps": 10,
        "write_model": write_model,
    }
    #                  'use_mapping': flare_calculator.use_mapping}

    md_kwargs = md_params[md_engine]

    # intialize velocity
    temperature = md_params["temperature"]
    MaxwellBoltzmannDistribution(super_cell, temperature * units.kB)
    Stationary(super_cell)  # zero linear momentum
    ZeroRotation(super_cell)  # zero angular momentum

    super_cell.calc = flare_calculator
    test_otf = OTF(
        super_cell,
        dt=0.001, # ps
        number_of_steps=number_of_steps,
        dft_calc=qe_calc,
        md_engine=md_engine,
        md_kwargs=md_kwargs,
        trajectory=f"{output_name}_otf.traj",
        **otf_params,
    )

    # TODO: test if mgp matches gp
    # TODO: see if there's difference between MD timestep & OTF timestep

    test_otf.run()

    # Check that the GP forces change.
    otf_traj = OtfAnalysis(output_name + ".out")
    comp1 = otf_traj.force_list[-2][1, 0]
    comp2 = otf_traj.force_list[-1][1, 0]
    assert comp1 != comp2

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
@pytest.mark.parametrize("write_model", [1, 2, 3])
def test_load_checkpoint(md_engine, write_model):
    output_name = f"{md_engine}_{write_model}"
    new_otf = OTF.from_checkpoint(output_name + "_checkpt.json")
    assert new_otf.curr_step == number_of_steps
    new_otf.number_of_steps = new_otf.number_of_steps + 3
    new_otf.run()


@pytest.mark.parametrize("md_engine", md_list)
@pytest.mark.parametrize("write_model", [1, 2, 3])
def test_otf_parser(md_engine, write_model):
    output_name = f"{md_engine}_{write_model}"
    otf_traj = OtfAnalysis(output_name + ".out")
    try:
        replicated_gp = otf_traj.make_gp()
    except:
        init_flare = FLARE_Calculator.from_file(output_name + "_flare.json")
        replicated_gp = otf_traj.make_gp(init_gp=init_flare.gp_model)

    print("ase otf traj parsed")
    # Check that the GP forces change.
    comp1 = otf_traj.force_list[-2][1, 0]
    comp2 = otf_traj.force_list[-1][1, 0]
    assert comp1 != comp2

#    for f in glob.glob(output_name + "*"):
#        os.remove(f)
