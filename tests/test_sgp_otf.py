import time, os, shutil, glob, subprocess
from copy import deepcopy
import pytest
import numpy as np
import yaml

from flare.io.otf_parser import OtfAnalysis
from flare.learners.otf import OTF
from flare.scripts.otf_train import fresh_start_otf, restart_otf


if not os.environ.get("lmp", None):
    pytest.skip(
        "lmp not found in environment: Please install LAMMPS "
        "and set the $lmp env. variable to point to the executatble.",
        allow_module_level=True,
    )


np.random.seed(12345)

# os.environ["ASE_LAMMPSRUN_COMMAND"] = os.environ.get("lmp")
md_list = ["VelocityVerlet", "PyLAMMPS"]
number_of_steps = 5


@pytest.mark.parametrize("md_engine", md_list)
def test_otf_md(md_engine):
    # Clean up old files
    for f in ["restart.dat", "thermo.txt", "traj.xyz"]:
        if f in os.listdir():
            os.remove(f)

    for f in glob.glob(md_engine + "*"):
        if "_ckpt" not in f:
            os.remove(f)
        else:
            shutil.rmtree(f)

    # Modify the config for different MD engines
    with open("../examples/test_SGP_LMP_fresh.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["dft_calc"]["kwargs"]["command"] = os.environ.get("lmp").replace(
        "full", "half"
    )

    if md_engine == "PyLAMMPS":
        config["flare_calc"]["use_mapping"] = True
        config["otf"]["md_kwargs"]["command"] = os.environ.get("lmp")
    else:
        config["flare_calc"]["use_mapping"] = False
        config["otf"]["md_engine"] = md_engine
        config["otf"]["md_kwargs"] = {}

    config["otf"]["output_name"] = md_engine

    print("fresh start")

    # Run OTF
    fresh_start_otf(config)

    print("done fresh start")

    # Check that the GP forces change.
    output_name = f"{md_engine}.out"
    otf_traj = OtfAnalysis(output_name)
    # comp1 = otf_traj.force_list[0][1, 0]
    # comp2 = otf_traj.force_list[-1][1, 0]
    # assert (comp1 != comp2)


@pytest.mark.parametrize("md_engine", md_list)
def test_otf_warm(md_engine):
    # Modify the config for different MD engines
    with open("../examples/test_SGP_LMP_fresh.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["dft_calc"]["kwargs"]["command"] = os.environ.get("lmp").replace(
        "full", "half"
    )

    if md_engine == "PyLAMMPS":
        config["flare_calc"]["use_mapping"] = True
        config["otf"]["md_kwargs"]["command"] = os.environ.get("lmp")
    else:
        config["flare_calc"]["use_mapping"] = False
        config["otf"]["md_engine"] = md_engine
        config["otf"]["md_kwargs"] = {}

    config["otf"]["output_name"] = md_engine

    config["flare_calc"] = {"gp": "SGP_Wrapper", "file": md_engine + "_flare.json"}

    print("warm start")

    # Run OTF
    fresh_start_otf(config)

    print("done warm start")

    # Check that the GP forces change.
    output_name = f"{md_engine}.out"
    otf_traj = OtfAnalysis(output_name)
    # comp1 = otf_traj.force_list[0][1, 0]
    # comp2 = otf_traj.force_list[-1][1, 0]
    # assert (comp1 != comp2)


@pytest.mark.parametrize("md_engine", md_list)
def test_load_checkpoint(md_engine):
    with open("../examples/test_restart.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["otf"]["checkpoint"] = md_engine + "_checkpt.json"
    restart_otf(config)

    for tmpfile in ["*.npy", "*.mgp", "*.var"]:
        for f in glob.glob(tmpfile):
            os.remove(f)

    for tmpdir in ["kv3*", "otf_data", "out", "mgp_grids"]:
        for f in glob.glob(tmpdir):
            shutil.rmtree(f, ignore_errors=True)


@pytest.mark.skipif(
    ("PyLAMMPS" not in md_list) or ("VelocityVerlet" not in md_list),
    reason="md_list does not include both PyLAMMPS and VelocityVerlet",
)
def test_lammps_match_ase_verlet():
    lammps_traj = OtfAnalysis("PyLAMMPS.out")
    verlet_traj = OtfAnalysis("VelocityVerlet.out")
    pos1 = lammps_traj.position_list[0]
    pos2 = verlet_traj.position_list[0]
    cell1 = lammps_traj.cell_list[0]
    cell2 = verlet_traj.cell_list[0]

    # check the volumes are the same
    assert np.linalg.det(cell1) == np.linalg.det(cell2)

    # check the positions only differ by a multiple of cell
    pos_diff = (pos1 - pos2) @ np.linalg.inv(cell1)
    for i in np.reshape(pos_diff.round(4), -1):
        assert i.is_integer()


@pytest.mark.parametrize("md_engine", md_list)
def test_otf_parser(md_engine):
    output_name = f"{md_engine}.out"
    otf_traj = OtfAnalysis(output_name)

    print("ase otf traj parsed")
    # Check that the GP forces change.
    comp1 = otf_traj.force_list[0][1, 0]
    comp2 = otf_traj.force_list[-1][1, 0]
    #    assert (comp1 != comp2)

    for tmpdir in [md_engine + "*ckpt_*", "tmp*"]:
        for f in glob.glob(tmpdir):
            shutil.rmtree(f)

    for tmpfile in ["*.flare", "log.*", md_engine + "*", "*SiC.tersoff", "*HHe.json"]:
        for f in glob.glob(tmpfile):
            os.remove(f)
