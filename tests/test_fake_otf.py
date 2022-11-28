import time, os, shutil, glob, subprocess
from copy import deepcopy
import pytest
import numpy as np
import yaml

from flare.bffs.sgp.calculator import SGP_Calculator
from flare.io.otf_parser import OtfAnalysis
from flare.learners.otf import OTF
from flare.scripts.otf_train import fresh_start_otf, restart_otf
from ase.io import read, write

np.random.seed(12345)

md_list = ["Fake"]
number_of_steps = 5

if not os.environ.get("lmp", None):
    pytest.skip(
        "lmp not found in environment: Please install LAMMPS "
        "and set the $lmp env. variable to point to the executatble.",
        allow_module_level=True,
    )


@pytest.mark.parametrize("md_engine", md_list)
def test_traj_with_varying_sizes(md_engine):
    for tmpdir in [md_engine + "*ckpt_*", "myotf*ckpt_*", "direct*ckpt_*", "tmp*"]:
        for f in glob.glob(tmpdir):
            shutil.rmtree(f)

    for tmpfile in [
        "*.flare",
        "log.*",
        md_engine + "*",
        "myotf*",
        "direct*",
        "*.json",
        "*.tersoff",
    ]:
        for f in glob.glob(tmpfile):
            os.remove(f)

    # Modify the config for different MD engines
    with open("../examples/test_SGP_Fake_fresh.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["supercell"]["file"] = "test_files/sic_dft.xyz"
    # config["dft_calc"]["kwargs"]["filename"] = "test_files/sic_dft.xyz"
    config["otf"]["md_kwargs"]["filenames"] = ["test_files/sic_dft.xyz"]
    config["otf"]["output_name"] = md_engine

    # Run fake OTF
    fresh_start_otf(config)
    print("Done fake OTF")

    # Check that the training data is correct.
    output_name = f"{md_engine}.out"
    fake_otf_traj = OtfAnalysis(output_name)

    dft_traj = read("test_files/sic_dft.xyz", index=":")
    print("number of dft frames: ", len(dft_traj))
    for i in range(1, number_of_steps):
        assert np.allclose(
            dft_traj[i].positions, fake_otf_traj.position_list[i - 1], atol=1e-4
        ), i
        if i in fake_otf_traj.dft_frames:
            assert np.allclose(
                dft_traj[i].get_forces(),
                fake_otf_traj.gp_force_list[fake_otf_traj.dft_frames.index(i)],
                atol=1e-4,
            ), i


@pytest.mark.parametrize("md_engine", md_list)
def test_otf_md(md_engine):
    for f in glob.glob(md_engine + "*") + glob.glob("myotf*"):
        if "_ckpt" not in f:
            if f in os.listdir():
                os.remove(f)
        else:
            shutil.rmtree(f)

    # Run OTF with real MD
    with open("../examples/test_SGP_LMP_fresh.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["dft_calc"]["kwargs"]["command"] = os.environ.get("lmp").replace(
        "full", "half"
    )
    config["otf"]["md_kwargs"]["command"] = os.environ.get("lmp")
    config["otf"]["output_name"] = "myotf"
    fresh_start_otf(config)
    print("Done real OTF")

    # Modify the config for different MD engines
    with open("../examples/test_SGP_Fake_fresh.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["otf"]["output_name"] = md_engine

    # Make a fake AIMD trajectory from the previous real OTF run
    otf_traj = OtfAnalysis("myotf.out")
    md_traj = read("myotf_md.xyz", index=":")
    dft_traj = read("myotf_dft.xyz", index=":")
    assert len(dft_traj) == len(otf_traj.dft_frames)
    for i in range(len(dft_traj)):
        md_traj[otf_traj.dft_frames[i]] = dft_traj[i]
    write("fake_dft.xyz", md_traj)
    print("Done making dft data")

    # Run fake OTF
    fresh_start_otf(config)
    print("Done fake OTF")

    # Check that the training data is correct.
    output_name = f"{md_engine}.out"
    fake_otf_traj = OtfAnalysis(output_name)

    dft_traj = read("fake_dft.xyz", index=":")
    print("number of dft frames: ", len(dft_traj))
    for i in range(1, number_of_steps):
        assert np.allclose(
            dft_traj[i].positions, fake_otf_traj.position_list[i - 1], atol=1e-4
        ), i
        if i in fake_otf_traj.dft_frames:
            assert np.allclose(
                dft_traj[i].get_forces(),
                fake_otf_traj.gp_force_list[fake_otf_traj.dft_frames.index(i)],
                atol=1e-4,
            ), i

    # Check the final SGPs from real and fake trainings are the same
    real_sgp_calc, _ = SGP_Calculator.from_file("myotf_flare.json")
    fake_sgp_calc, _ = SGP_Calculator.from_file(f"{md_engine}_flare.json")
    assert np.allclose(real_sgp_calc.gp_model.hyps, fake_sgp_calc.gp_model.hyps)
    assert np.allclose(
        real_sgp_calc.gp_model.sparse_gp.Kuu, fake_sgp_calc.gp_model.sparse_gp.Kuu
    )

    # Check fake md with "direct" mode matches "bayesian" mode
    with open("../examples/test_SGP_Fake_fresh.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["supercell"]["file"] = "myotf_dft.xyz"
    # config["dft_calc"]["kwargs"]["filename"] = "myotf_dft.xyz"
    config["otf"]["md_kwargs"]["filenames"] = ["myotf_dft.xyz"]
    config["otf"]["output_name"] = "direct"
    config["otf"]["build_mode"] = "direct"
    config["otf"]["update_style"] = None
    config["otf"]["update_threshold"] = None
    fresh_start_otf(config)

    fake_sgp_calc, _ = SGP_Calculator.from_file(f"direct_flare.json")
    assert np.allclose(real_sgp_calc.gp_model.hyps, fake_sgp_calc.gp_model.hyps)
    assert np.allclose(
        real_sgp_calc.gp_model.sparse_gp.Kuu, fake_sgp_calc.gp_model.sparse_gp.Kuu
    )


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


def test_fakemd_match_gpfa():
    pytest.skip()


@pytest.mark.parametrize("md_engine", md_list)
def test_otf_parser(md_engine):
    output_name = f"{md_engine}.out"
    otf_traj = OtfAnalysis(output_name)

    print("ase otf traj parsed")
    # Check that the GP forces change.
    comp1 = otf_traj.force_list[0][1, 0]
    comp2 = otf_traj.force_list[-1][1, 0]
    assert comp1 != comp2

    for tmpdir in [md_engine + "*ckpt_*", "myotf*ckpt_*", "direct*ckpt_*", "tmp*"]:
        for f in glob.glob(tmpdir):
            shutil.rmtree(f)

    for tmpfile in [
        "*.flare",
        "log.*",
        md_engine + "*",
        "myotf*",
        "direct*",
        "*.json",
        "*.tersoff",
        "*.xyz",
    ]:
        for f in glob.glob(tmpfile):
            os.remove(f)
