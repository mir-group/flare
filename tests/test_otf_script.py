import time, os, shutil, glob, subprocess
from copy import deepcopy
import pytest
import numpy as np
import yaml

from flare.io.otf_parser import OtfAnalysis
from flare.scripts.otf_train import fresh_start_otf, restart_otf
from flare.bffs.gp.calculator import FLARE_Calculator

md_list = ["VelocityVerlet", "NPT"]
number_of_steps = 5
write_model_list = [1, 2, 3, 4]

np.random.seed(12345)


@pytest.fixture(scope="module")
def md_params():
    md_dict = {"temperature": 500}
    print(md_list)

    for md_engine in md_list:
        for f in glob.glob(md_engine + "*"):
            if "_ckpt" not in f:
                os.remove(f)
            else:
                shutil.rmtree(f)

        if md_engine == "VelocityVerlet":
            md_dict[md_engine] = {}
        elif md_engine == "NPT":
            md_dict[md_engine] = {"temperature": md_dict["temperature"]}
            md_dict[md_engine].update(
                {"externalstress": 0, "ttime": 25, "pfactor": 1.0}
            )
        else:
            raise Exception

    yield md_dict
    del md_dict


@pytest.mark.parametrize("md_engine", md_list)
@pytest.mark.parametrize("write_model", write_model_list)
def test_otf_md(md_engine, md_params, write_model):
    output_name = f"{md_engine}_{write_model}"

    with open("../examples/test_GP_LJ_fresh.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["otf"]["md_engine"] = md_engine
    config["otf"]["md_kwargs"] = md_params[md_engine]
    config["otf"]["output_name"] = output_name
    config["otf"]["write_model"] = write_model

    fresh_start_otf(config)

    # Check that the GP forces change.
    otf_traj = OtfAnalysis(output_name + ".out")
    comp1 = otf_traj.force_list[-2][1, 0]
    comp2 = otf_traj.force_list[-1][1, 0]
    assert comp1 != comp2


@pytest.mark.parametrize("md_engine", md_list)
@pytest.mark.parametrize("write_model", write_model_list)
def test_load_checkpoint(md_engine, write_model):
    with open("../examples/test_restart.yaml", "r") as f:
        config = yaml.safe_load(f)

    output_name = f"{md_engine}_{write_model}"
    config["otf"]["checkpoint"] = output_name + "_checkpt.json"
    restart_otf(config)

    for tmpfile in ["*.npy", "*.mgp", "*.var"]:
        for f in glob.glob(tmpfile):
            os.remove(f)

    for tmpdir in ["kv3*", "otf_data", "out", "mgp_grids"]:
        for f in glob.glob(tmpdir):
            shutil.rmtree(f, ignore_errors=True)


@pytest.mark.parametrize("md_engine", md_list)
@pytest.mark.parametrize("write_model", write_model_list)
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

    # Test the new merged OTF matches the output of previous ASE_OTF
    if md_engine in ["VelocityVerlet", "NPT"] and write_model == 1:
        otf_traj_old = OtfAnalysis(f"test_files/{output_name}.out")
        assert np.allclose(otf_traj.force_list[-1], otf_traj_old.force_list[-1])
        assert np.allclose(otf_traj.position_list[-1], otf_traj_old.position_list[-1])

    for f in glob.glob(output_name + "*"):
        if "ckpt" in f and "json" not in f:
            shutil.rmtree(f)
        else:
            os.remove(f)
