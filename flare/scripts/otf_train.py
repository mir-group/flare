import time, os, shutil, glob, subprocess, sys
from copy import deepcopy
import pytest
import pkgutil, pyclbr
import importlib
import inspect
import numpy as np

from flare.otf_parser import OtfAnalysis
from flare.gp import GaussianProcess
from flare.mgp import MappedGaussianProcess
from flare.ase.calculator import FLARE_Calculator
from flare.otf import OTF
from flare.utils.parameter_helper import ParameterHelper

from ase import units
import ase.calculators as ase_calculators
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase import io
from ase.symbols import symbols2numbers

import yaml

with open(sys.argv[1], "r") as f:
    config = yaml.safe_load(f)

# -------------- Set up Supercell from ASE Atoms --------------
super_cell = io.read(config["ase_atoms"]["file"], format=config["ase_atoms"]["format"])
super_cell *= config["ase_atoms"]["replicate"]

# jitter positions to give nonzero force on first frame
for atom_pos in super_cell.positions:
    for coord in range(3):
        atom_pos[coord] += (2 * np.random.random() - 1) * config["ase_atoms"]["jitter"]

# ----------------- Set up ASE DFT calculator -----------------
# find the module including the ASE DFT calculator class by name
dft_module_name = ""
ase_dft_calc_name = config["ase_dft_calc"]["name"]
for importer, modname, ispkg in pkgutil.iter_modules(ase_calculators.__path__):
    module_info = pyclbr.readmodule("ase.calculators." + modname)
    if ase_dft_calc_name in module_info:
        dft_module_name = modname
        break

# import ASE DFT calculator module, and build a DFT calculator class object
dft_calc = None
dft_module = importlib.import_module("ase.calculators." + dft_module_name)
for name, obj in inspect.getmembers(dft_module, inspect.isclass):
    if name == ase_dft_calc_name:
        dft_calc = obj(**config["ase_dft_calc"]["kwargs"])
dft_calc.set(**config["ase_dft_calc"]["params"])

# ---------- create gaussian process model -------------------
flare_config = config["ase_flare_calc"]

# set up GP hyperparameters
pm = ParameterHelper(
    kernels=flare_config["kernels"],
    random=flare_config["random_init_hyps"],
    parameters=flare_config["gp_parameters"],
)
hm = pm.as_dict()

gp_model = GaussianProcess(
    kernels=flare_config["kernels"],
    component="mc",
    hyps=hm["hyps"],
    cutoffs=hm["cutoffs"],
    hyp_labels=hm["hyp_labels"],
    opt_algorithm=flare_config["opt_algorithm"],
    n_cpus=flare_config["n_cpus"],
)

# ----------- create mapped gaussian process ------------------
if flare_config["use_mapping"]:
    unique_species = []
    for s in super_cell.symbols:
        if s not in unique_species:
            unique_species.append(s)
    coded_unique_species = symbols2numbers(unique_species)
    mgp_model = MappedGaussianProcess(
        grid_params=flare_config["grid_params"],
        unique_species=coded_unique_species,
        n_cpus=flare_config["n_cpus"],
        var_map=flare_config["var_map"],
    )
else:
    mgp_model = None

# ------------ create ASE's flare calculator -----------------------
flare_calc = FLARE_Calculator(
    gp_model=gp_model,
    mgp_model=mgp_model,
    par=flare_config["par"],
    use_mapping=flare_config["use_mapping"],
)

# On-the-fly training engine
# intialize velocity
MaxwellBoltzmannDistribution(super_cell, config["otf"]["temperature"] * units.kB)
Stationary(super_cell)  # zero linear momentum
ZeroRotation(super_cell)  # zero angular momentum

test_otf = OTF(
    super_cell,
    flare_calc=flare_calc,
    dft_calc=dft_calc,
    **config["otf"],
)

# run on-the-fly training
test_otf.run()
