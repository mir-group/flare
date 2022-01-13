import time, os, shutil, glob, subprocess
from copy import deepcopy
import pytest
import numpy as np
#import sys
#config_path = os.getcwd()
#print("Importing configurations from", config_path)
#sys.path.append(config_path)
#print(locals().keys())
#from config import *
##import config
#print(locals().keys())
import pkgutil, pyclbr
import importlib
import inspect

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


# -------------- Set up Supercell from ASE Atoms --------------
super_cell = read(ase_atoms_file, ase_atoms_format)
super_cell *= replicate

# jitter positions to give nonzero force on first frame
for atom_pos in super_cell.positions:
    for coord in range(3):
        atom_pos[coord] += (2 * np.random.random() - 1) * jitter

# ----------------- Set up ASE DFT calculator -----------------
# find the module including the ASE DFT calculator class by name
dft_module_name = ""
for importer, modname, ispkg in pkgutil.iter_modules(ase_calculators.__path__):
    module_info = pyclbr.readmodule("ase.calculators." + modname)
    if ase_dft_calc_name in module_info:
        dft_module_name = modname
        break
print(dft_module_name)

# import ASE DFT calculator module, and build a DFT calculator class object
dft_calc = None
dft_module = importlib.import_module("ase.calculators." + dft_module_name)
for name, obj in inspect.getmembers(dft_module, inspect.isclass):
    if name == ase_dft_calc_name:
        print(name)
        dft_calc = obj(**ase_dft_calc_kwargs)
dft_calc.set(**ase_dft_calc_params)

# ---------- create gaussian process model -------------------

## set up GP hyperparameters
#kernels = ["twobody", "threebody"]  # use 2+3 body kernel
#parameters = {"cutoff_twobody": 10.0, "cutoff_threebody": 6.0}
#pm = ParameterHelper(kernels=kernels, random=True, parameters=parameters)
#
#hm = pm.as_dict()
#hyps = hm["hyps"]
#cut = hm["cutoffs"]
#print("hyps", hyps)

gp_model = GaussianProcess(
    kernels=kernels,
    component=components,
    hyps=hyps,
    cutoffs=cutoffs,
    hyp_labels=hyp_labels,
    opt_algorithm=opt_algorithm,
    n_cpus=n_cpus,
)

# ----------- create mapped gaussian process ------------------
coded_unique_species = symbols2numbers(unique_species)
mgp_model = MappedGaussianProcess(
    grid_params=grid_params, 
    unique_species=coded_unique_species, 
    n_cpus=n_cpus, 
    var_map=var_map,
)

# ------------ create ASE's flare calculator -----------------------
flare_calc = FLARE_Calculator(
    gp_model, mgp_model=mgp_model, par=par, use_mapping=use_mapping
)

# On-the-fly training engine
# intialize velocity
MaxwellBoltzmannDistribution(super_cell, temperature * units.kB)
Stationary(super_cell)  # zero linear momentum
ZeroRotation(super_cell)  # zero angular momentum

test_otf = OTF(
    super_cell,
    dt=dt,
    number_of_steps=number_of_steps,
    flare_calc=flare_calc,
    dft_calc=dft_calc,
    md_engine=md_engine,
    md_kwargs=md_kwargs,
    trajectory=f"{output_name}_otf.traj",
    init_atoms=init_atoms, 
    output_name=output_name,
    std_tolerance_factor=std_tolerance_factor,
    max_atoms_added=max_atoms_added,
    freeze_hyps=freeze_hyps,
    write_model=write_model,
)

# run on-the-fly training
test_otf.run()
