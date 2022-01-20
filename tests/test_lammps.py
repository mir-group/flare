import pytest
import os
import numpy as np
from copy import deepcopy
from ase import Atom, Atoms
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from flare.ase.lammps import LAMMPS_MOD, LAMMPS_MD

@pytest.mark.skipif(
    not os.environ.get("lmp", False),
    reason=(
        "lmp not found "
        "in environment: Please install LAMMPS "
        "and set the $lmp env. "
        "variable to point to the executatble."
    ),
)
def test_lmp_calc():
    Ni = bulk("Ni", cubic=True)
    H = Atom("H", position=Ni.cell.diagonal() / 2)
    NiH = Ni + H

    os.environ["ASE_LAMMPSRUN_COMMAND"] = os.environ.get("lmp")
    os.system("wget https://openkim.org/files/MO_418978237058_005/NiAlH_jea.eam.alloy")
    files = ["NiAlH_jea.eam.alloy"]
    param_dict = {
        "pair_style": "eam/alloy",
        "pair_coeff": ["* * NiAlH_jea.eam.alloy H Ni"],
        "compute": ["1 all pair/local dist", "2 all reduce max c_1"],
        "velocity": ["all create 300 12345 dist gaussian rot yes mom yes"],
        "fix": ["1 all nvt temp 300 300 $(100.0*dt)"],
        "dump_period": 1,
        "timestep": 0.001,
    }

    ase_lmp_calc = LAMMPS(
        label="ase", files=files, keep_tmp_files=True, tmp_dir="tmp"
    )
    ase_lmp_calc.set(**param_dict)
    ase_atoms = deepcopy(NiH) 
    ase_atoms.calc = ase_lmp_calc
    ase_lmp_calc.calculate(ase_atoms)

    mod_lmp_calc = LAMMPS_MOD(
        label="mod", files=files, keep_tmp_files=True, tmp_dir="tmp"
    )
    mod_lmp_calc.set(**param_dict)
    mod_atoms = deepcopy(NiH) 
    mod_atoms.calc = mod_lmp_calc
    mod_lmp_calc.calculate(mod_atoms, set_atoms=True)

    assert np.allclose(ase_atoms.get_potential_energy(), mod_atoms.get_potential_energy())
    assert np.allclose(ase_atoms.get_forces(), mod_atoms.get_forces())
    assert np.allclose(ase_atoms.get_stress(), mod_atoms.get_stress())

    os.remove("NiAlH_jea.eam.alloy")
