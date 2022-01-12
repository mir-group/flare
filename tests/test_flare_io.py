import pytest
import os
import numpy as np
try:
    from pymatgen.io.vasp.outputs import Vasprun
    _pmg_present = True
except:
    _pmg_present = False

from flare.struc import Structure, get_unique_species
from flare.dft_interface.vasp_util import md_trajectory_from_vasprun
from flare.utils.flare_io import md_trajectory_to_file, md_trajectory_from_file

pytestmark = pytest.mark.filterwarnings(
    "ignore::UserWarning", "ignore::pymatgen.io.vasp.outputs.UnconvergedVASPWarning"
)


@pytest.mark.skipif(not _pmg_present, reason=("pymatgen not found "))
def test_read_write_trajectory():
    structures = md_trajectory_from_vasprun("test_files/test_vasprun.xml")
    fname = "tst_traj.json"
    md_trajectory_to_file(fname, structures)
    fstructures = md_trajectory_from_file(fname)
    for s, f in zip(structures, fstructures):
        assert np.isclose(s.forces, f.forces).all()
        assert np.isclose(s.positions, f.positions).all()
    os.system("rm tst_traj.json")
