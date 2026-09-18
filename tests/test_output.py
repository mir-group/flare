import numpy as np

from ase.io import read

from flare.atoms import FLARE_Atoms
from flare.io.output import Output


def test_write_md_config_writes_xyz_trajectory(tmp_path):
    atoms = FLARE_Atoms(symbols=["H"], positions=[[1, 2, 3]], cell=np.eye(3))
    atoms.forces = np.zeros((1, 3))
    atoms.stds = np.zeros((1, 3))

    output_name = tmp_path / "otf"
    output = Output(str(output_name), print_as_xyz=True, always_flush=True)
    output.write_md_config(
        dt=0.001,
        curr_step=1,
        structure=atoms,
        temperature=300,
        KE=0,
        start_time=0,
        dft_step=False,
        velocities=np.zeros((1, 3)),
    )
    output.conclude_run()

    trajectory = read(f"{output_name}.xyz", index=":")
    assert len(trajectory) == 1
    assert np.allclose(trajectory[0].positions, atoms.positions)
