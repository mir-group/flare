"""Regression coverage for NumPy scalar and ASE LAMMPS API changes."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from flare.atoms import FLARE_Atoms
from flare.bffs.gp.calculator import FLARE_Calculator
from flare.md import lammps


def test_lammps_custom_commands_remain_complete_lines(monkeypatch, tmp_path):
    atoms = FLARE_Atoms(symbols="H", positions=[[0, 0, 0]], cell=[10, 10, 10])
    calc = lammps.LAMMPS_MOD(
        command="unused",
        tmp_dir=str(tmp_path),
        model_post=["group mobile type 1"],
        region=["box block 0 1 0 1 0 1"],
        compute=["unc all flare/std/atom coefficients.flare"],
        fix=["thermostat all nvt temp 300 300 0.1"],
    )
    monkeypatch.setattr(calc, "run", Mock())

    calc.calculate(atoms)

    assert calc.parameters["model_post"] == [
        "group mobile type 1",
        "\nregion box block 0 1 0 1 0 1\n",
        "\ncompute unc all flare/std/atom coefficients.flare\n",
    ]
    # ASE supplies its default NVE fix only when the caller has not set one.
    assert calc.parameters["fix"] == ["thermostat all nvt temp 300 300 0.1"]


@pytest.mark.parametrize("variance", [4.0, np.array([4.0]), np.array([[4.0]])])
@pytest.mark.parametrize("rebuild", [False, True])
def test_mapped_calculator_accepts_singleton_variance(variance, rebuild):
    atoms = FLARE_Atoms(symbols="H", positions=[[0, 0, 0]], cell=[10, 10, 10])
    prediction = (np.zeros(3), variance, np.zeros(6), 1.5)
    gp_model = SimpleNamespace(cutoffs={"twobody": 2.0})
    single_map = Mock(bounds=[[1.0], [2.0]])
    body_map = Mock(maps=[single_map])
    body_map.find_map_index.return_value = 0
    mgp_model = SimpleNamespace(
        hyps_mask=None,
        maps={"twobody": body_map},
        predict=Mock(return_value=prediction),
    )
    if rebuild:
        mgp_model.predict.side_effect = [
            ValueError({"twobody": ["H_H"]}, {"twobody": [0.5]}),
            prediction,
        ]

    calc = FLARE_Calculator(gp_model, mgp_model=mgp_model, use_mapping=True)
    if rebuild:
        with pytest.warns(UserWarning, match="Re-build map"):
            calc.calculate_mgp(atoms)
        single_map.build_map.assert_called_once_with(gp_model)
    else:
        calc.calculate_mgp(atoms)

    np.testing.assert_array_equal(calc.results["stds"], [[2.0, 0.0, 0.0]])
    np.testing.assert_array_equal(calc.results["local_energies"], [1.5])


class _FixedSGP(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self):
        super().__init__()
        self.gp_model = SimpleNamespace(single_atom_energies=None, species_map={})

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros(6),
            "stds": np.array([[0.2, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        }


@pytest.mark.parametrize("column_shape", [(2,), (2, 1)])
@pytest.mark.parametrize("fallback", [False, True])
def test_sgp_match_accepts_scalar_dump_columns(monkeypatch, column_shape, fallback):
    atoms = FLARE_Atoms(
        symbols="H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10]
    )
    atoms.set_array("c_unc", np.array([0.2, 0.5]).reshape(column_shape))
    atoms.calc = SinglePointCalculator(
        atoms, energy=float(fallback), forces=np.zeros((2, 3)), stress=np.zeros(6)
    )
    sgp_calc = _FixedSGP()
    lmp_calc = Mock(
        results={"energy": 0.0, "forces": np.zeros((2, 3)), "stress": np.zeros(6)}
    )
    get_calc = Mock(return_value=(lmp_calc, {}))
    monkeypatch.setattr(lammps, "get_flare_lammps_calc", get_calc)
    monkeypatch.setattr(lammps, "read", Mock(return_value=atoms))
    logger = Mock()

    lammps.check_sgp_match(atoms, sgp_calc, logger, ["H"], "unused")

    assert get_calc.call_count == int(fallback)
    logger.info.assert_any_call("Maximal absolute uncertainty difference: 0.0")


def test_lammps_backup_appends_only_new_history(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tmp").mkdir()

    md = lammps.LAMMPS_MD.__new__(lammps.LAMMPS_MD)
    md.thermo_file = "otf_thermo.txt"
    md.traj_xyz_file = "otf_md.xyz"
    md.params = {"units": "metal"}

    frame = FLARE_Atoms(symbols="H", positions=[[0, 0, 0]], cell=[10, 10, 10])
    frame.calc = SinglePointCalculator(
        frame, energy=0.0, forces=np.zeros((1, 3)), stress=np.zeros(6)
    )
    thermo_row = "0 300 0 1 1 0 0 0 0 0 0 0\n"
    thermo_path = tmp_path / "tmp" / md.thermo_file
    thermo_path.write_text(thermo_row * 2)

    md.backup([frame])

    with thermo_path.open("a") as thermo_file:
        thermo_file.write(thermo_row * 2)

    # The second backup must not parse the previously written XYZ trajectory.
    monkeypatch.setattr(lammps, "read", Mock(side_effect=AssertionError))
    md.backup([frame])

    assert (tmp_path / md.thermo_file).read_text() == thermo_path.read_text()
    assert len(read(md.traj_xyz_file, index=":")) == 2
