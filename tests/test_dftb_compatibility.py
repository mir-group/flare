"""Regression coverage for ASE's xTB DFTB+ input generation."""

from ase import Atoms
from ase.calculators.dftb import Dftb


def test_xtb_input_does_not_require_slater_koster_files(tmp_path):
    """GFN1-xTB must not look up ``.skf`` files while writing its input."""
    calculator = Dftb(
        directory=tmp_path,
        Hamiltonian_="xTB",
        Hamiltonian_Method="GFN1-xTB",
    )

    calculator.write_input(Atoms("H", positions=[[0.0, 0.0, 0.0]]))

    input_text = (tmp_path / "dftb_in.hsd").read_text()
    assert "Hamiltonian = xTB" in input_text
    assert "Method = GFN1-xTB" in input_text
    assert not list(tmp_path.glob("*.skf"))
