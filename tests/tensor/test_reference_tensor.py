"""Torch parity against the recorded native reference, with no native import."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from flare.tensor import B2, StructureBatch
from flare.tensor.kernels import normalized_dot_product
from flare.tensor.neighbors import neighbor_list


DATA = Path(__file__).parent / "data"
CASES = ("molecule", "crystal", "binary", "missing_species", "isolated")


@pytest.fixture(scope="module")
def reference():
    with np.load(DATA / "b2_reference.npz", allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in arrays.files}
    metadata = json.loads((DATA / "b2_reference.json").read_text())
    return values, metadata


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def inputs(reference, case):
    data, meta = reference
    settings = meta["settings"]
    descriptor = B2(
        settings["n_species"], settings["n_radial"], settings["l_max"],
        settings["cutoff"],
    )
    positions = tensor(data[case + "__positions"])
    cell = tensor(data[case + "__cell"])
    species = torch.as_tensor(data[case + "__species"], dtype=torch.long)
    pbc = torch.as_tensor(data[case + "__pbc"], dtype=torch.bool)
    edges = neighbor_list(positions, cell, descriptor.cutoff, pbc)
    return descriptor, positions, cell, species, pbc, edges


@pytest.mark.parametrize("case", CASES)
def test_raw_descriptor_and_derivatives(reference, case):
    data, _ = reference
    descriptor, positions, cell, species, pbc, edges = inputs(reference, case)

    def evaluate(coordinates):
        return descriptor(coordinates, cell, species, pbc, edges=edges)

    def strained(strain):
        deformation = torch.eye(3, dtype=positions.dtype) + strain
        return descriptor(
            positions @ deformation.T, cell @ deformation.T, species, pbc,
            edges=edges,
        )

    values = evaluate(positions)
    position_jacobian = torch.func.jacrev(evaluate)(positions)
    strain_jacobian = torch.func.jacrev(strained)(torch.zeros_like(cell))
    np.testing.assert_allclose(values.detach(), data[case + "__raw_b2"], atol=1e-10, rtol=2e-10)
    np.testing.assert_allclose(
        position_jacobian.detach(), data[case + "__raw_b2_position_jacobian"],
        atol=2e-9, rtol=2e-9,
    )
    np.testing.assert_allclose(
        strain_jacobian.detach(), data[case + "__raw_b2_strain_jacobian"],
        atol=2e-9, rtol=2e-9,
    )


@pytest.mark.parametrize("case", CASES)
def test_reference_to_energy_force_stress_blocks(reference, case):
    data, meta = reference
    descriptor, positions, cell, species, pbc, edges = inputs(reference, case)
    inducing = tensor(data["inducing_raw_b2"])
    inducing_species = torch.as_tensor(data["inducing_species"], dtype=torch.long)
    amplitude = float(data["hyperparameters"][0])

    def energy_columns(coordinates, lattice):
        values = descriptor(coordinates, lattice, species, pbc, edges=edges)
        return normalized_dot_product(
            inducing, values, inducing_species, species,
            amplitude=amplitude, power=meta["settings"]["power"],
        ).sum(dim=1)

    energy = energy_columns(positions, cell)
    forces = -torch.func.jacrev(energy_columns, argnums=0)(positions, cell)

    def strained(strain):
        deformation = torch.eye(3, dtype=positions.dtype) + strain
        return energy_columns(positions @ deformation.T, cell @ deformation.T)

    stress_matrix = -torch.func.jacrev(strained)(torch.zeros_like(cell)) / torch.linalg.det(cell).abs()
    stresses = stress_matrix[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    block = torch.cat((energy[:, None], forces.flatten(1), stresses), dim=1)
    np.testing.assert_allclose(block.detach(), data[case + "__Kz_efs"], atol=2e-9, rtol=2e-9)


def test_inducing_covariance(reference):
    data, meta = reference
    inducing = tensor(data["inducing_raw_b2"])
    species = torch.as_tensor(data["inducing_species"], dtype=torch.long)
    matrix = normalized_dot_product(
        inducing, inducing, species, species,
        amplitude=float(data["hyperparameters"][0]), power=meta["settings"]["power"],
    )
    np.testing.assert_allclose(matrix, data["Kzz"], atol=2e-12, rtol=2e-12)


def test_ragged_descriptor_matches_separate_reference_structures(reference):
    data, _ = reference
    structures = [inputs(reference, case) for case in CASES]
    sizes = [len(entry[1]) for entry in structures]
    batch = StructureBatch(
        positions=torch.cat([entry[1] for entry in structures]),
        species=torch.cat([entry[3] for entry in structures]),
        cells=torch.stack([entry[2] for entry in structures]),
        pbc=torch.stack([entry[4] for entry in structures]),
        ptr=torch.as_tensor(np.cumsum([0] + sizes), dtype=torch.long),
    )
    values = structures[0][0].forward_batch(batch)
    expected = np.concatenate([data[case + "__raw_b2"] for case in CASES])
    np.testing.assert_allclose(values, expected, atol=1e-10, rtol=2e-10)


def test_backend_import_does_not_load_native_extension():
    script = """
import sys
from importlib.machinery import PathFinder
# Resolve this checkout before an unrelated editable FLARE installation.
sys.meta_path.insert(0, PathFinder)
class NoNative:
    def find_spec(self, fullname, path=None, target=None):
        if '_C_flare' in fullname or fullname.startswith('flare.bffs'):
            raise AssertionError('Native backend import: ' + fullname)
sys.meta_path.insert(0, NoNative())
import torch
from flare.tensor import B2
q = B2(n_species=1)(torch.zeros((1,3), dtype=torch.float64),
                    torch.zeros((3,3), dtype=torch.float64),
                    torch.zeros(1, dtype=torch.long), pbc=False)
assert torch.count_nonzero(q) == 0
"""
    subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
        check=True, capture_output=True, text=True,
    )
