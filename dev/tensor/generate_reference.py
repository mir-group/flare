#!/usr/bin/env python3
"""Regenerate the tiny native FLARE numerical reference (NumPy + native extension).

Build the extension with build_reference.py first, then run this script with
--extension and --build-manifest. This intentionally imports neither Torch nor
FLARE's Python wrappers. The checked-in fixture can be tested without C++.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import numpy as np

BASELINE = "199273867d48f3a91415fa44da6c0dfc5ead759f"
ROOT = Path(__file__).resolve().parents[2]
SETTINGS = {"n_species": 2, "n_radial": 3, "l_max": 2, "cutoff": 3.2,
            "radial_basis": "chebyshev", "cutoff_function": "quadratic",
            "radial_interval": [0.0, 1.0], "power": 2,
            "empty_threshold": 1e-8}
HYPERPARAMETERS = np.array([1.3, 0.2, 0.15, 0.04], dtype=np.float64)
JITTER = 1e-8
STRAIN_COMPONENTS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
CASES = {
    "molecule": {
        "positions": [[2.0, 2.0, 2.0], [3.13, 2.21, 2.09], [2.17, 3.43, 2.32]],
        "cell": (np.eye(3) * 12).tolist(), "species": [0, 0, 0],
        "pbc": [False, False, False],
    },
    "crystal": {
        "positions": [[0.12, 0.18, 0.21], [1.82, 1.61, 0.31],
                      [1.57, 0.23, 1.78], [0.27, 1.85, 1.62]],
        "cell": [[3.6, 0.1, 0.0], [0.0, 3.5, 0.15], [0.05, 0.0, 3.7]],
        "species": [0, 0, 0, 0], "pbc": [True, True, True],
    },
    "binary": {
        "positions": [[1.7, 2.1, 2.2], [2.81, 2.38, 2.05],
                      [2.12, 3.49, 2.44], [2.04, 2.42, 3.76]],
        "cell": (np.eye(3) * 11).tolist(), "species": [0, 1, 0, 1],
        "pbc": [False, False, False],
    },
    "missing_species": {
        "positions": [[2.0, 2.0, 2.0], [3.21, 2.13, 2.27], [2.35, 3.58, 2.11]],
        "cell": (np.eye(3) * 12).tolist(), "species": [1, 1, 1],
        "pbc": [False, False, False],
    },
    "isolated": {
        "positions": [[2.0, 2.0, 2.0], [8.0, 8.0, 8.0]],
        "cell": (np.eye(3) * 16).tolist(), "species": [0, 1],
        "pbc": [False, False, False],
    },
}
TRAINING = ["molecule", "binary"]
SELECTED = {"molecule": [0, 1], "binary": [1, 2]}
RELATIVE_NOISE = {"molecule": [1.0, 1.0, 1.0], "binary": [1.2, 0.8, 1.1]}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_native(path):
    # Direct extension import also avoids stale editable-install finders.
    spec = importlib.util.spec_from_file_location("_C_flare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_build(extension, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["baseline_revision"] != BASELINE:
        raise ValueError("The reference must be built from the pinned baseline revision.")
    if manifest["extension_sha256"] != sha256(extension):
        raise ValueError("Extension hash does not match its build manifest.")
    for name, expected in manifest["source_hashes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Native source differs from the recorded build: {name}")
    # The compiler command is supplied by the build script; omit local paths
    # from the portable fixture. Keep the source and binary fingerprints.
    return {key: value for key, value in manifest.items() if key != "compiler_command"}


def make_structure(native, descriptor, case, positions=None, cell=None):
    # Native Structure is fully periodic. Nonperiodic cases have large cells
    # and no periodic-image neighbors, so their reference values are identical.
    return native.Structure(np.asarray(case["cell"] if cell is None else cell),
                            case["species"],
                            np.asarray(case["positions"] if positions is None else positions),
                            SETTINGS["cutoff"], [descriptor])


def atom_order(descriptor_values, arrays):
    result = np.empty((descriptor_values.n_atoms,) + arrays[0].shape[1:])
    for indices, values in zip(descriptor_values.atom_indices, arrays):
        result[np.asarray(indices)] = values
    return result


def descriptor_arrays(structure):
    """Convert native species-major, neighbor derivatives to atom-major tensors."""
    desc = structure.descriptors[0]
    raw = atom_order(desc, list(desc.descriptors))
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = np.divide(raw, norm, out=np.zeros_like(raw), where=norm >= 1e-8)
    n, width = raw.shape
    position_jacobian = np.zeros((n, width, n, 3))
    strain_jacobian = np.zeros((n, width, 3, 3))
    for species in range(SETTINGS["n_species"]):
        for local, atom in enumerate(desc.atom_indices[species]):
            start = desc.cumulative_neighbor_counts[species][local]
            count = desc.neighbor_counts[species][local]
            for edge in range(start, start + count):
                neighbor = desc.neighbor_indices[species][edge]
                derivative = desc.descriptor_force_dervs[species][3 * edge:3 * edge + 3].T
                position_jacobian[atom, :, neighbor, :] += derivative
                position_jacobian[atom, :, atom, :] -= derivative
                displacement = desc.neighbor_coordinates[species][edge]
                strain_jacobian[atom] += derivative[:, :, None] * displacement[None, None, :]
    return {"raw_b2": raw, "normalized_b2": normalized,
            "raw_b2_position_jacobian": position_jacobian,
            "raw_b2_strain_jacobian": strain_jacobian}


def labels(n_atoms, index):
    # Numerical algebra fixtures, not a fitted physical potential.
    return np.r_[[-0.4 + 0.15 * index],
                 0.07 * np.sin(np.arange(3 * n_atoms) + 0.3 * (index + 1)),
                 0.012 * np.cos(np.arange(6) + 0.2 * index)]


def max_scaled_error(actual, expected):
    return float(np.max(np.abs(actual - expected) / (1.0 + np.abs(expected)), initial=0))


def check_geometry(native, descriptor, kernel, gp, name, arrays):
    """Central differences check independent physical coordinate/strain changes."""
    case = CASES[name]
    pos = np.array(case["positions"])
    cell = np.array(case["cell"])
    n = len(pos)
    reference = arrays[name + "__Kz_efs"]
    reports = []
    for step in [2e-5, 1e-5]:
        jac = np.empty_like(arrays[name + "__raw_b2_position_jacobian"])
        force = np.empty((gp.n_sparse, n, 3))
        strain_jac = np.empty_like(arrays[name + "__raw_b2_strain_jacobian"])
        stress = np.empty((gp.n_sparse, 6))
        for atom in range(n):
            for comp in range(3):
                shift = np.zeros_like(pos)
                shift[atom, comp] = step
                plus = make_structure(native, descriptor, case, pos + shift)
                minus = make_structure(native, descriptor, case, pos - shift)
                jac[:, :, atom, comp] = (descriptor_arrays(plus)["raw_b2"] -
                                        descriptor_arrays(minus)["raw_b2"]) / (2 * step)
                kp = kernel.envs_struc(gp.sparse_descriptors[0], plus.descriptors[0], kernel.kernel_hyperparameters)
                km = kernel.envs_struc(gp.sparse_descriptors[0], minus.descriptors[0], kernel.kernel_hyperparameters)
                force[:, atom, comp] = -(kp[:, 0] - km[:, 0]) / (2 * step)
        for comp in range(3):
            for comp2 in range(3):
                deformation = np.zeros((3, 3))
                deformation[comp, comp2] = step
                plus = make_structure(native, descriptor, case,
                                      pos @ (np.eye(3) + deformation).T,
                                      cell @ (np.eye(3) + deformation).T)
                minus = make_structure(native, descriptor, case,
                                       pos @ (np.eye(3) - deformation).T,
                                       cell @ (np.eye(3) - deformation).T)
                strain_jac[:, :, comp, comp2] = (descriptor_arrays(plus)["raw_b2"] -
                                               descriptor_arrays(minus)["raw_b2"]) / (2 * step)
                if (comp, comp2) in STRAIN_COMPONENTS:
                    kp = kernel.envs_struc(gp.sparse_descriptors[0], plus.descriptors[0], kernel.kernel_hyperparameters)
                    km = kernel.envs_struc(gp.sparse_descriptors[0], minus.descriptors[0], kernel.kernel_hyperparameters)
                    stress[:, STRAIN_COMPONENTS.index((comp, comp2))] = -(kp[:, 0] - km[:, 0]) / (2 * step * np.linalg.det(cell))
        report = {"step": step,
                  "descriptor_position": max_scaled_error(jac, arrays[name + "__raw_b2_position_jacobian"]),
                  "descriptor_strain": max_scaled_error(strain_jac, arrays[name + "__raw_b2_strain_jacobian"]),
                  "force_kernel": max_scaled_error(force.reshape(gp.n_sparse, -1), reference[:, 1:1 + 3*n]),
                  "stress_kernel": max_scaled_error(stress, reference[:, 1 + 3*n:])}
        if max(value for key, value in report.items() if key != "step") > 2e-8:
            raise AssertionError(f"Native derivative finite difference failed for {name}: {report}")
        reports.append(report)
    return reports


def generate(native):
    descriptor = native.B2("chebyshev", "quadratic", [0.0, SETTINGS["cutoff"]], [], [2, 3, 2])
    kernel = native.NormalizedDotProduct(HYPERPARAMETERS[0], SETTINGS["power"])
    gp = native.SparseGP([kernel], *HYPERPARAMETERS[1:])
    gp.Kuu_jitter = JITTER
    structures = {name: make_structure(native, descriptor, case) for name, case in CASES.items()}
    noise_kind = []
    relative_noise = []
    for index, name in enumerate(TRAINING):
        structure = structures[name]
        y = labels(structure.noa, index)
        structure.energy, structure.forces, structure.stresses = y[:1], y[1:-6], y[-6:]
        gp.add_training_structure(structure, [-1], *RELATIVE_NOISE[name])
        gp.add_specific_environments(structure, SELECTED[name])
        kinds = np.r_[0, np.ones(3 * structure.noa, dtype=int), np.full(6, 2)]
        noise_kind.extend(kinds)
        relative_noise.extend(np.array(RELATIVE_NOISE[name])[kinds])
    gp.update_matrices_QR()
    gp.compute_likelihood_gradient_stable(False)
    arrays = {"Kzz": np.array(gp.Kuu), "Kzy": np.array(gp.Kuf),
              "y": np.array(gp.y), "noise_variance": 1 / np.array(gp.noise_vector),
              "noise_kind": np.array(noise_kind), "relative_noise": np.array(relative_noise),
              "hyperparameters": HYPERPARAMETERS.copy(), "jitter": np.array(JITTER),
              "log_marginal_likelihood": np.array(gp.log_marginal_likelihood),
              "likelihood_gradient": np.array(gp.likelihood_gradient), "alpha": np.array(gp.alpha),
              "inducing_raw_b2": np.concatenate(gp.sparse_descriptors[0].descriptors),
              "inducing_species": np.concatenate([np.full(len(values), species, dtype=int)
                                                  for species, values in enumerate(gp.sparse_descriptors[0].descriptors)])}
    for name, structure in structures.items():
        prefix = name + "__"
        case = CASES[name]
        arrays.update({prefix + key: np.array(value) for key, value in case.items()})
        arrays.update({prefix + key: value for key, value in descriptor_arrays(structure).items()})
        arrays[prefix + "Kz_efs"] = kernel.envs_struc(gp.sparse_descriptors[0], structure.descriptors[0], kernel.kernel_hyperparameters)
        prior = kernel.struc_struc(structure.descriptors[0], structure.descriptors[0], kernel.kernel_hyperparameters)
        arrays[prefix + "prior_efs_diag"] = np.diag(prior).copy()
        gp.predict_mean(structure)
        arrays[prefix + "mean_efs"] = np.array(structure.mean_efs)
        gp.predict_SOR(structure)
        arrays[prefix + "sor_variance_efs"] = np.array(structure.variance_efs)
        gp.predict_DTC(structure)
        arrays[prefix + "dtc_variance_efs"] = np.array(structure.variance_efs)
        desc = structure.descriptors[0]
        local = np.empty(structure.noa)
        local[np.concatenate(desc.atom_indices)] = gp.compute_cluster_uncertainties(structure)[0]
        arrays[prefix + "local_variance"] = local
    # Independent observation-space Gaussian calculation. The native solver
    # works in inducing space via QR; this tiny dense check is intentionally
    # different and also exposes the native jitter-gradient discrepancy.
    regularized = arrays["Kzz"] + JITTER * np.eye(gp.n_sparse)
    solved = np.linalg.solve(regularized, arrays["Kzy"])
    q = arrays["Kzy"].T @ solved
    covariance = q + np.diag(arrays["noise_variance"])
    inverse_y = np.linalg.solve(covariance, arrays["y"])
    dense_likelihood = -0.5 * (arrays["y"] @ inverse_y +
                              np.linalg.slogdet(covariance)[1] +
                              gp.n_labels * np.log(2 * np.pi))
    np.testing.assert_allclose(dense_likelihood, arrays["log_marginal_likelihood"], atol=2e-11, rtol=2e-12)
    sensitivity = np.outer(inverse_y, inverse_y) - np.linalg.solve(covariance, np.eye(gp.n_labels))
    derivative = 2 / HYPERPARAMETERS[0] * (2 * q - solved.T @ arrays["Kzz"] @ solved)
    gradient = [0.5 * np.sum(sensitivity * derivative)]
    for kind in range(3):
        derivative_diag = 2 * HYPERPARAMETERS[kind + 1] * arrays["relative_noise"]**2 * (arrays["noise_kind"] == kind)
        gradient.append(0.5 * np.diag(sensitivity) @ derivative_diag)
    arrays["likelihood_gradient_dense"] = np.array(gradient)
    derivative_zz = 2 / HYPERPARAMETERS[0] * arrays["Kzz"]
    correction = 0.5 * np.trace(np.linalg.solve(arrays["Kzz"], derivative_zz) -
                                np.linalg.solve(regularized, derivative_zz))
    predicted_native = arrays["likelihood_gradient_dense"].copy()
    predicted_native[0] += correction
    np.testing.assert_allclose(predicted_native, arrays["likelihood_gradient"], rtol=2e-11, atol=3e-11)
    reports = {name: check_geometry(native, descriptor, kernel, gp, name, arrays) for name in CASES}
    hyper_fd = []
    for step in [2e-5, 1e-5]:
        result = []
        for index in range(len(HYPERPARAMETERS)):
            shift = np.zeros(4)
            shift[index] = step
            gp.set_hyperparameters(HYPERPARAMETERS + shift)
            gp.compute_likelihood_gradient_stable(False)
            plus = gp.log_marginal_likelihood
            gp.set_hyperparameters(HYPERPARAMETERS - shift)
            gp.compute_likelihood_gradient_stable(False)
            result.append((plus - gp.log_marginal_likelihood) / (2 * step))
        error = max_scaled_error(result, arrays["likelihood_gradient"])
        dense_error = max_scaled_error(result, arrays["likelihood_gradient_dense"])
        if dense_error > 2e-7:
            raise AssertionError(f"Likelihood finite difference failed: {dense_error}")
        hyper_fd.append({"step": step, "gradient": result, "scaled_error": error,
                         "dense_scaled_error": dense_error})
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise AssertionError(f"Nonfinite reference array: {name}")
    return arrays, {"geometry": reports, "likelihood": hyper_fd}


def save_npz(path, arrays):
    # Fixed ZIP timestamps and sorted keys make repeated runs byte-reproducible.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tests/tensor/data")
    parser.add_argument("--check", action="store_true", help="Compare regenerated arrays without writing fixtures.")
    args = parser.parse_args()
    provenance = verify_build(args.extension, args.build_manifest)
    arrays, finite_differences = generate(load_native(args.extension))
    npz_path = args.output_dir / "b2_reference.npz"
    if args.check:
        with np.load(npz_path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise AssertionError("Fixture keys differ; regenerate explicitly.")
            for key, value in arrays.items():
                np.testing.assert_allclose(value, saved[key], rtol=2e-10, atol=2e-10, err_msg=key)
        print(f"Validated {len(arrays)} reference arrays and two-step finite differences.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(npz_path, arrays)
    metadata = {"schema_version": 1, "baseline_revision": BASELINE,
                "settings": SETTINGS, "cases": list(CASES), "training_cases": TRAINING,
                "selected_atoms": SELECTED, "relative_noise": RELATIVE_NOISE,
                "hyperparameter_order": ["signal_std", "energy_noise_std", "force_noise_std", "stress_noise_std"],
                "observation_order": "Per structure: total energy, atom-major xyz forces, native stress xx xy xz yy yz zz",
                "stress_convention": "Native stress = -dE/depsilon / volume; R'=R@(I+epsilon).T, cell'=cell@(I+epsilon).T",
                "noise_convention": "noise_variance = (noise_std * relative_noise)^2; native noise_vector is precision",
                "descriptor_layout": "Flatten (species*n_radial+radial) pairs a<=b, then l; no sqrt(2) factors",
                "jacobian_layout": {"raw_b2_position_jacobian": "atom, feature, displaced_atom, xyz",
                                    "raw_b2_strain_jacobian": "atom, feature, epsilon_row, epsilon_column"},
                "periodicity_note": "C++ is fully periodic. False-PBC cases use boxes with no periodic-image neighbors.",
                "labels_note": "Deterministic synthetic labels test algebra; they are not DFT data or a physical potential.",
                "uncertainty_note": "Local values are variances, reordered to atom order; no noise or normalization by signal_std. Isolated descriptors are zero and have zero kernel variance.",
                "known_native_limitation": "The native stable likelihood signal gradient uses unjittered Kuu inverse in one complexity term; likelihood_gradient_dense is the correct fixed-jitter target, while likelihood_gradient preserves the native result. Tests account for the discrepancy analytically.",
                "precision": "float64", "array_count": len(arrays),
                "npz_sha256": sha256(npz_path), "build": provenance,
                "finite_difference_validation": finite_differences,
                "comparison_tolerances": {"fixture_regeneration": {"rtol": 2e-10, "atol": 2e-10},
                                          "geometry_scaled_error": 2e-8,
                                          "dense_likelihood_gradient_scaled_error": 2e-7,
                                          "torch_raw_descriptor": {"rtol": 2e-10, "atol": 1e-10},
                                          "torch_descriptor_derivative": {"rtol": 2e-9, "atol": 2e-9},
                                          "torch_observation_kernel": {"rtol": 2e-9, "atol": 2e-9},
                                          "torch_inducing_kernel": {"rtol": 2e-12, "atol": 2e-12}}}
    (args.output_dir / "b2_reference.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(arrays)} arrays to {npz_path}; geometry and hyperparameter finite differences passed.")


if __name__ == "__main__":
    main()
