"""Torch sparse GP through the stateful ASE on-the-fly interface."""

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")
pytest.importorskip("wandb")

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io import write

from flare.learners.otf import OTF
from flare.scripts.otf_train import get_flare_calc, main
from flare.tensor import B2
from flare.tensor.otf import TorchSGPCalculator, TorchSGPModel


def _atoms():
    return Atoms(
        numbers=[18, 18, 18],
        positions=[[0., 0., 0.], [1.5, .2, .1], [.2, 1.6, .3]],
        cell=np.eye(3) * 7, pbc=True,
    )


def _config():
    return dict(
        gp="TorchSGP", species=[18], cutoff=3.0,
        kernels=[dict(name="NormalizedDotProduct", sigma=1.3, power=2)],
        descriptors=[dict(name="B2", nmax=2, lmax=1,
                          radial_basis="chebyshev", cutoff_function="quadratic")],
        energy_noise=.2, forces_noise=.15, stress_noise=.04,
        variance_type="SOR", use_mapping=False, max_iterations=2,
    )


def test_configuration_model_update_prediction_and_roundtrip(tmp_path):
    config = _config()
    config["bounds"] = [[1.0, 1.5], [.1, .3], [.05, .2], [.02, .1]]
    calculator, kernels = get_flare_calc(config)
    assert kernels is None and isinstance(calculator, TorchSGPCalculator)
    atoms = _atoms()
    reference = LennardJones()
    labeled = atoms.copy()
    labeled.calc = reference
    forces = labeled.get_forces()
    energy = labeled.get_potential_energy()
    stress = -labeled.get_stress()[[0, 5, 4, 1, 3, 2]]
    model = calculator.gp_model
    model.update_db(atoms, forces, custom_range=[0, 1], energy=energy, stress=stress)
    first = model.predict(atoms)
    assert first["forces"].shape == (3, 3)
    assert first["stds"].shape == (3, 3)
    assert np.isfinite(first["energy"])
    assert all(np.isfinite(first[key]).all() for key in ("forces", "stress", "stds"))
    initial_likelihood = model.likelihood
    model.train()
    assert model.likelihood >= initial_likelihood - 1e-8
    assert np.isfinite(model.likelihood_gradient).all()
    assert all(low <= value <= high for value, (low, high)
               in zip(model.hyps, config["bounds"]))
    probe = model.hyps
    probe[0] += 1e-5
    plus = -float(model._fit_with_hyps(torch.tensor(probe[0]),
                                       torch.tensor(probe[1:])).negative_log_likelihood)
    probe[0] -= 2e-5
    minus = -float(model._fit_with_hyps(torch.tensor(probe[0]),
                                        torch.tensor(probe[1:])).negative_log_likelihood)
    np.testing.assert_allclose(model.likelihood_gradient[0],
                               (plus - minus) / (2e-5), rtol=2e-3, atol=2e-4)
    path = tmp_path / "torch_model.json"
    calculator.write_model(path)
    restored = TorchSGPCalculator.from_file(path)
    loaded, _ = get_flare_calc({"gp": "TorchSGP", "file": str(path)})
    assert isinstance(loaded, TorchSGPCalculator)
    np.testing.assert_allclose(restored.gp_model.hyps, model.hyps)
    for key, value in model.predict(atoms).items():
        np.testing.assert_allclose(restored.gp_model.predict(atoms)[key], value,
                                   rtol=1e-10, atol=1e-10)
    restored.gp_model.variance_type = "DTC"
    assert np.isfinite(restored.gp_model.predict(atoms)["stds"]).all()
    restored.gp_model.variance_type = "local"
    local = restored.gp_model.predict(atoms)["stds"]
    assert np.isfinite(local).all() and np.all(local[:, 1:] == 0)
    before = model.predict(atoms)["forces"]
    second = atoms.copy()
    second.positions[0, 0] += .05
    second.calc = LennardJones()
    model.update_db(
        second, second.get_forces(), custom_range=[2],
        energy=second.get_potential_energy(),
        stress=-second.get_stress()[[0, 5, 4, 1, 3, 2]],
    )
    assert not np.allclose(model.predict(atoms)["forces"], before)


@pytest.mark.parametrize("variance_type", ["SOR", "DTC", "local"])
def test_ase_otf_dft_updates_and_checkpoint_resume(tmp_path, monkeypatch,
                                                   variance_type):
    monkeypatch.chdir(tmp_path)
    config = _config()
    config["variance_type"] = variance_type
    calculator, _ = get_flare_calc(config)
    run = OTF(
        _atoms(), dt=.001, number_of_steps=3, dft_calc=LennardJones(),
        md_engine="VelocityVerlet", md_kwargs={}, flare_calc=calculator,
        force_only=True, std_tolerance_factor=-1e-8, init_atoms=[0, 1, 2],
        max_atoms_added=1, train_hyps=(1, 1), write_model=1,
        output_name="torch_otf",
    )
    run.run()
    assert run.curr_step == 3
    assert run.dft_count >= 2
    assert len(run.gp.training_data) == run.dft_count
    assert len(run.gp._inducing) == 3 + run.dft_count - 1
    assert not np.allclose(run.gp.hyps, [1.3, .2, .15, .04])
    saved = run.gp.predict(run.atoms)
    resumed = OTF.from_checkpoint("torch_otf_checkpt.json")
    assert resumed.curr_step == run.curr_step
    assert resumed.dft_count == run.dft_count
    for key in saved:
        np.testing.assert_allclose(resumed.gp.predict(resumed.atoms)[key], saved[key],
                                   rtol=1e-10, atol=1e-10)
    resumed.number_of_steps = 4
    resumed.run()
    assert resumed.curr_step == 4
    assert resumed.dft_count > run.dft_count

    uninterrupted_path = tmp_path / "uninterrupted"
    uninterrupted_path.mkdir()
    monkeypatch.chdir(uninterrupted_path)
    baseline_calc, _ = get_flare_calc(config)
    baseline = OTF(
        _atoms(), dt=.001, number_of_steps=4, dft_calc=LennardJones(),
        md_engine="VelocityVerlet", md_kwargs={}, flare_calc=baseline_calc,
        force_only=True, std_tolerance_factor=-1e-8, init_atoms=[0, 1, 2],
        max_atoms_added=1, train_hyps=(1, 1), write_model=1,
        output_name="torch_otf_uninterrupted",
    )
    baseline.run()
    assert resumed.dft_count == baseline.dft_count
    np.testing.assert_allclose(resumed.gp.hyps, baseline.gp.hyps,
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(resumed.atoms.positions, baseline.atoms.positions,
                               rtol=1e-10, atol=1e-10)
    assert [record["selected"] for record in resumed.gp.training_data] == [
        record["selected"] for record in baseline.gp.training_data
    ]
    for key, value in baseline.gp.predict(baseline.atoms).items():
        np.testing.assert_allclose(resumed.gp.predict(resumed.atoms)[key], value,
                                   rtol=1e-10, atol=1e-10)


def test_ase_otf_uses_energy_force_and_stress_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calculator, _ = get_flare_calc(_config())
    run = OTF(
        _atoms(), dt=.001, number_of_steps=1, dft_calc=LennardJones(),
        md_engine="VelocityVerlet", md_kwargs={}, flare_calc=calculator,
        force_only=False, init_atoms=[0, 1, 2], train_hyps=(0, 0),
        output_name="torch_otf_efs",
    )
    run.run()
    assert run.dft_count == 1
    record = run.gp.training_data[0]
    assert np.isfinite(record["energy"])
    assert np.isfinite(record["forces"]).all()
    assert np.isfinite(record["stress"]).all()
    assert run.gp._layouts[0].size == 1 + 9 + 6


def test_ase_npt_uses_torch_stress(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calculator, _ = get_flare_calc(_config())
    run = OTF(
        _atoms(), dt=.001, number_of_steps=2, dft_calc=LennardJones(),
        md_engine="NPT",
        md_kwargs=dict(temperature_K=500, externalstress=0, ttime=25, pfactor=1.0),
        flare_calc=calculator, force_only=False, init_atoms=[0, 1, 2],
        train_hyps=(0, 0), output_name="torch_npt",
    )
    run.run()
    assert run.curr_step == 2
    assert np.isfinite(run.flare_calc.results["stress"]).all()


def test_yaml_entry_point_runs_torch_ase_otf(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write("atoms.xyz", _atoms())
    config = dict(
        supercell=dict(file="atoms.xyz"),
        dft_calc=dict(name="LennardJones"),
        flare_calc=_config(),
        otf=dict(
            dt=.001, number_of_steps=2, md_engine="VelocityVerlet",
            md_kwargs={}, force_only=True, init_atoms=[0, 1, 2],
            train_hyps=[0, 0], write_model=1, output_name="yaml_torch_otf",
        ),
    )
    config_path = tmp_path / "otf.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr("sys.argv", ["otf_train", str(config_path)])
    main()
    assert (tmp_path / "yaml_torch_otf_checkpt.json").exists()
    assert (tmp_path / "yaml_torch_otf_flare.json").exists()


@pytest.mark.parametrize("variance_type", ["SOR", "DTC", "local"])
def test_adapter_predictions_match_native_sgp_when_available(variance_type):
    pytest.importorskip("flare.bffs.sgp._C_flare")
    atoms = _atoms()
    atoms.numbers = [18, 1, 18]
    labeled = atoms.copy()
    labeled.calc = LennardJones()
    forces = labeled.get_forces()
    energy = labeled.get_potential_energy()
    stress = -labeled.get_stress()[[0, 5, 4, 1, 3, 2]]
    config = _config()
    config["species"] = [18, 1]
    config["single_atom_energies"] = [.12, -.03]
    config["variance_type"] = variance_type
    torch_calc, _ = get_flare_calc(config)
    native_config = config.copy()
    native_config["gp"] = "SGP_Wrapper"
    native_calc, _ = get_flare_calc(native_config)
    for calculator in (torch_calc, native_calc):
        calculator.gp_model.update_db(
            atoms, forces, custom_range=[0, 1, 2], energy=energy, stress=stress,
            rel_e_noise=.8, rel_f_noise=1.3, rel_s_noise=.7,
        )
        calculator.gp_model.set_L_alpha()
    torch_values = torch_calc.gp_model.predict(atoms)
    native_calc.calculate(atoms)
    for key in ("energy", "forces", "stress", "stds"):
        np.testing.assert_allclose(torch_values[key], native_calc.results[key],
                                   rtol=2e-5, atol=2e-6)


def test_unsupported_mapping_and_engine():
    config = _config()
    config["use_mapping"] = True
    with pytest.raises(NotImplementedError, match="mapped"):
        get_flare_calc(config)
    config = _config()
    config["kernels"][0]["name"] = "SquaredExponential"
    with pytest.raises(NotImplementedError, match="NormalizedDotProduct"):
        get_flare_calc(config)
    config = _config()
    config["descriptors"][0]["name"] = "B3"
    with pytest.raises(NotImplementedError, match="B2"):
        get_flare_calc(config)
    config = _config()
    config["bounds"] = [[0, 1], [None, None], [None, None], [None, None]]
    with pytest.raises(ValueError, match="bounds must be positive"):
        get_flare_calc(config)
    model = TorchSGPModel(B2(1, 2, 1, 3.0), {18: 0})
    with pytest.raises(NotImplementedError, match="ASE MD engines"):
        OTF(_atoms(), dt=.001, number_of_steps=1, dft_calc=LennardJones(),
            md_engine="PyLAMMPS", md_kwargs={},
            flare_calc=TorchSGPCalculator(model))
    molecule = Atoms(numbers=[18, 18], positions=[[0., 0., 0.], [1.5, 0., 0.]])
    with pytest.raises(ValueError, match="full-rank cell"):
        OTF(molecule, dt=.001, number_of_steps=1, dft_calc=LennardJones(),
            md_engine="VelocityVerlet", md_kwargs={},
            flare_calc=TorchSGPCalculator(model))
