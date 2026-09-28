"""Stateful ASE adapter for the experimental Torch sparse GP backend."""

import json
from numbers import Real
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from scipy.optimize import minimize
import torch

from .b2 import B2
from .kernels import normalized_dot_product
from .lambda_cache import (
    assemble_cached_grouped_lambda_observation_covariance,
    build_grouped_lambda_cache,
)
from .linalg import fit_sparse_gp
from .observations import ObservationLayout, native_stress_to_ase
from .prediction import predict_mean_efs
from .uncertainty import predict_variance_efs


class TorchSGPModel:
    """Mutable training state behind the ASE OTF calculator.

    This initial adapter supports one normalized-dot B2 kernel on float64 CPU.
    Geometry and descriptor caches are fixed between database updates;
    hyperparameter gradients flow through the kernel and noise values.
    """

    def __init__(
        self, descriptor, species_map, *, amplitude=1.0, power=2,
        sigma_e=0.1, sigma_f=0.1, sigma_s=0.1, variance_type="SOR",
        single_atom_energies=None, energy_training=True, force_training=True,
        stress_training=True, jitter=1e-8, max_iterations=10,
        opt_method="BFGS", bounds=None,
    ):
        if not isinstance(descriptor, B2):
            raise TypeError("descriptor must be a Torch B2")
        if sorted(species_map.values()) != list(range(descriptor.n_species)):
            raise ValueError("species_map must cover the fixed B2 vocabulary")
        if variance_type not in ("SOR", "DTC", "local"):
            raise ValueError("variance_type must be SOR, DTC, or local")
        if not isinstance(power, int) or isinstance(power, bool) or power < 1:
            raise ValueError("power must be a positive integer")
        self.descriptor = descriptor
        self.species_map = {int(k): int(v) for k, v in species_map.items()}
        self.variance_type = variance_type
        self.power = power
        self.single_atom_energies = (
            np.zeros(descriptor.n_species, dtype=np.float64)
            if single_atom_energies is None else
            np.asarray(single_atom_energies, dtype=np.float64)
        )
        if (self.single_atom_energies.shape != (descriptor.n_species,)
                or not np.isfinite(self.single_atom_energies).all()):
            raise ValueError("single_atom_energies must cover the species vocabulary")
        self.energy_training = bool(energy_training)
        self.force_training = bool(force_training)
        self.stress_training = bool(stress_training)
        self._hyps = np.asarray([amplitude, sigma_e, sigma_f, sigma_s], dtype=np.float64)
        if not np.isfinite(self._hyps).all() or np.any(self._hyps <= 0):
            raise ValueError("amplitude and noise standard deviations must be positive")
        self.jitter = float(jitter)
        self.max_iterations = int(max_iterations)
        self.opt_method = opt_method
        self.bounds = bounds
        self._log_bounds()
        self.hyps_mask = None
        self.hyp_labels = ["Signal", "Energy noise", "Force noise", "Stress noise"]
        self.likelihood_gradient = np.zeros(4, dtype=np.float64)
        self.training_data = []
        self._caches = []
        self._layouts = []
        self._labels = []
        self._relative_noises = []
        self._inducing = torch.empty((0, descriptor.n_features), dtype=torch.float64)
        self._inducing_species = torch.empty(0, dtype=torch.long)
        self.posterior = None
        self._dirty = True
        self._refit()

    @property
    def cutoff(self):
        return self.descriptor.cutoff

    @property
    def hyps(self):
        return self._hyps.copy()

    @property
    def hyps_and_labels(self):
        return self.hyps, self.hyp_labels

    @property
    def force_noise(self):
        return float(self._hyps[2])

    @property
    def likelihood(self):
        return -float(self.posterior.negative_log_likelihood) if self.posterior else 0.0

    def __len__(self):
        return len(self.training_data)

    def __str__(self):
        return (f"Torch sparse GP: {len(self.training_data)} training frames, "
                f"{len(self._inducing)} inducing environments, hyps={self._hyps}")

    def _geometry(self, atoms):
        if not isinstance(atoms, Atoms):
            raise TypeError("Torch OTF updates require ASE Atoms")
        try:
            coded = [self.species_map[int(number)] for number in atoms.numbers]
        except KeyError as exc:
            raise ValueError(f"atomic number {exc.args[0]} is outside species_map") from exc
        positions = torch.as_tensor(np.asarray(atoms.positions).copy(), dtype=torch.float64)
        cell = torch.as_tensor(np.asarray(atoms.cell).copy(), dtype=torch.float64)
        species = torch.tensor(coded, dtype=torch.long)
        pbc = tuple(bool(value) for value in atoms.pbc)
        return positions, cell, species, pbc

    def _refit(self):
        amplitude = torch.tensor(self._hyps[0], dtype=torch.float64)
        noises = torch.tensor(self._hyps[1:], dtype=torch.float64)
        self.posterior = self._fit_with_hyps(amplitude, noises).detach()
        self._dirty = False

    def _fit_with_hyps(self, amplitude, noises):
        inducing = self._inducing
        kinds = self._inducing_species
        Kzz = normalized_dot_product(
            inducing, inducing, kinds, kinds, amplitude=amplitude, power=self.power,
        )
        Kzy = assemble_cached_grouped_lambda_observation_covariance(
            self._caches, inducing, kinds, self._layouts,
            amplitude=amplitude, power=self.power,
        )
        labels = (torch.cat(self._labels) if self._labels else
                  torch.empty(0, dtype=torch.float64))
        noise_variance = (
            torch.cat([layout.noise_variance(noises, relative)
                       for layout, relative in zip(self._layouts, self._relative_noises)])
            if self._layouts else torch.empty(0, dtype=torch.float64)
        )
        return fit_sparse_gp(Kzz, Kzy, labels, noise_variance, jitter=self.jitter)

    def update_db(
        self, structure, forces, custom_range=(), energy=None, stress=None,
        mode="specific", update_qr=True, atom_indices=None,
        rel_e_noise=1.0, rel_f_noise=1.0, rel_s_noise=1.0,
    ):
        """Append one DFT-labeled structure and selected inducing atoms."""
        if mode not in ("specific", "all"):
            raise NotImplementedError("Torch OTF supports specific or all inducing selection")
        positions, cell, species, pbc = self._geometry(structure)
        selected = (list(range(len(positions))) if mode == "all" else
                    [int(index) for index in custom_range])
        if len(selected) != len(set(selected)) or any(
            index < 0 or index >= len(positions) for index in selected
        ):
            raise ValueError("inducing atom indices must be unique and in range")
        layout = ObservationLayout.from_masks(
            len(positions), energy=self.energy_training and energy is not None,
            force_mask=torch.full((len(positions), 3),
                                  self.force_training and forces is not None,
                                  dtype=torch.bool),
            stress_mask=torch.full((6,), self.stress_training and stress is not None,
                                   dtype=torch.bool),
        )
        energy_value = None if energy is None else float(np.asarray(energy).reshape(-1)[0])
        force_values = None if forces is None else np.asarray(forces, dtype=np.float64).reshape(-1, 3)
        stress_values = None if stress is None else np.asarray(stress, dtype=np.float64).reshape(6)
        relative = torch.tensor([rel_e_noise, rel_f_noise, rel_s_noise], dtype=torch.float64)
        if not torch.isfinite(relative).all() or bool((relative <= 0).any()):
            raise ValueError("relative noise multipliers must be positive and finite")
        labels = layout.pack_labels(
            energy=None if energy_value is None else torch.tensor(energy_value, dtype=torch.float64),
            forces=None if force_values is None else torch.tensor(force_values, dtype=torch.float64),
            stress=None if stress_values is None else torch.tensor(stress_values, dtype=torch.float64),
            species=species, atomic_offsets=self.single_atom_energies,
        )
        cache = build_grouped_lambda_cache(
            self.descriptor, positions, cell, species, pbc=pbc,
        )
        self.training_data.append(dict(
            numbers=np.asarray(structure.numbers, dtype=int).tolist(),
            positions=positions.tolist(), cell=cell.tolist(), pbc=list(pbc),
            energy=energy_value,
            forces=None if force_values is None else force_values.tolist(),
            stress=None if stress_values is None else stress_values.tolist(),
            selected=selected, relative_noise=relative.tolist(),
        ))
        self._caches.append(cache)
        self._layouts.append(layout)
        self._labels.append(labels)
        self._relative_noises.append(relative)
        if selected:
            self._inducing = torch.cat((self._inducing, cache.values[selected]))
            self._inducing_species = torch.cat((self._inducing_species, species[selected]))
        self._dirty = True
        if update_qr:
            self._refit()

    def set_L_alpha(self):
        if self._dirty:
            self._refit()

    def _log_bounds(self):
        if self.bounds is None:
            return None
        if (not isinstance(self.bounds, (list, tuple)) or len(self.bounds) != 4
                or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                       for pair in self.bounds)):
            raise ValueError("Torch OTF expects four (lower, upper) hyperparameter bounds")

        def valid(value):
            return (value is None or
                    (isinstance(value, Real) and not isinstance(value, bool)
                     and np.isfinite(value) and value > 0))

        for low, high in self.bounds:
            if (not valid(low) or not valid(high)
                    or (low is not None and high is not None and low > high)):
                raise ValueError("Torch OTF bounds must be positive, finite, and ordered")
        return [(None if low is None else np.log(low),
                 None if high is None else np.log(high))
                for low, high in self.bounds]

    def train(self, logger_name=None):
        """Optimize amplitude and observation noises, then refit the posterior."""
        if not self._labels or self.max_iterations <= 0:
            return
        bounds = self._log_bounds()

        def objective(log_hyps):
            logs = torch.tensor(log_hyps, dtype=torch.float64, requires_grad=True)
            values = logs.exp()
            posterior = self._fit_with_hyps(values[0], values[1:])
            loss = posterior.negative_log_likelihood
            gradient, = torch.autograd.grad(loss, logs)
            return float(loss.detach()), gradient.detach().numpy().astype(np.float64)

        method = "L-BFGS-B" if bounds is not None else self.opt_method
        result = minimize(objective, np.log(self._hyps), jac=True, method=method,
                          bounds=bounds, options={"maxiter": self.max_iterations})
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise RuntimeError("Torch OTF hyperparameter optimization produced nonfinite values")
        self._hyps = np.exp(result.x)
        self.likelihood_gradient = -np.asarray(result.jac, dtype=np.float64) / self._hyps
        self._refit()

    def predict(self, atoms):
        positions, cell, species, pbc = self._geometry(atoms)
        mean = predict_mean_efs(
            self.descriptor, self._inducing, self._inducing_species,
            self.posterior.alpha, positions, cell, species, pbc=pbc,
            amplitude=self._hyps[0], power=self.power,
            atomic_offsets=self.single_atom_energies, forces=True, stress=True,
        )
        with torch.no_grad():
            if self.variance_type == "local":
                raw = self.descriptor(positions, cell, species, pbc=pbc)
                cross = normalized_dot_product(
                    self._inducing, raw, self._inducing_species, species,
                    amplitude=self._hyps[0], power=self.power,
                )
                prior = normalized_dot_product(
                    raw, raw, species, species, amplitude=self._hyps[0],
                    power=self.power,
                ).diagonal()
                score = self.posterior.local_selection_score(cross, prior, self._hyps[0])
                stds = np.zeros((len(positions), 3), dtype=np.float64)
                stds[:, 0] = score.numpy()
            else:
                layout = ObservationLayout.from_masks(
                    len(positions), force_mask=torch.ones((len(positions), 3), dtype=torch.bool),
                )
                variance = predict_variance_efs(
                    self.posterior, self.descriptor, self._inducing,
                    self._inducing_species, positions, cell, species, pbc=pbc,
                    variance_type=self.variance_type, layout=layout,
                    amplitude=self._hyps[0], power=self.power,
                )
                stds = variance.clamp_min(0).sqrt().reshape(-1, 3).numpy()
        return dict(
            energy=float(mean.energy), forces=mean.forces.numpy(),
            stress=native_stress_to_ase(mean.stress).numpy(), stds=stds,
        )

    def as_dict(self):
        return dict(
            descriptor=dict(n_species=self.descriptor.n_species,
                            n_radial=self.descriptor.n_radial,
                            lmax=self.descriptor.lmax, cutoff=self.descriptor.cutoff),
            species_map=self.species_map, hyps=self._hyps.tolist(), power=self.power,
            variance_type=self.variance_type,
            single_atom_energies=self.single_atom_energies.tolist(),
            energy_training=self.energy_training, force_training=self.force_training,
            stress_training=self.stress_training, jitter=self.jitter,
            max_iterations=self.max_iterations, opt_method=self.opt_method,
            bounds=self.bounds, training_data=self.training_data,
        )

    @classmethod
    def from_dict(cls, state):
        model = cls(
            B2(**state["descriptor"]),
            {int(k): int(v) for k, v in state["species_map"].items()},
            amplitude=state["hyps"][0], sigma_e=state["hyps"][1],
            sigma_f=state["hyps"][2], sigma_s=state["hyps"][3],
            power=state["power"], variance_type=state["variance_type"],
            single_atom_energies=state["single_atom_energies"],
            energy_training=state["energy_training"],
            force_training=state["force_training"], stress_training=state["stress_training"],
            jitter=state["jitter"], max_iterations=state["max_iterations"],
            opt_method=state["opt_method"], bounds=state["bounds"],
        )
        for record in state["training_data"]:
            atoms = Atoms(numbers=record["numbers"], positions=record["positions"],
                          cell=record["cell"], pbc=record["pbc"])
            model.update_db(
                atoms, record["forces"], custom_range=record["selected"],
                energy=record["energy"], stress=record["stress"],
                update_qr=False,
                rel_e_noise=record["relative_noise"][0],
                rel_f_noise=record["relative_noise"][1],
                rel_s_noise=record["relative_noise"][2],
            )
        model._refit()
        return model


class TorchSGPCalculator(Calculator):
    """ASE calculator satisfying the non-mapped OTF prediction interface."""

    implemented_properties = ["energy", "forces", "stress", "stds"]
    backend = "torch_sgp"

    def __init__(self, gp_model):
        super().__init__()
        if not isinstance(gp_model, TorchSGPModel):
            raise TypeError("gp_model must be a TorchSGPModel")
        self.gp_model = gp_model
        self.use_mapping = False

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = self.gp_model.predict(self.atoms)

    def get_uncertainties(self, atoms):
        return self.get_property("stds", atoms)

    def write_model(self, name):
        path = Path(name)
        if path.suffix != ".json":
            path = Path(str(path) + ".json")
        path.write_text(json.dumps(self.as_dict()) + "\n")

    def as_dict(self):
        return {"class": "TorchSGP_Calculator", "gp_model": self.gp_model.as_dict()}

    @classmethod
    def from_file(cls, name):
        state = json.loads(Path(name).read_text())
        if state.get("class") != "TorchSGP_Calculator":
            raise ValueError("file is not a TorchSGP_Calculator model")
        return cls(TorchSGPModel.from_dict(state["gp_model"]))
