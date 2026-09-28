"""Masked observation operators: native parity and independent derivatives."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from flare.tensor import (
    B2,
    StructureBatch,
    assemble_cached_grouped_lambda_observation_covariance,
    build_grouped_lambda_cache,
    cached_grouped_lambda_observation_covariance,
    fit_sparse_gp,
    normalized_dot_product,
)
from flare.tensor.neighbors import neighbor_list
from flare.tensor.observations import (
    ObservationLayout,
    ase_stress_to_native,
    assemble_cached_q_observation_covariance,
    assemble_observation_covariance,
    build_q_cache,
    cached_q_observation_covariance,
    inducing_observation_covariance,
    native_stress_to_ase,
)


DATA = Path(__file__).parent / "data"
CASES = ("molecule", "crystal", "binary", "missing_species", "isolated")


@pytest.fixture(scope="module")
def reference():
    with np.load(DATA / "b2_reference.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return arrays, json.loads((DATA / "b2_reference.json").read_text())


def system(reference, name="binary", dtype=torch.float64, device="cpu"):
    arrays, metadata = reference
    settings = metadata["settings"]
    descriptor = B2(settings["n_species"], settings["n_radial"], settings["l_max"], settings["cutoff"])
    tensor = lambda value: torch.as_tensor(value, dtype=dtype, device=device)
    z = tensor(arrays["inducing_raw_b2"])
    zs = torch.as_tensor(arrays["inducing_species"], dtype=torch.long, device=device)
    positions, cell = tensor(arrays[name + "__positions"]), tensor(arrays[name + "__cell"])
    species = torch.as_tensor(arrays[name + "__species"], dtype=torch.long, device=device)
    pbc = torch.as_tensor(arrays[name + "__pbc"], dtype=torch.bool, device=device)
    return descriptor, z, zs, positions, cell, species, pbc, tensor(arrays["hyperparameters"][0])


def evaluate(entry, layout, **kwargs):
    descriptor, z, zs, positions, cell, species, pbc, amplitude = entry
    return inducing_observation_covariance(
        descriptor, z, zs, positions, cell, species, layout,
        pbc=pbc, amplitude=amplitude, **kwargs,
    )


def test_layout_and_partial_labels_offsets_gradients():
    layout = ObservationLayout.from_masks(
        2, energy=True, force_mask=[[True, False, True], [False, True, False]],
        stress_mask=[False, True, False, False, False, True],
    )
    assert layout.n_atoms == 2 and layout.size == 6
    assert layout.full_indices.tolist() == [0, 1, 3, 5, 8, 12]
    assert layout.noise_kind.tolist() == [0, 1, 1, 1, 2, 2]
    energy = torch.tensor(11.0, dtype=torch.float64, requires_grad=True)
    forces = torch.tensor([[1., float("nan"), 2.], [float("nan"), 3., float("nan")]], dtype=torch.float64)
    stress = torch.tensor([float("nan"), 4., float("nan"), float("nan"), float("nan"), 5.], dtype=torch.float64)
    offsets = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    packed = layout.pack_labels(energy, forces, stress, torch.tensor([1, 0]), offsets)
    torch.testing.assert_close(packed, torch.tensor([6., 1., 2., 3., 4., 5.], dtype=torch.float64))
    energy_grad, offset_grad = torch.autograd.grad(packed.sum(), (energy, offsets))
    torch.testing.assert_close(energy_grad, torch.ones_like(energy))
    torch.testing.assert_close(offset_grad, -torch.ones_like(offsets))
    forces[0, 0] = float("nan")
    with pytest.raises(ValueError, match="selected forces"):
        layout.pack_labels(energy, forces, stress)


def test_energy_only_and_empty_layouts():
    assert ObservationLayout.energy_only(2).full_indices.tolist() == [0]
    assert ObservationLayout.full(2).size == 13
    layout = ObservationLayout.from_masks(2)
    assert layout.size == 0 and layout.full_indices.shape == (0,)
    assert layout.pack_labels().shape == (0,)
    # Unselected kinds impose no requirement on absent/nonfinite labels/noises.
    assert layout.noise_variance(torch.full((3,), float("nan"))).shape == (0,)
    torch.testing.assert_close(ObservationLayout.energy_only(2).pack_labels(3.0, forces=float("nan")), torch.tensor([3.0]))
    with pytest.raises(ValueError, match="selected energy"):
        ObservationLayout.energy_only(2).pack_labels()


@pytest.mark.parametrize("factory", [
    lambda: ObservationLayout.from_masks(-1),
    lambda: ObservationLayout.from_masks(1, force_mask=[[1, 0, 0]]),
    lambda: ObservationLayout.from_masks(1, force_mask=[[True, False, True], [True, False, True]]),
    lambda: ObservationLayout.from_masks(1, stress_mask=[True]),
    lambda: ObservationLayout.from_masks(1, energy=1),
])
def test_invalid_layouts(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_noise_kind_scaling_and_gradient():
    layout = ObservationLayout.from_masks(1, energy=True, force_mask=[[False, True, True]], stress_mask=[True] * 6)
    std = torch.tensor([0.4, 0.2, 0.03], dtype=torch.float64, requires_grad=True)
    relative = torch.tensor([1.3, 0.8, 2.1], dtype=torch.float64, requires_grad=True)
    variance = layout.noise_variance(std, relative)
    expected = ((std * relative).square())[torch.tensor([0, 1, 1, 2, 2, 2, 2, 2, 2])]
    torch.testing.assert_close(variance, expected)
    gradients = torch.autograd.grad(variance.sum(), (std, relative))
    counts = torch.tensor([1., 2., 6.], dtype=torch.float64)
    torch.testing.assert_close(gradients[0], counts * 2 * std * relative.square())
    torch.testing.assert_close(gradients[1], counts * 2 * relative * std.square())
    only_energy = ObservationLayout.energy_only(1)
    torch.testing.assert_close(only_energy.noise_variance(torch.tensor([1., float("nan"), 0.])), torch.ones(1))
    for bad in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive"):
            only_energy.noise_variance(torch.tensor([bad, 1., 1.]))
        with pytest.raises(ValueError, match="positive"):
            only_energy.noise_variance(torch.ones(3), torch.tensor([bad, 1., 1.]))


def test_stress_conversion_preserves_graph_and_has_no_shear_factor():
    values = torch.arange(12, dtype=torch.float64).reshape(2, 6).requires_grad_()
    ase = native_stress_to_ase(values)
    torch.testing.assert_close(ase[0], torch.tensor([0., -3., -5., -4., -2., -1.], dtype=torch.float64))
    torch.testing.assert_close(ase_stress_to_native(ase), values)
    torch.testing.assert_close(torch.autograd.grad(ase.sum(), values)[0], -torch.ones_like(values))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("chunk_size", [1, 3, 32])
def test_native_full_observation_parity(reference, case, chunk_size):
    entry = system(reference, case)
    actual = evaluate(entry, ObservationLayout.full(len(entry[3])), chunk_size=chunk_size)
    expected = torch.as_tensor(reference[0][case + "__Kz_efs"])
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("chunk_size", [1, 2, 32])
@pytest.mark.parametrize("assembly", ["q", "lambda"])
def test_streamed_q_matches_direct_autograd_and_native(reference, case, chunk_size, assembly):
    entry = system(reference, case)
    layout = ObservationLayout.full(len(entry[3]))
    direct = evaluate(entry, layout, chunk_size=chunk_size, assembly="autograd")
    streamed = evaluate(entry, layout, chunk_size=chunk_size, assembly=assembly)
    expected = torch.as_tensor(reference[0][case + "__Kz_efs"])
    torch.testing.assert_close(streamed, direct, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(streamed, expected, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("chunk_size", [1, 2, 32])
def test_cached_q_matches_direct_autograd_and_native(reference, case, chunk_size):
    entry = system(reference, case)
    descriptor, z, zs, positions, cell, species, pbc, amplitude = entry
    cache = build_q_cache(descriptor, positions, cell, species, pbc=pbc)
    layout = ObservationLayout.full(len(species))
    direct = evaluate(entry, layout, chunk_size=chunk_size, assembly="autograd")
    actual = cached_q_observation_covariance(
        cache, z, zs, layout, amplitude=amplitude, chunk_size=chunk_size,
    )
    expected = torch.as_tensor(reference[0][case + "__Kz_efs"])
    assert not cache.values.requires_grad and not cache.q.requires_grad
    torch.testing.assert_close(actual, direct, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("case", CASES)
def test_cached_grouped_lambda_matches_native_and_cached_q(reference, case):
    descriptor, z, zs, positions, cell, species, pbc, amplitude = system(reference, case)
    cache = build_grouped_lambda_cache(descriptor, positions, cell, species,
                                       pbc=pbc, center_tile=3)
    q_cache = build_q_cache(descriptor, positions, cell, species, pbc=pbc)
    layout = ObservationLayout.full(len(species))
    actual = cached_grouped_lambda_observation_covariance(
        cache, z, zs, layout, amplitude=amplitude, sparse_chunk=2,
    )
    expected = torch.as_tensor(reference[0][case + "__Kz_efs"])
    q_block = cached_q_observation_covariance(
        q_cache, z, zs, layout, amplitude=amplitude, chunk_size=2,
    )
    assert not cache.values.requires_grad and not cache.coefficients.requires_grad
    assert all(not group.derivatives.requires_grad
               for tile in cache.tiles for group in tile.groups)
    torch.testing.assert_close(actual, q_block, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("power", [1, 2, 3])
def test_cached_grouped_lambda_masks_and_kernel_gradients(reference, power):
    descriptor, z, zs, positions, cell, species, pbc, amplitude = system(reference)
    positions = positions.clone().requires_grad_()
    cell = cell.clone().requires_grad_()
    cache = build_grouped_lambda_cache(descriptor, positions, cell, species,
                                       pbc=pbc, center_tile=2)
    q_cache = build_q_cache(descriptor, positions, cell, species, pbc=pbc)
    layout = ObservationLayout.from_masks(
        len(species), energy=True,
        force_mask=torch.arange(3 * len(species)).reshape(-1, 3) % 3 == 0,
        stress_mask=[True, False, True, False, False, True],
    )
    inducing = z.clone().requires_grad_()
    signal = amplitude.clone().requires_grad_()
    actual = cached_grouped_lambda_observation_covariance(
        cache, inducing, zs, layout, amplitude=signal, power=power,
        sparse_chunk=2,
    )
    expected = cached_q_observation_covariance(
        q_cache, inducing, zs, layout, amplitude=signal, power=power,
        chunk_size=2,
    )
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    weights = torch.linspace(0.2, 1.1, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    gradients = torch.autograd.grad(
        (actual * weights).sum(), (inducing, signal, positions, cell),
        allow_unused=True,
    )
    q_gradients = torch.autograd.grad((expected * weights).sum(), (inducing, signal))
    for result, target in zip(gradients[:2], q_gradients):
        torch.testing.assert_close(result, target, atol=2e-9, rtol=2e-9)
    assert gradients[2:] == (None, None)


def test_cached_grouped_lambda_ragged_and_empty_edges(reference):
    descriptor, z, zs, positions, cell, species, pbc, amplitude = system(reference)
    edges = neighbor_list(positions, cell, descriptor.cutoff, pbc)
    keep = ((edges[0] % 3 != 0)
            & (torch.arange(len(edges[0]), device=positions.device) % 5 != 0))
    for selected in (tuple(edge[keep] for edge in edges),
                     tuple(edge[:0] for edge in edges)):
        cache = build_grouped_lambda_cache(
            descriptor, positions, cell, species, pbc=pbc, edges=selected,
            center_tile=2,
        )
        q_cache = build_q_cache(descriptor, positions, cell, species,
                                pbc=pbc, edges=selected)
        layout = ObservationLayout.full(len(species))
        actual = cached_grouped_lambda_observation_covariance(
            cache, z, zs, layout, amplitude=amplitude, sparse_chunk=2,
        )
        expected = cached_q_observation_covariance(
            q_cache, z, zs, layout, amplitude=amplitude, chunk_size=2,
        )
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


def test_cached_grouped_lambda_multi_structure_and_validation(reference):
    entries = [system(reference, name) for name in ("molecule", "binary")]
    z, zs, amplitude = entries[0][1], entries[0][2], entries[0][-1]
    caches = [build_grouped_lambda_cache(entry[0], entry[3], entry[4], entry[5],
                                         pbc=entry[6], center_tile=2)
              for entry in entries]
    layouts = [ObservationLayout.full(len(entry[5])) for entry in entries]
    actual = assemble_cached_grouped_lambda_observation_covariance(
        caches, z, zs, layouts, amplitude=amplitude, sparse_chunk=2,
    )
    torch.testing.assert_close(actual, torch.as_tensor(reference[0]["Kzy"]),
                               atol=2e-9, rtol=2e-9)
    assert assemble_cached_grouped_lambda_observation_covariance(
        [], z, zs, [], amplitude=amplitude,
    ).shape == (len(z), 0)
    with pytest.raises(ValueError, match="one observation layout"):
        assemble_cached_grouped_lambda_observation_covariance(caches, z, zs, layouts[:1])
    with pytest.raises(ValueError, match="sparse_chunk"):
        cached_grouped_lambda_observation_covariance(caches[0], z, zs, layouts[0],
                                                      sparse_chunk=0)
    with pytest.raises(ValueError, match="cached dtype"):
        cached_grouped_lambda_observation_covariance(caches[0], z.float(), zs,
                                                      layouts[0])


def test_cached_grouped_lambda_float32(reference):
    descriptor, z, zs, positions, cell, species, pbc, amplitude = system(
        reference, dtype=torch.float32,
    )
    cache = build_grouped_lambda_cache(descriptor, positions, cell, species,
                                       pbc=pbc, center_tile=2)
    actual = cached_grouped_lambda_observation_covariance(
        cache, z, zs, ObservationLayout.full(len(species)),
        amplitude=amplitude, sparse_chunk=2,
    )
    expected = torch.as_tensor(reference[0]["binary__Kz_efs"], dtype=torch.float32)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


def test_cached_grouped_lambda_fit_gradient_matches_cached_q(reference):
    entries = [system(reference, name) for name in ("molecule", "binary")]
    z = entries[0][1].clone().requires_grad_()
    zs = entries[0][2]
    amplitude = entries[0][-1].clone().requires_grad_()
    layouts = [ObservationLayout.full(len(entry[5])) for entry in entries]
    lambda_caches = [build_grouped_lambda_cache(
        entry[0], entry[3], entry[4], entry[5], pbc=entry[6], center_tile=2,
    ) for entry in entries]
    q_caches = [build_q_cache(entry[0], entry[3], entry[4], entry[5],
                              pbc=entry[6]) for entry in entries]
    Kzz = normalized_dot_product(z, z, zs, zs, amplitude=amplitude)
    lambda_block = assemble_cached_grouped_lambda_observation_covariance(
        lambda_caches, z, zs, layouts, amplitude=amplitude, sparse_chunk=2,
    )
    q_block = assemble_cached_q_observation_covariance(
        q_caches, z, zs, layouts, amplitude=amplitude, chunk_size=2,
    )
    y = torch.as_tensor(reference[0]["y"])
    noise = torch.as_tensor(reference[0]["noise_variance"])
    jitter = float(reference[0]["jitter"])
    lambda_fit = fit_sparse_gp(Kzz, lambda_block, y, noise, jitter=jitter)
    q_fit = fit_sparse_gp(Kzz, q_block, y, noise, jitter=jitter)
    torch.testing.assert_close(lambda_fit.alpha, q_fit.alpha, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(lambda_fit.negative_log_likelihood,
                               q_fit.negative_log_likelihood, atol=2e-9, rtol=2e-9)
    gradients = torch.autograd.grad(lambda_fit.negative_log_likelihood,
                                    (z, amplitude), retain_graph=True)
    expected = torch.autograd.grad(q_fit.negative_log_likelihood,
                                   (z, amplitude))
    for actual, target in zip(gradients, expected):
        torch.testing.assert_close(actual, target, atol=2e-8, rtol=2e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cached_grouped_lambda_cuda_parity(reference):
    descriptor, z, zs, positions, cell, species, pbc, amplitude = system(
        reference, device="cuda",
    )
    cache = build_grouped_lambda_cache(descriptor, positions, cell, species,
                                       pbc=pbc, center_tile=2)
    layout = ObservationLayout.full(len(species), device="cuda")
    actual = cached_grouped_lambda_observation_covariance(
        cache, z, zs, layout, amplitude=amplitude, sparse_chunk=2,
    )
    expected = torch.as_tensor(reference[0]["binary__Kz_efs"], device="cuda")
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


def test_cached_q_ragged_assembly_and_kernel_gradients(reference):
    entries = [system(reference, name) for name in ("molecule", "binary")]
    descriptor, z, zs, *_, amplitude = entries[0]
    layouts = [ObservationLayout.full(len(entry[3])) for entry in entries]
    caches = [
        build_q_cache(entry[0], entry[3], entry[4], entry[5], pbc=entry[6])
        for entry in entries
    ]
    actual = assemble_cached_q_observation_covariance(
        caches, z, zs, layouts, amplitude=amplitude, chunk_size=2,
    )
    torch.testing.assert_close(actual, torch.as_tensor(reference[0]["Kzy"]), atol=2e-9, rtol=2e-9)

    trainable_z = z.clone().requires_grad_()
    trainable_amplitude = amplitude.clone().requires_grad_()
    weights = torch.linspace(0.2, 1.1, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    cached = cached_q_observation_covariance(
        caches[0], trainable_z, zs, layouts[0],
        amplitude=trainable_amplitude, chunk_size=2,
    )
    weights = weights[:, :cached.shape[1]]
    cached_gradients = torch.autograd.grad((cached * weights).sum(), (trainable_z, trainable_amplitude))
    direct_entry = list(entries[0])
    direct_entry[1] = trainable_z
    direct_entry[-1] = trainable_amplitude
    direct = evaluate(direct_entry, layouts[0], chunk_size=2, assembly="q")
    direct_gradients = torch.autograd.grad((direct * weights).sum(), (trainable_z, trainable_amplitude))
    torch.testing.assert_close(cached_gradients[0], direct_gradients[0], atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(cached_gradients[1], direct_gradients[1], atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("power", [1, 3])
def test_cached_q_matches_transient_q_for_other_kernel_powers(reference, power):
    entry = list(system(reference, "binary"))
    entry[-1] = torch.tensor(0.8, dtype=entry[3].dtype)
    cache = build_q_cache(entry[0], entry[3], entry[4], entry[5], pbc=entry[6])
    layout = ObservationLayout.full(len(entry[3]))
    transient = evaluate(entry, layout, power=power, chunk_size=2, assembly="q")
    cached = cached_q_observation_covariance(
        cache, entry[1], entry[2], layout, amplitude=entry[-1], power=power, chunk_size=2,
    )
    torch.testing.assert_close(cached, transient, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("power", [1, 3])
@pytest.mark.parametrize("assembly", ["q", "lambda"])
def test_streamed_q_matches_direct_for_other_kernel_powers(reference, power, assembly):
    entry = list(system(reference, "binary"))
    entry[-1] = torch.tensor(0.8, dtype=entry[3].dtype)
    layout = ObservationLayout.full(len(entry[3]))
    direct = evaluate(entry, layout, power=power, chunk_size=2, assembly="autograd")
    streamed = evaluate(entry, layout, power=power, chunk_size=2, assembly=assembly)
    torch.testing.assert_close(streamed, direct, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("kind", ["energy", "force", "stress", "mixed"])
@pytest.mark.parametrize("assembly", ["autograd", "q", "lambda"])
def test_exact_masked_native_subsets(reference, kind, assembly):
    entry = system(reference)
    n = len(entry[3])
    force_mask = torch.arange(3 * n).reshape(n, 3) % 2 == 0
    stress_mask = torch.tensor([True, False, True, False, True, False])
    layout = ObservationLayout.from_masks(
        n, energy=kind in ("energy", "mixed"),
        force_mask=force_mask if kind in ("force", "mixed") else None,
        stress_mask=stress_mask if kind in ("stress", "mixed") else None,
    )
    actual = evaluate(entry, layout, chunk_size=2, assembly=assembly)
    expected = torch.as_tensor(reference[0]["binary__Kz_efs"])[:, layout.full_indices]
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("assembly", ["autograd", "q", "lambda"])
def test_ragged_training_matrix_and_empty_batch(reference, assembly):
    entries = [system(reference, name) for name in ("molecule", "binary")]
    sizes = [len(entry[3]) for entry in entries]
    batch = StructureBatch(
        torch.cat([entry[3] for entry in entries]), torch.cat([entry[5] for entry in entries]),
        torch.stack([entry[4] for entry in entries]), torch.stack([entry[6] for entry in entries]),
        torch.tensor([0, sizes[0], sum(sizes)]),
    )
    descriptor, z, zs, *_, amplitude = entries[0]
    layouts = [ObservationLayout.full(n) for n in sizes]
    actual = assemble_observation_covariance(
        descriptor, z, zs, batch, layouts, amplitude=amplitude, chunk_size=2,
        assembly=assembly,
    )
    torch.testing.assert_close(actual, torch.as_tensor(reference[0]["Kzy"]), atol=2e-9, rtol=2e-9)
    with pytest.raises(ValueError, match="one observation"):
        assemble_observation_covariance(
            descriptor, z, zs, batch, layouts[:1], assembly=assembly,
        )
    empty_batch = StructureBatch(batch.positions[:0], batch.species[:0], batch.cells[:0], batch.pbc[:0], torch.zeros(1, dtype=torch.long))
    assert assemble_observation_covariance(
        descriptor, z, zs, empty_batch, [], assembly=assembly,
    ).shape == (len(z), 0)


def test_zero_volume_allowed_only_without_stress_and_empty_axes(reference):
    entry = list(system(reference, "molecule"))
    entry[4] = torch.zeros_like(entry[4])
    entry[6] = False
    layout = ObservationLayout.from_masks(len(entry[3]), energy=True, force_mask=torch.ones_like(entry[3], dtype=torch.bool))
    assert torch.isfinite(evaluate(entry, layout)).all()
    with pytest.raises(ValueError, match="positive cell volume"):
        evaluate(entry, ObservationLayout.full(len(entry[3])))
    assert evaluate(entry, ObservationLayout.from_masks(len(entry[3]))).shape == (len(entry[1]), 0)
    entry[1] = entry[1][:0].clone().requires_grad_()
    entry[2] = entry[2][:0]
    empty = evaluate(entry, layout)
    assert empty.shape == (0, layout.size)
    assert torch.autograd.grad(empty.sum(), entry[1])[0].shape == entry[1].shape


def test_coincident_atoms_rejected_by_operator(reference):
    entry = list(system(reference, "molecule"))
    entry[3] = entry[3].clone()
    entry[3][1] = entry[3][0]
    with pytest.raises(ValueError, match="coincident"):
        evaluate(entry, ObservationLayout.energy_only(len(entry[3])))


def test_observation_derivatives_against_coordinate_and_strain_finite_difference(reference):
    entry = list(system(reference))
    entry[1], entry[2] = entry[1][:2], entry[2][:2]
    n = len(entry[3])
    energy_layout, full = ObservationLayout.energy_only(n), ObservationLayout.full(n)
    actual = evaluate(entry, full, chunk_size=1)
    step = 1e-5
    displaced = torch.zeros_like(entry[3])
    displaced[1, 2] = step
    plus, minus = entry.copy(), entry.copy()
    plus[3], minus[3] = entry[3] + displaced, entry[3] - displaced
    force = -(evaluate(plus, energy_layout) - evaluate(minus, energy_layout))[:, 0] / (2 * step)
    torch.testing.assert_close(actual[:, 1 + 3 + 2], force, atol=2e-8, rtol=2e-7)
    strain = torch.zeros_like(entry[4])
    strain[0, 1] = step
    identity = torch.eye(3, dtype=entry[3].dtype)
    plus[3], plus[4] = entry[3] @ (identity + strain).T, entry[4] @ (identity + strain).T
    minus[3], minus[4] = entry[3] @ (identity - strain).T, entry[4] @ (identity - strain).T
    stress = -(evaluate(plus, energy_layout) - evaluate(minus, energy_layout))[:, 0] / (2 * step * torch.linalg.det(entry[4]).abs())
    torch.testing.assert_close(actual[:, 1 + 3 * n + 1], stress, atol=2e-9, rtol=2e-7)


@pytest.mark.parametrize("chunk_size", [1, 32])
@pytest.mark.parametrize("assembly", ["autograd", "q", "lambda"])
def test_mixed_parameter_derivatives_survive_force_and_stress_operators(reference, chunk_size, assembly):
    entry = list(system(reference))
    entry[1] = entry[1][:2].clone().requires_grad_()
    entry[2] = entry[2][:2]
    entry[3] = entry[3].clone().requires_grad_()
    entry[7] = entry[7].clone().requires_grad_()
    layout = ObservationLayout.full(len(entry[3]))
    matrix = evaluate(entry, layout, chunk_size=chunk_size, assembly=assembly)
    weights = torch.linspace(0.2, 1.1, matrix.numel(), dtype=matrix.dtype).reshape_as(matrix)
    gradients = torch.autograd.grad((matrix * weights).sum(), (entry[1], entry[3], entry[7]))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    torch.testing.assert_close(gradients[2], (matrix * weights).sum() * 2 / entry[7], atol=2e-10, rtol=2e-10)
    # A directional finite difference probes force/reference mixed derivatives.
    generator = torch.Generator().manual_seed(782)
    direction = torch.randn(entry[1].shape, dtype=entry[1].dtype, generator=generator)
    direction = direction / direction.norm()
    step = 1e-5
    plus, minus = entry.copy(), entry.copy()
    plus[1], minus[1] = entry[1].detach() + step * direction, entry[1].detach() - step * direction
    numerical = ((
        evaluate(plus, layout, chunk_size=chunk_size, assembly=assembly)
        - evaluate(minus, layout, chunk_size=chunk_size, assembly=assembly)
    ) * weights).sum() / (2 * step)
    torch.testing.assert_close((gradients[0] * direction).sum(), numerical, atol=2e-8, rtol=2e-7)
    # A coordinate direction additionally probes the differentiated force graph.
    displacement = torch.zeros_like(entry[3])
    displacement[0, 1] = step
    plus, minus = entry.copy(), entry.copy()
    plus[3], minus[3] = entry[3].detach() + displacement, entry[3].detach() - displacement
    numerical = ((
        evaluate(plus, layout, chunk_size=chunk_size, assembly=assembly)
        - evaluate(minus, layout, chunk_size=chunk_size, assembly=assembly)
    ) * weights).sum() / (2 * step)
    torch.testing.assert_close(gradients[1][0, 1], numerical, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("assembly", ["autograd", "q", "lambda"])
def test_float32_and_empty_structure(reference, assembly):
    entry = system(reference, dtype=torch.float32)
    actual = evaluate(entry, ObservationLayout.full(len(entry[3])), chunk_size=2, assembly=assembly)
    expected = torch.as_tensor(reference[0]["binary__Kz_efs"], dtype=torch.float32)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    empty = list(entry)
    empty[3], empty[5] = entry[3][:0], entry[5][:0]
    actual = evaluate(empty, ObservationLayout.full(0))
    torch.testing.assert_close(actual, torch.zeros_like(actual))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("assembly", ["autograd", "q", "lambda"])
def test_cuda_observation_parity(reference, assembly):
    entry = system(reference, device="cuda")
    actual = evaluate(
        entry, ObservationLayout.full(len(entry[3]), device="cuda"), chunk_size=2,
        assembly=assembly,
    )
    expected = torch.as_tensor(reference[0]["binary__Kz_efs"], device="cuda")
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
