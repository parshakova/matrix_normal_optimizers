from unittest.mock import patch

import numpy as np
import pytest
import torch

from mnopt import CenteredAdam, MN, MND, MNDMuon
from mnopt import utils
from reference import mean_step, muon_factors


SHAPES = [(2, 3), (3, 2), (3, 3), (1, 4), (4, 1)]
VARIANTS = [(True, None), (False, None), (False, .99)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("centered,beta2", VARIANTS)
@pytest.mark.parametrize("eps", [1e-9, .5])
@pytest.mark.parametrize("optimizer_type,inverse_method", [(MND, None), (MN, "eigh"), (MN, "newton_schulz")])
def test_mean_methods_match_printed_algorithms(shape, centered, beta2, eps, optimizer_type, inverse_method):
    rng = np.random.default_rng(15)
    parameter = torch.nn.Parameter(torch.tensor(rng.normal(size=shape), dtype=torch.float64))
    weights = parameter.detach().numpy().copy()
    dense = optimizer_type is MN
    options = dict(beta=.8, eps=eps, lr=.007, weight_decay=.03, centered=centered, beta2=beta2)
    if dense:
        options["shrinkage"] = .13
        options["inverse_method"] = inverse_method
    optimizer = optimizer_type([parameter], **options)
    m, n = shape
    row = np.eye(m) / m if dense else np.ones(m) / m
    col = np.eye(n) / n if dense else np.ones(n) / n
    mean = np.zeros(shape)
    for iteration in range(6):
        gradient = rng.normal(size=shape) if iteration != 2 else np.zeros(shape)
        expected, mean, row, col = mean_step(
            gradient, mean, row, col, beta=.8, eps=eps, centered=centered, beta2=beta2,
            dense=dense, shrinkage=.13 if dense else 0,
            inverse_method=inverse_method,
        )
        weights = (1 - .007 * .03) * weights - .007 * expected
        parameter.grad = torch.tensor(gradient)
        optimizer.step()
        state = optimizer.state[parameter]
        np.testing.assert_allclose(parameter.detach(), weights, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(state["momentum"], mean, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(state["row_factor"], row, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(state["col_factor"], col, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("eps", [1e-9, .5])
@pytest.mark.parametrize("polar_method", ["svd", "polar_express"])
def test_muon_matches_one_sweep_with_previous_direction(shape, eps, polar_method):
    rng = np.random.default_rng(42)
    parameter = torch.nn.Parameter(torch.tensor(rng.normal(size=shape), dtype=torch.float64))
    weights = parameter.detach().numpy().copy()
    optimizer = MNDMuon([parameter], beta=.8, eps=eps, lr=.004, polar_method=polar_method)
    momentum = np.zeros(shape)
    previous_direction = np.zeros(shape)
    row, col = np.ones(shape[0]) / shape[0], np.ones(shape[1]) / shape[1]
    for _ in range(5):
        gradient = rng.normal(size=shape)
        momentum, row, col = muon_factors(gradient, momentum, previous_direction, row, col, .8, eps)
        if polar_method == "svd":
            left, _, right = np.linalg.svd(momentum, full_matrices=False)
            direction = left @ right
        else:
            direction = utils.polar_express(torch.tensor(momentum), 5).numpy()
        weights -= .004 * direction
        parameter.grad = torch.tensor(gradient)
        optimizer.step()
        state = optimizer.state[parameter]
        np.testing.assert_allclose(state["momentum"], momentum, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(state["row_factor"], row, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(state["col_factor"], col, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(parameter.detach(), weights, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(state["direction"], direction, rtol=3e-12, atol=3e-12)
        previous_direction = direction


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)])
@pytest.mark.parametrize("warmup_steps", [0, 2, 50])
def test_centered_adam_bias_correction_and_phase_boundary(shape, warmup_steps):
    rng = np.random.default_rng(9)
    weights = np.asarray(rng.normal(size=shape))
    parameter = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float64))
    optimizer = CenteredAdam([parameter], beta=.95, lr=.003, eps=.01, warmup_steps=warmup_steps)
    mean, moment = np.zeros(shape), np.zeros(shape)
    for step in range(1, 54):
        gradient = np.asarray(rng.normal(size=shape))
        mean = .95 * mean + (1 - .95) * gradient
        moment = .95 * moment + (1 - .95) * gradient**2
        corrected_mean = mean / (1 - .95**step)
        variance = moment / (1 - .95**step)
        if step > warmup_steps:
            variance -= corrected_mean**2
        weights = weights - .003 * corrected_mean / (np.sqrt(np.maximum(variance, 0)) + .01)
        parameter.grad = torch.tensor(gradient)
        optimizer.step()
        np.testing.assert_allclose(parameter.detach(), weights, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("optimizer_type,options", [(MND, {}), (MN, {}), (MN, {"inverse_method": "eigh"}), (MNDMuon, {})])
def test_exactly_one_trace_balance_per_matrix_update(optimizer_type, options):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = optimizer_type([parameter], **options)
    with patch.object(utils, "balance", wraps=utils.balance) as balancing:
        for _ in range(4):
            parameter.grad = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10
            optimizer.step()
        assert balancing.call_count == 4


def test_first_fifty_centered_adam_updates_match_adamw():
    generator = torch.Generator().manual_seed(4)
    parameter = torch.nn.Parameter(torch.randn(3, 2, generator=generator, dtype=torch.float64))
    other = torch.nn.Parameter(parameter.detach().clone())
    centered = CenteredAdam([parameter], weight_decay=.1)
    adam = torch.optim.AdamW([other], lr=.003, betas=(.95, .95), eps=1e-8, weight_decay=.1)
    for _ in range(50):
        gradient = torch.randn(3, 2, generator=generator, dtype=torch.float64)
        parameter.grad, other.grad = gradient.clone(), gradient.clone()
        centered.step()
        adam.step()
        torch.testing.assert_close(parameter, other, atol=1e-14, rtol=1e-14)
    assert centered.param_groups[0]["warmup_step"] == 50


@pytest.mark.parametrize("optimizer_type", [MND, MN, MNDMuon])
def test_auxiliary_adamw_and_explicit_embedding_routing(optimizer_type):
    parameters = [torch.nn.Parameter(torch.randn(shape, dtype=torch.float64)) for shape in [(), (3,), (2, 3), (2, 2, 2)]]
    expected = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    optimizer = optimizer_type([{"params": parameters, "matrix": False}], lr=.002, weight_decay=.02)
    adam = torch.optim.AdamW(expected, lr=.002, betas=(.95, .95), eps=1e-8, weight_decay=.02)
    for _ in range(5):
        for parameter, other in zip(parameters, expected):
            gradient = torch.randn_like(parameter)
            parameter.grad, other.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        adam.step()
    for parameter, other in zip(parameters, expected):
        torch.testing.assert_close(parameter, other, rtol=1e-13, atol=1e-13)


def test_paper_mnd_initialization_and_column_prior_are_distinct_from_dense_mn():
    gradient = torch.tensor([[1., 2.]], dtype=torch.float64)
    mnd_parameter = torch.nn.Parameter(torch.zeros_like(gradient))
    mn_parameter = torch.nn.Parameter(torch.zeros_like(gradient))
    mnd = MND([mnd_parameter], beta=.9)
    mn = MN([mn_parameter], beta=.9, shrinkage=0, inverse_method="eigh")
    mnd_parameter.grad = gradient.clone()
    mn_parameter.grad = gradient.clone()
    mnd.step()
    mn.step()
    mnd_state, mn_state = mnd.state[mnd_parameter], mn.state[mn_parameter]
    mnd_product = torch.outer(mnd_state["row_factor"], mnd_state["col_factor"])
    mn_product = torch.kron(mn_state["col_factor"], mn_state["row_factor"]).diag()
    torch.testing.assert_close(mnd_product.flatten(), torch.tensor([.54, .81], dtype=torch.float64))
    torch.testing.assert_close(mn_product, torch.tensor([.6975, .9675], dtype=torch.float64))
    assert not torch.allclose(mnd_product.flatten(), mn_product)
