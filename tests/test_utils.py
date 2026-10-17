import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
import torch

from mnopt import utils


def test_clipped_inverses_including_active_floor():
    values = torch.tensor([-1., 0., .05, .1, 2.], dtype=torch.float64)
    expected = torch.tensor([10., 10., 10., 10., .5], dtype=torch.float64)
    torch.testing.assert_close(utils.inverse(values, .1), expected)
    torch.testing.assert_close(utils.inverse_sqrt(values, .1), expected.sqrt())
    matrix = torch.diag(values)
    torch.testing.assert_close(utils.matrix_power(matrix, -1, .1), torch.diag(expected))
    torch.testing.assert_close(utils.matrix_power(matrix, -.5, .1), torch.diag(expected.sqrt()))


@pytest.mark.parametrize("case", json.loads((Path(__file__).parent / "data" / "newton_schulz.json").read_text())["cases"])
def test_newton_schulz_matches_archived_gptopt_solver(case):
    dtype = getattr(torch, case["dtype"])
    matrix = torch.tensor(case["input"], dtype=dtype)
    before = matrix.clone()
    expected = torch.tensor(case["output"], dtype=dtype)
    result = utils.matrix_power_newton_schulz(matrix, case["exponent"], case["eps"], case["ns_iters"])
    # Archived outputs can differ by a few ULPs across CPU/PyTorch builds.
    torch.testing.assert_close(result, expected, rtol=8 * torch.finfo(dtype).eps, atol=0)
    torch.testing.assert_close(matrix, before, rtol=0, atol=0)


@pytest.mark.parametrize("exponent", [-1, -.5])
def test_newton_schulz_converges_to_regularized_power(exponent):
    matrix = torch.tensor([[2., .3], [.3, .4]], dtype=torch.float64)
    eps = .1
    shifted = matrix + eps * torch.eye(2, dtype=matrix.dtype)
    expected = utils.matrix_power(shifted, exponent, eps)
    errors = [torch.linalg.norm(utils.matrix_power_newton_schulz(matrix, exponent, eps, steps) - expected)
              for steps in (1, 6, 12)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-12
    assert torch.linalg.norm(expected - utils.matrix_power(matrix, exponent, eps)) > .01


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("exponent", [-1, -.5])
def test_newton_schulz_accumulates_in_float32_for_low_precision(dtype, exponent):
    matrix = torch.tensor([[2., .3], [.3, .4]], dtype=dtype)
    result = utils.matrix_power_newton_schulz(matrix, exponent, 1e-8)
    expected = utils.matrix_power_newton_schulz(matrix.float(), exponent, 1e-8).to(dtype)
    assert result.dtype == dtype
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("matrix,exponent,ns_iters", [
    (torch.ones(2, 3), -1, 6),
    (torch.ones(2, 2), .5, 6),
    (torch.ones(2, 2), -1, 0),
])
def test_invalid_newton_schulz_inputs(matrix, exponent, ns_iters):
    with pytest.raises(ValueError):
        utils.matrix_power_newton_schulz(matrix, exponent, 1e-8, ns_iters)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("eps", [1e-9, 10.])
def test_balance_preserves_covariance_and_guards_traces(dense, eps):
    row = torch.tensor([.2, .3], dtype=torch.float64)
    col = torch.tensor([.7, .8, .9], dtype=torch.float64)
    if dense:
        row, col = torch.diag(row), torch.diag(col)
    before = torch.kron(col, row)
    result_row, result_col = utils.balance(row, col, eps)
    torch.testing.assert_close(torch.kron(result_col, result_row), before)
    if eps == 1e-9:
        row_trace = result_row.trace() if dense else result_row.sum()
        col_trace = result_col.trace() if dense else result_col.sum()
        torch.testing.assert_close(row_trace, col_trace)


@pytest.mark.parametrize("amount", [0., .01, .5, 1.])
def test_shrink_preserves_trace_and_does_not_mutate_input(amount):
    matrix = torch.tensor([[2., .3], [.3, .8]], dtype=torch.float64)
    before = matrix.clone()
    shrunk = utils.shrink(matrix, amount)
    torch.testing.assert_close(shrunk.trace(), matrix.trace())
    torch.testing.assert_close(shrunk, (1 - amount) * matrix + amount * matrix.trace() / 2 * torch.eye(2))
    torch.testing.assert_close(matrix, before)


@pytest.mark.parametrize("u,v,dimension", [(1., 0., 3), (2., -3., 2), (2., 3., 2), (1e-20, 1e20, 2)])
def test_variance_root_matches_high_precision_solution(u, v, dimension):
    with localcontext() as context:
        context.prec = 150
        ud, vd, qd = Decimal(str(u)), Decimal(str(v)), Decimal(dimension)
        expected = (((vd*vd + 4*qd*ud).sqrt() - vd) / (2*qd))**2
    result = utils.variance_root(torch.tensor(u, dtype=torch.float64), torch.tensor(v, dtype=torch.float64), dimension, 1e-120)
    assert result.item() == pytest.approx(float(expected), rel=1e-14)
    torch.testing.assert_close(utils.variance_root(torch.zeros(2), torch.zeros(2), dimension, .1), torch.full((2,), .1))


@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (3, 3)])
def test_svd_polar_full_rank_and_zero(shape):
    matrix = torch.tensor(np.random.default_rng(2).normal(size=shape))
    direction = utils.polar_svd(matrix)
    gram = direction.T @ direction if shape[0] >= shape[1] else direction @ direction.T
    torch.testing.assert_close(gram, torch.eye(min(shape), dtype=torch.float64))
    torch.testing.assert_close((matrix * direction).sum(), torch.linalg.svdvals(matrix).sum())
    torch.testing.assert_close(utils.polar_svd(7 * matrix), direction)
    assert torch.count_nonzero(utils.polar_svd(torch.zeros_like(matrix))) == 0


def test_svd_rank_deficiency_does_not_invent_directions():
    matrix = torch.diag(torch.tensor([2., 0.], dtype=torch.float64))
    torch.testing.assert_close(utils.polar_svd(matrix), torch.diag(torch.tensor([1., 0.], dtype=torch.float64)))


def test_polar_express_matches_archived_paper_solver():
    cases = json.loads((Path(__file__).parent / "data" / "polar_express.json").read_text())["cases"]
    for case in cases:
        matrix = torch.tensor(case["input"], dtype=torch.float32)
        expected = torch.tensor(case["output"], dtype=torch.float32)
        torch.testing.assert_close(utils.polar_express(matrix, case["steps"]), expected, rtol=0, atol=0)
