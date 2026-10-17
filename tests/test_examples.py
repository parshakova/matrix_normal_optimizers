import json
from pathlib import Path

import pytest
import torch

from mnopt import CenteredAdam, MN, MND, MNDMuon
from examples.regression import run_regression


ROOT = Path(__file__).resolve().parents[1]
METHODS = {"centered_adam": CenteredAdam, "mn": MN, "mnd": MND, "mnd_muon": MNDMuon}


@pytest.mark.parametrize("config", sorted((ROOT / "configs").glob("*.json")), ids=lambda path: path.stem)
def test_config_constructs_optimizer_and_runs(config):
    values = json.loads(config.read_text())
    parameter = torch.nn.Parameter(torch.ones(3, 2))
    optimizer = METHODS[values["optimizer"]]([parameter], **values["kwargs"])
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    assert torch.isfinite(parameter).all()


@pytest.mark.parametrize("method", METHODS)
def test_regression_loss_history_matches_returned_model(method):
    result = run_regression(method, steps=80)
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randn(64, 4, generator=generator)
    targets = inputs @ torch.tensor([[.3], [-.7], [.2], [.5]]) + .1
    prediction = torch.nn.functional.linear(inputs, result["weight"], result["bias"])
    final = torch.nn.functional.mse_loss(prediction, targets).item()
    assert len(result["losses"]) == 81
    assert result["losses"][0] == pytest.approx(.25808781385421753)
    assert result["losses"][-1] == pytest.approx(final, rel=1e-6)
    assert final < result["losses"][0]


@pytest.mark.parametrize("method", METHODS.values())
def test_learning_rate_scheduler_changes_step_size(method):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = method([parameter], lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == .0005
    optimizer.zero_grad(set_to_none=True)
    assert parameter.grad is None
