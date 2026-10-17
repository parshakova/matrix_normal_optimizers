import copy
import io

import pytest
import torch

from mnopt import CenteredAdam, MN, MND, MNDMuon


METHODS = [CenteredAdam, MND, MN, MNDMuon]


def test_mn_default_and_group_solver_choices_survive_checkpoint():
    parameters = [torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float64)) for _ in range(2)]
    controls = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    options = [{"inverse_method": "newton_schulz", "ns_iters": 3}, {"inverse_method": "eigh", "ns_iters": 9}]
    optimizer = MN([dict(params=[parameter], **option) for parameter, option in zip(parameters, options)])
    defaults = MN([torch.nn.Parameter(torch.ones(1, 1))]).defaults
    assert defaults["inverse_method"] == "newton_schulz"
    assert defaults["ns_iters"] == 6
    separate = [MN([parameter], **option) for parameter, option in zip(controls, options)]
    for index in range(5):
        if index == 2:
            checkpoint = copy.deepcopy(optimizer.state_dict())
            optimizer = MN([{"params": [parameter]} for parameter in parameters])
            optimizer.load_state_dict(checkpoint)
            for group, option in zip(optimizer.param_groups, options):
                assert group["inverse_method"] == option["inverse_method"]
                assert group["ns_iters"] == option["ns_iters"]
        gradient = torch.tensor([[.2, -.1, .4], [.3, .5, -.2]], dtype=torch.float64) * (index + 1)
        for parameter, control in zip(parameters, controls):
            parameter.grad, control.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        for control_optimizer in separate:
            control_optimizer.step()
        for parameter, control in zip(parameters, controls):
            torch.testing.assert_close(parameter, control, rtol=0, atol=0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_checkpoint_resume_preserves_parameters_state_and_precision(method, dtype):
    generator = torch.Generator().manual_seed(71)
    parameter = torch.nn.Parameter(torch.randn(2, 3, generator=generator, dtype=dtype))
    options = dict(lr=.001)
    if method is CenteredAdam:
        options["warmup_steps"] = 3
    uninterrupted = method([parameter], **options)
    gradients = [torch.randn(2, 3, generator=generator, dtype=dtype) for _ in range(7)]
    for gradient in gradients[:3]:
        parameter.grad = gradient.clone()
        uninterrupted.step()
    buffer = io.BytesIO()
    torch.save(uninterrupted.state_dict(), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    other = torch.nn.Parameter(parameter.detach().clone())
    resumed = method([other], **options)
    resumed.load_state_dict(checkpoint)
    for name, value in uninterrupted.state[parameter].items():
        restored = resumed.state[other][name]
        if isinstance(value, torch.Tensor):
            assert restored.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
            torch.testing.assert_close(value, restored, atol=0, rtol=0)
    for gradient in gradients[3:]:
        parameter.grad, other.grad = gradient.clone(), gradient.clone()
        uninterrupted.step()
        resumed.step()
        torch.testing.assert_close(parameter, other, atol=0, rtol=0)
    for name, value in uninterrupted.state[parameter].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, resumed.state[other][name], atol=0, rtol=0)


@pytest.mark.parametrize("method", METHODS)
def test_missing_gradients_skip_state_and_weight_decay(method):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = method([parameter], weight_decay=.4)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.ones_like(parameter))
    assert parameter not in optimizer.state


@pytest.mark.parametrize("method", METHODS)
def test_closure_preserves_gradient_context_and_returns_loss(method):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = method([parameter], lr=.0001)
    calls = []

    def closure():
        calls.append(torch.is_grad_enabled())
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    result = optimizer.step(closure)
    assert calls == [True]
    assert result.item() == 6
    assert parameter.square().sum() < 6


@pytest.mark.parametrize("method", METHODS)
def test_parameter_group_learning_rates_and_weight_decay(method):
    first = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float64))
    second = torch.nn.Parameter(first.detach().clone())
    optimizer = method([{"params": [first], "lr": .001}, {"params": [second], "lr": .002}], weight_decay=.02)
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)
    optimizer.step()
    torch.testing.assert_close(1 - second, 2 * (1 - first), rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("method", METHODS)
def test_zero_gradients_remain_finite_and_leave_parameters_unchanged(method):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = method([parameter])
    for _ in range(5):
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
    torch.testing.assert_close(parameter, torch.ones_like(parameter))
    for value in optimizer.state[parameter].values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("key,value", [("lr", -1), ("eps", 0), ("eps", float("nan")), ("beta", 1), ("beta", 0), ("weight_decay", -1)])
def test_invalid_common_arguments(method, key, value):
    with pytest.raises(ValueError):
        method([torch.nn.Parameter(torch.ones(2, 3))], **{key: value})


@pytest.mark.parametrize("method,options", [
    (CenteredAdam, {"warmup_steps": -1}),
    (CenteredAdam, {"warmup_steps": True}),
    (MND, {"centered": "yes"}),
    (MND, {"beta2": .99}),
    (MN, {"beta2": 2, "centered": False}),
    (MN, {"shrinkage": -1}),
    (MN, {"shrinkage": float("nan")}),
    (MN, {"inverse_method": "svd"}),
    (MN, {"ns_iters": 0}),
    (MN, {"ns_iters": True}),
    (MN, {"ns_iters": 1.5}),
    (MNDMuon, {"polar_method": "nesterov"}),
    (MNDMuon, {"polar_steps": 0}),
    (MNDMuon, {"polar_steps": True}),
])
def test_invalid_method_specific_arguments(method, options):
    with pytest.raises(ValueError):
        method([torch.nn.Parameter(torch.ones(2, 3))], **options)


@pytest.mark.parametrize("method", [MND, MN, MNDMuon])
def test_unsupported_legacy_options_are_not_exposed(method):
    for options in ({"clip_inverse": False}, {"factor_loop": 2}, {"rms_norm": True}, {"nesterov": True}):
        with pytest.raises(TypeError):
            method([torch.nn.Parameter(torch.ones(2, 3))], **options)


@pytest.mark.parametrize("method", METHODS)
def test_sparse_and_complex_parameters_are_rejected(method):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    parameter.grad = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.]), (2, 3))
    with pytest.raises(RuntimeError, match="dense"):
        method([parameter]).step()
    parameter = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.complex64))
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="real"):
        method([parameter]).step()


@pytest.mark.parametrize("method", [MND, MN, MNDMuon])
def test_auto_routing_matches_explicit_groups_and_rejects_invalid_matrix_shapes(method):
    matrix = torch.nn.Parameter(torch.ones(2, 3))
    vector = torch.nn.Parameter(torch.ones(3))
    other_matrix = torch.nn.Parameter(matrix.detach().clone())
    other_vector = torch.nn.Parameter(vector.detach().clone())
    auto = method([matrix, vector])
    explicit = method([{"params": [other_matrix], "matrix": True}, {"params": [other_vector], "matrix": False}])
    for first, second in ((matrix, other_matrix), (vector, other_vector)):
        gradient = torch.randn_like(first)
        first.grad, second.grad = gradient.clone(), gradient.clone()
    auto.step()
    explicit.step()
    torch.testing.assert_close(matrix, other_matrix)
    torch.testing.assert_close(vector, other_vector)
    with pytest.raises(ValueError, match="two-dimensional"):
        method([{"params": [vector], "matrix": True}])


def test_warmup_counter_survives_missing_gradients_and_checkpoint():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = CenteredAdam([parameter], warmup_steps=2)
    optimizer.step()
    optimizer.step()
    assert optimizer.param_groups[0]["warmup_step"] == 2
    resumed = CenteredAdam([parameter], warmup_steps=2)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    parameter.grad = torch.tensor([.5, -1.])
    resumed.step()
    assert resumed.param_groups[0]["warmup_step"] == 3
    assert resumed.state[parameter]["step"] == 1


def test_incompatible_optimizer_checkpoint_is_rejected():
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    with pytest.raises(ValueError, match="different optimizer"):
        MN([parameter]).load_state_dict(MND([parameter]).state_dict())


@pytest.mark.parametrize("method", METHODS)
def test_outer_autocast_does_not_reduce_optimizer_state_precision(method):
    parameter = torch.nn.Parameter(torch.randn(2, 3))
    other = torch.nn.Parameter(parameter.detach().clone())
    optimizer, control = method([parameter]), method([other])
    for _ in range(3):
        gradient = torch.randn_like(parameter)
        parameter.grad, other.grad = gradient.clone(), gradient.clone()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            optimizer.step()
        control.step()
    torch.testing.assert_close(parameter, other, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("method", METHODS)
def test_cuda_updates_are_finite_and_keep_state_on_device(method):
    parameter = torch.nn.Parameter(torch.ones(3, 2, device="cuda", dtype=torch.bfloat16))
    optimizer = method([parameter])
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    assert torch.isfinite(parameter).all()
    for value in optimizer.state[parameter].values():
        if isinstance(value, torch.Tensor):
            assert value.device == parameter.device
            assert value.dtype == torch.float32


@pytest.mark.compile
@pytest.mark.parametrize("method,options", [(CenteredAdam, {"warmup_steps": 1}), (MND, {}), (MN, {}), (MN, {"inverse_method": "eigh"}), (MNDMuon, {})])
def test_compiled_kernel_matches_eager(method, options):
    parameter = torch.nn.Parameter(torch.tensor([[.4, -.7], [1., .2]]))
    other = torch.nn.Parameter(parameter.detach().clone())
    compiled = method([parameter], compile=True, **options)
    eager = method([other], **options)
    for index in range(12):
        gradient = torch.tensor([[.3, .4], [-.2, .1]]) if index % 2 else torch.tensor([[-.1, .2], [.4, .5]])
        parameter.grad, other.grad = gradient.clone(), gradient.clone()
        compiled.step()
        eager.step()
    torch.testing.assert_close(parameter, other, rtol=3e-5, atol=3e-6)
