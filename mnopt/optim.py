import math

import torch
from torch.optim import Optimizer

from . import utils


def _state_dtype(parameter):
    return torch.float64 if parameter.dtype == torch.float64 else torch.float32


def _positive(group, key, allow_zero=False):
    value = group[key]
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{key} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def _gradient(parameter):
    if parameter.grad.layout != torch.strided:
        raise RuntimeError("mnopt requires dense gradients")
    if parameter.is_complex():
        raise RuntimeError("mnopt requires real parameters")
    if not parameter.numel():
        raise ValueError("mnopt requires nonempty parameters")
    return parameter.grad.to(dtype=_state_dtype(parameter))


class _PaperOptimizer(Optimizer):
    def _validate(self, group):
        _positive(group, "lr", allow_zero=True)
        _positive(group, "weight_decay", allow_zero=True)
        _positive(group, "eps")
        if not 0 < group["beta"] < 1:
            raise ValueError("beta must be strictly between 0 and 1")

    def state_dict(self):
        result = super().state_dict()
        result["mnopt_method"] = type(self).__name__
        return result

    def load_state_dict(self, state_dict):
        if state_dict.get("mnopt_method") != type(self).__name__:
            raise ValueError("Cannot load a checkpoint from a different optimizer")
        super().load_state_dict(state_dict)
        # PyTorch otherwise casts floating optimizer state to low-precision parameters.
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups):
            self._validate(group)
            for saved_id, parameter in zip(saved_group["params"], group["params"]):
                for name, value in state_dict["state"].get(saved_id, {}).items():
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        self.state[parameter][name] = value.detach().to(
                            device=parameter.device, dtype=_state_dtype(parameter)
                        ).clone()


class CenteredAdam(_PaperOptimizer):
    """Bias-corrected Centered Adam with the paper's initial AdamW phase."""

    def __init__(self, params, lr=0.003, beta=0.95, eps=1e-8, weight_decay=0.0,
                 warmup_steps=50, compile=False):
        if type(compile) is not bool:
            raise ValueError("compile must be a boolean")
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay,
                        warmup_steps=warmup_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._validate(group)
        self._adam = torch.compile(utils.adam_direction) if compile else utils.adam_direction
        self._compiled = compile

    def _validate(self, group):
        super()._validate(group)
        if type(group["warmup_steps"]) is not int or group["warmup_steps"] < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._validate(group)
            group["warmup_step"] = group.get("warmup_step", 0) + 1
            centered = group["warmup_step"] > group["warmup_steps"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = _gradient(parameter)
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(gradient)
                    state["exp_avg_sq"] = torch.zeros_like(gradient)
                state["step"] += 1
                step = gradient.new_tensor(state["step"], dtype=torch.float64) if self._compiled else state["step"]
                with torch.autocast(device_type=parameter.device.type, enabled=False):
                    direction, mean, moment = self._adam(
                        gradient, state["exp_avg"], state["exp_avg_sq"],
                        group["beta"], group["beta"], group["eps"], step, centered,
                    )
                state["exp_avg"].copy_(mean)
                state["exp_avg_sq"].copy_(moment)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(direction.to(parameter.dtype), alpha=-group["lr"])
        return loss


class _MatrixOptimizer(_PaperOptimizer):
    def __init__(self, params, defaults, kernel, compile):
        if type(compile) is not bool:
            raise ValueError("compile must be a boolean")
        defaults = dict(defaults, matrix=None, adam_betas=(0.95, 0.95), adam_eps=1e-8)
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._validate(group)
            for parameter in group["params"]:
                if group["matrix"] is True and parameter.ndim != 2:
                    raise ValueError("matrix=True requires two-dimensional parameters")
        self._kernel = torch.compile(kernel) if compile else kernel

    def _validate(self, group):
        super()._validate(group)
        if group["matrix"] is not None and type(group["matrix"]) is not bool:
            raise ValueError("matrix must be None, True, or False")
        betas = group["adam_betas"]
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("adam_betas must contain two values in [0, 1)")
        _positive(group, "adam_eps")

    def _init_matrix_state(self, state, gradient):
        m, n = gradient.shape
        state["momentum"] = torch.zeros_like(gradient)
        if isinstance(self, MN):
            state["row_factor"] = torch.eye(m, dtype=gradient.dtype, device=gradient.device) / m
            state["col_factor"] = torch.eye(n, dtype=gradient.dtype, device=gradient.device) / n
        else:
            state["row_factor"] = gradient.new_full((m,), 1 / m)
            state["col_factor"] = gradient.new_full((n,), 1 / n)
        if isinstance(self, MNDMuon):
            state["direction"] = torch.zeros_like(gradient)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._validate(group)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = _gradient(parameter)
                matrix = parameter.ndim == 2 if group["matrix"] is None else group["matrix"]
                if matrix and parameter.ndim != 2:
                    raise ValueError("matrix=True requires two-dimensional parameters")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["matrix"] = matrix
                    if matrix:
                        self._init_matrix_state(state, gradient)
                    else:
                        state["exp_avg"] = torch.zeros_like(gradient)
                        state["exp_avg_sq"] = torch.zeros_like(gradient)
                if state["matrix"] != matrix:
                    raise ValueError("Cannot change parameter routing after initialization")
                state["step"] += 1
                with torch.autocast(device_type=parameter.device.type, enabled=False):
                    if matrix:
                        direction = self._matrix_direction(gradient, state, group)
                    else:
                        beta1, beta2 = group["adam_betas"]
                        direction, mean, moment = utils.adam_direction(
                            gradient, state["exp_avg"], state["exp_avg_sq"],
                            beta1, beta2, group["adam_eps"], state["step"], False,
                        )
                        state["exp_avg"].copy_(mean)
                        state["exp_avg_sq"].copy_(moment)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(direction.to(parameter.dtype), alpha=-group["lr"])
        return loss


class _MeanOptimizer(_MatrixOptimizer):
    def _validate(self, group):
        super()._validate(group)
        if type(group["centered"]) is not bool:
            raise ValueError("centered must be a boolean")
        beta2 = group["beta2"]
        if beta2 is not None:
            if not 0 < beta2 < 1:
                raise ValueError("beta2 must be strictly between 0 and 1")
            if group["centered"]:
                raise ValueError("The paper's beta2 variant requires centered=False")

    def _matrix_direction(self, gradient, state, group):
        beta, beta2 = group["beta"], group["beta2"]
        decay = beta if beta2 is None else beta2
        evidence = beta * (1 - beta) if beta2 is None else 1 - beta2
        arguments = (
            gradient, state["momentum"], state["row_factor"], state["col_factor"],
            beta, decay, evidence, group["eps"], group["centered"],
        )
        if isinstance(self, MN):
            arguments += (group["shrinkage"], group["inverse_method"], group["ns_iters"])
        direction, momentum, row, col = self._kernel(*arguments)
        state["momentum"].copy_(momentum)
        state["row_factor"].copy_(row)
        state["col_factor"].copy_(col)
        return direction


class MND(_MeanOptimizer):
    """Algorithm 1: diagonal covariance factors with a column trace correction."""

    def __init__(self, params, lr=0.003, beta=0.925, eps=1e-9, weight_decay=0.0,
                 centered=True, beta2=None, compile=False):
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay,
                        centered=centered, beta2=beta2)
        super().__init__(params, defaults, utils.mnd_direction, compile)


class MN(_MeanOptimizer):
    """Algorithm 3: approximate dense MN, with trace-preserving shrinkage."""

    def __init__(self, params, lr=0.0006, beta=0.9, eps=1e-8, weight_decay=0.0,
                 centered=True, beta2=None, shrinkage=0.01, compile=False,
                 inverse_method="newton_schulz", ns_iters=6):
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay,
                        centered=centered, beta2=beta2, shrinkage=shrinkage,
                        inverse_method=inverse_method, ns_iters=ns_iters)
        super().__init__(params, defaults, utils.mn_direction, compile)

    def _validate(self, group):
        super()._validate(group)
        if not 0 <= group["shrinkage"] <= 1:
            raise ValueError("shrinkage must be in [0, 1]")
        if group["inverse_method"] not in ("newton_schulz", "eigh"):
            raise ValueError("inverse_method must be 'newton_schulz' or 'eigh'")
        if type(group["ns_iters"]) is not int or group["ns_iters"] < 1:
            raise ValueError("ns_iters must be a positive integer")


class MNDMuon(_MatrixOptimizer):
    """Algorithm 2: one factor sweep followed by whitened momentum and polar."""

    def __init__(self, params, lr=0.03, beta=0.9, eps=1e-9, weight_decay=0.0,
                 polar_method="polar_express", polar_steps=5, compile=False):
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay,
                        polar_method=polar_method, polar_steps=polar_steps)
        super().__init__(params, defaults, utils.mnd_muon_momentum, compile)

    def _validate(self, group):
        super()._validate(group)
        if group["polar_method"] not in ("polar_express", "svd"):
            raise ValueError("polar_method must be 'polar_express' or 'svd'")
        if type(group["polar_steps"]) is not int or group["polar_steps"] < 1:
            raise ValueError("polar_steps must be a positive integer")

    def _matrix_direction(self, gradient, state, group):
        momentum, row, col = self._kernel(
            gradient, state["momentum"], state["direction"],
            state["row_factor"], state["col_factor"],
            group["beta"], group["eps"],
        )
        # Keep BF16 rounding in the polar solver independent of compiler fusion.
        direction = utils.polar_express(momentum, group["polar_steps"]) if group["polar_method"] == "polar_express" else utils.polar_svd(momentum)
        state["momentum"].copy_(momentum)
        state["row_factor"].copy_(row)
        state["col_factor"].copy_(col)
        state["direction"].copy_(direction)
        return direction
