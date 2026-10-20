import math

import torch


def inverse(diagonal, eps):
    return diagonal.clamp_min(eps).reciprocal()


def inverse_sqrt(diagonal, eps):
    return diagonal.clamp_min(eps).rsqrt()


def balance(row, col, eps):
    row_trace = row.sum() if row.ndim == 1 else row.trace()
    col_trace = col.sum() if col.ndim == 1 else col.trace()
    scale = (col_trace.clamp_min(eps) / row_trace.clamp_min(eps)).sqrt()
    return scale * row, col / scale


def shrink(matrix, amount):
    matrix = (matrix + matrix.T) * 0.5
    result = (1 - amount) * matrix
    result.diagonal().add_(amount * matrix.trace() / matrix.shape[0])
    return result


def matrix_power(matrix, exponent, eps):
    values, vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    return (vectors * values.clamp_min(eps).pow(exponent)) @ vectors.T


# Newton–Schulz solvers adapted from gowerrobert/GPT-opt,
# gptopt/optim/matrix_functions.py: https://github.com/gowerrobert/GPT-opt
def matrix_power_newton_schulz(matrix, exponent, eps, ns_iters=6):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if exponent not in (-1, -0.5):
        raise ValueError("Newton–Schulz supports exponents -1 and -0.5")
    if type(ns_iters) is not int or ns_iters < 1:
        raise ValueError("ns_iters must be a positive integer")

    dtype = matrix.dtype
    matrix = matrix.to(torch.float64 if dtype == torch.float64 else torch.float32)
    eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    matrix = (matrix + matrix.T) * 0.5 + eps * eye
    if exponent == -1:
        scale = matrix.norm(p="fro").square().clamp_min(eps)
        result = matrix / scale
        for _ in range(ns_iters):
            result = result @ (2 * eye - matrix @ result)
        return ((result + result.T) * 0.5).to(dtype)

    scale = matrix.norm(p="fro").clamp_min(eps)
    y, z = matrix / scale, eye
    for _ in range(ns_iters):
        update = (3 * eye - z @ y) * 0.5
        y, z = y @ update, update @ z
    return (z / scale.sqrt()).to(dtype)


def variance_root(u, v, dimension, eps):
    # Rationalize the positive-v branch to avoid subtracting close numbers.
    u = u.clamp_min(0)
    discriminant = torch.hypot(v, 2 * math.sqrt(dimension) * u.sqrt())
    half_sum = 0.5 * discriminant + 0.5 * v.abs()
    root = torch.where(
        v >= 0,
        u / half_sum.clamp_min(torch.finfo(u.dtype).tiny),
        half_sum / dimension,
    )
    return root.square().clamp_min(eps)


# Polar Express: https://github.com/NoahAmsel/PolarExpress
# Fixed coefficient schedule and iteration from the paper, via
# gowerrobert/GPT-opt, gptopt/optim/polar.py.
# MIT License
_POLAR_COEFFICIENTS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
)


def polar_express(matrix, steps):
    work = matrix.to(torch.bfloat16)
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.T
    norm = work.norm()
    work = work / (torch.where(norm > 0, norm, torch.ones_like(norm)) * 1.01)
    for index in range(steps):
        position = min(index, len(_POLAR_COEFFICIENTS) - 1)
        a, b, c = _POLAR_COEFFICIENTS[position]
        if position < len(_POLAR_COEFFICIENTS) - 1:
            a, b, c = a / 1.01, b / 1.01**3, c / 1.01**5
        a, b, c = a / 1.01, b / 1.01, c / 1.01
        gram = work @ work.T
        polynomial = b * gram + (c * gram) @ gram
        work = a * work + polynomial @ work
    if transposed:
        work = work.T
    return work.to(matrix.dtype)


def polar_svd(matrix):
    left, values, right = torch.linalg.svd(matrix, full_matrices=False)
    cutoff = torch.finfo(matrix.dtype).eps * max(matrix.shape) * values.max()
    return (left * (values > cutoff).to(matrix.dtype)) @ right


def adam_direction(gradient, mean, moment, beta1, beta2, eps, step, centered):
    mean = beta1 * mean + (1 - beta1) * gradient
    moment = beta2 * moment + (1 - beta2) * gradient.square()
    corrected_mean = mean / (1 - beta1**step)
    variance = moment / (1 - beta2**step)
    if centered:
        variance = variance - corrected_mean.square()
    direction = corrected_mean / (variance.clamp_min(0).sqrt() + eps)
    return direction, mean, moment


def mnd_direction(gradient, mean, row, col, beta, decay, evidence, eps, centered):
    m, n = gradient.shape
    residual = gradient - mean if centered else gradient
    mean = beta * mean + (1 - beta) * gradient
    squared = residual.square()
    new_row = decay * row + evidence / n * (squared @ inverse(col, eps))
    inv_row = inverse(new_row, eps)
    new_col = decay / m * torch.dot(inv_row, row) * col
    new_col = new_col + evidence / m * (squared.T @ inv_row)
    row, col = balance(new_row, new_col, eps)
    direction = mean * inverse_sqrt(row, eps)[:, None] * inverse_sqrt(col, eps)[None, :]
    return direction, mean, row, col


def mn_direction(gradient, mean, row, col, beta, decay, evidence, eps, centered,
                 shrinkage, inverse_method="newton_schulz", ns_iters=6):
    def power(factor, exponent):
        matrix = shrink(factor, shrinkage)
        if inverse_method == "eigh":
            return matrix_power(matrix, exponent, eps)
        return matrix_power_newton_schulz(matrix, exponent, eps, ns_iters)

    m, n = gradient.shape
    residual = gradient - mean if centered else gradient
    mean = beta * mean + (1 - beta) * gradient
    new_row = decay * row + evidence / n * (
        residual @ power(col, -1) @ residual.T
    )
    new_col = decay * col + evidence / m * (
        residual.T @ power(new_row, -1) @ residual
    )
    row, col = balance(new_row, new_col, eps)
    direction = (
        power(row, -0.5)
        @ mean
        @ power(col, -0.5)
    )
    return direction, mean, row, col


def mnd_muon_momentum(gradient, momentum, previous_direction, row, col, beta, eps):
    m, n = gradient.shape
    q = beta * torch.outer(row, col) + (1 - beta) * gradient.square()
    linear = (1 - beta) * gradient * previous_direction
    new_row = variance_root(q @ inverse(col, eps), linear @ inverse_sqrt(col, eps), n, eps)
    new_col = variance_root(
        q.T @ inverse(new_row, eps), linear.T @ inverse_sqrt(new_row, eps), m, eps
    )
    row, col = balance(new_row, new_col, eps)
    whitened = gradient * inverse_sqrt(row, eps)[:, None] * inverse_sqrt(col, eps)[None, :]
    momentum = beta * momentum + (1 - beta) * whitened
    return momentum, row, col
