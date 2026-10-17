import numpy as np


def power(matrix, exponent, eps):
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.maximum(values, eps)**exponent) @ vectors.T


def newton_schulz_power(matrix, exponent, eps, ns_iters):
    identity = np.eye(len(matrix))
    matrix = (matrix + matrix.T) / 2 + eps * identity
    if exponent == -1:
        estimate = matrix / max(np.linalg.norm(matrix, "fro")**2, eps)
        for _ in range(ns_iters):
            estimate = 2 * estimate - estimate @ (matrix @ estimate)
        return (estimate + estimate.T) / 2
    scale = max(np.linalg.norm(matrix, "fro"), eps)
    y, z = matrix / scale, identity
    for _ in range(ns_iters):
        correction = (3 * identity - z @ y) / 2
        y, z = y @ correction, correction @ z
    return z / np.sqrt(scale)


def shrink(matrix, amount):
    return (1 - amount) * matrix + amount * np.trace(matrix) / len(matrix) * np.eye(len(matrix))


def mean_step(gradient, mean, row, col, *, beta, eps, centered, beta2, dense=False,
              shrinkage=0, inverse_method="eigh", ns_iters=6):
    def factor_power(factor, exponent):
        matrix = shrink(factor, shrinkage)
        if inverse_method == "newton_schulz":
            return newton_schulz_power(matrix, exponent, eps, ns_iters)
        return power(matrix, exponent, eps)

    m, n = gradient.shape
    residual = gradient - mean if centered else gradient
    next_mean = beta * mean + (1 - beta) * gradient
    decay = beta if beta2 is None else beta2
    gain = beta * (1 - beta) if beta2 is None else 1 - beta2
    if dense:
        next_row = decay * row + gain / n * residual @ factor_power(col, -1) @ residual.T
        next_col = decay * col + gain / m * residual.T @ factor_power(next_row, -1) @ residual
        alpha = np.sqrt(max(np.trace(next_col), eps) / max(np.trace(next_row), eps))
    else:
        next_row = np.array([
            decay * row[i] + gain / n * sum(residual[i, j]**2 / max(col[j], eps) for j in range(n))
            for i in range(m)
        ])
        next_col = np.array([
            decay / m * col[j] * sum(row[i] / max(next_row[i], eps) for i in range(m))
            + gain / m * sum(residual[i, j]**2 / max(next_row[i], eps) for i in range(m))
            for j in range(n)
        ])
        alpha = np.sqrt(max(sum(next_col), eps) / max(sum(next_row), eps))
    next_row, next_col = alpha * next_row, next_col / alpha
    if dense:
        direction = factor_power(next_row, -.5) @ next_mean @ factor_power(next_col, -.5)
    else:
        direction = np.array([
            [next_mean[i, j] / np.sqrt(max(next_row[i], eps) * max(next_col[j], eps)) for j in range(n)]
            for i in range(m)
        ])
    return direction, next_mean, next_row, next_col


def muon_factors(gradient, momentum, previous_direction, row, col, beta, eps):
    m, n = gradient.shape
    q = beta * np.outer(row, col) + (1 - beta) * gradient**2
    linear = (1 - beta) * gradient * previous_direction

    def root(u, v, dimension):
        return max(((np.sqrt(v*v + 4*dimension*u) - v) / (2*dimension))**2, eps)

    next_row = np.array([
        root(sum(q[i, j] / max(col[j], eps) for j in range(n)),
             sum(linear[i, j] / np.sqrt(max(col[j], eps)) for j in range(n)), n)
        for i in range(m)
    ])
    next_col = np.array([
        root(sum(q[i, j] / max(next_row[i], eps) for i in range(m)),
             sum(linear[i, j] / np.sqrt(max(next_row[i], eps)) for i in range(m)), m)
        for j in range(n)
    ])
    alpha = np.sqrt(max(sum(next_col), eps) / max(sum(next_row), eps))
    next_row, next_col = alpha * next_row, next_col / alpha
    next_momentum = np.array([
        [beta * momentum[i, j] + (1 - beta) * gradient[i, j]
         / np.sqrt(max(next_row[i], eps) * max(next_col[j], eps)) for j in range(n)]
        for i in range(m)
    ])
    return next_momentum, next_row, next_col
