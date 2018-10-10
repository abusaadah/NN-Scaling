import numpy as np


def uniform(n):
    return [np.float32(1.) for i in range(n)]

 
def harmonic(n):
    return [np.float32(1. / i) for i in range(1, n + 1)]


def linear_func(n):
    return[np.float32(1.0 - 1.0 * i/n) for i in range(1, n + 1)]

def exp(n):
    return [np.float32(np.exp(-1 * i)) for i in range(n)]


def uniform_exp(n, k=1):
    n_half = n // 2
    n_rest = n - n // 2
    return uniform(n_half) + np.exp(n_rest, k)

def step(n, steps=[1]):
    num = len(steps)
    coeffs = []
    for step in steps:
        coeffs.extend([step for j in range(int(np.ceil(1. * n / num)))])
    return coeffs[:n]


def three_steps(n):
    return step(n, steps=[1, 0.5, 0.25])


def four_steps(n):
    return step(n, steps=[1, 0.75, 0.5, 0.25])


def mag_steps(n):
    return step(n, steps=[1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
