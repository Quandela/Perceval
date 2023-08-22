# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import random

import perceval as pcvl
import perceval.components.unitary_components as comp
import numpy as np
from math import comb
from scipy.optimize import minimize

# Differential equation parameters
lambd = 8
kappa = 0.1


def F(u_prime, u, x):  # DE: F(u_prime, u, x) = 0
    # Must work with numpy arrays
    return (u_prime + lambd * u * (kappa + np.tan(lambd * x)))


# Boundary condition (f(x_0)=f_0)
x_0 = 0
f_0 = 1

# Modeling parameters
n_grid = 50  # number of grid points of the discretized differential equation
range_min = 0  # minimum of the interval on which we wish to approximate our function
range_max = 1  # maximum of the interval on which we wish to approximate our function
X = np.linspace(range_min, range_max, n_grid)  # Optimisation grid

# Parameters of the quantum machine learning procedure
eta = 5  # weight granted to the initial condition
a = 200  # Approximate boundaries of the interval that the image of the trial function can cover

N = m = 6
N2 = N ** 2

input_state = pcvl.BasicState([1] * N + [0] * (m - N))
s1 = pcvl.SLOSBackend()
s1.set_circuit(pcvl.Unitary(pcvl.Matrix.random_unitary(m)))
s1.preprocess([input_state])

random.seed(0)
np.random.seed(0)
pcvl.random_seed(0)

fock_dim = comb(N + m - 1, N)
lambda_random = np.random.rand(fock_dim)
lambda_random = a * (lambda_random - np.mean(lambda_random)) / np.std(lambda_random)

dx = (range_max - range_min) / (n_grid - 1)
parameters = np.random.normal(size=4 * N2)


def calc(circuit, input_state, coefs):
    s1.set_circuit(circuit)
    probs = s1.all_prob(input_state)
    return np.sum(np.multiply(probs, coefs))


def computation(params):
    """compute the loss function of a given differential equation in order for it to be optimized"""
    coefs = lambda_random  # coefficients of the M observable
    # initial condition with the two universal interferometers and the phase shift in the middle

    U_1 = pcvl.Matrix.random_unitary(N, params[:2 * N2])
    U_2 = pcvl.Matrix.random_unitary(N, params[2 * N2:4 * N2])

    # Circuit creation
    px = pcvl.P("px")
    c = (comp.Unitary(U_2)
         // (0, comp.PS(px))
         // comp.Unitary(U_1))

    px.set_value(x_0)
    f_theta_0 = calc(c, input_state, coefs)

    # boundary condition given a weight eta
    loss = eta * (f_theta_0 - f_0) ** 2 * n_grid

    # Warning : Y[0] is before the domain we are interested in (used for differentiation), the domain begins at Y[1]
    Y = np.zeros(n_grid + 2)

    # Small optimisation working if x_0 == range_min
    if x_0 == range_min:
        Y[1] = f_theta_0
        assigned = 1

    else:
        assigned = 0

    px.set_value(range_min - dx)
    Y[0] = calc(c, input_state, coefs)

    for i in range(assigned, n_grid):
        x = X[i]

        px.set_value(x)
        Y[i + 1] = calc(c, input_state, coefs)

    # Y_prime[0] is the beginning of the domain /!\ not the same for Y
    px.set_value(range_max + dx)
    Y[n_grid + 1] = calc(c, input_state, coefs)

    Y_prime = (Y[2:] - Y[:-2]) / (2 * dx)

    # This method is apparently the fastest to calculate the L2 norm squared
    loss += np.sum((F(Y_prime, Y[1:-1], X)) ** 2)

    current_loss = loss / n_grid
    return current_loss


def test_QML_DE_solver(benchmark):
    benchmark(minimize, computation, parameters,
              method='BFGS', options={'gtol': 1E-2})
