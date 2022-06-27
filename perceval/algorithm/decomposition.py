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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy

import numpy as np
import sympy as sp
import scipy as scp

from perceval.utils import Matrix, global_params

from .solve import solve


def add_phases(phase_shifter_fn, D):
    phases = []
    for idx in range(len(D)):
        iD = D[idx]
        a = iD.real
        b = iD.imag
        if b != 0 or a < 0:
            if b == 0:
                phi = np.pi
            elif a == 0:
                if b > 0:
                    phi = np.pi / 2
                else:
                    phi = 3 * np.pi / 2
            else:
                phi = np.arctan(b / a)
                if a < 0:
                    phi = phi + np.pi
            phases = [(idx, phase_shifter_fn(phi))] + phases
    return phases


def decompose_triangle(u,
                       component,
                       phase_shifter_fn,
                       permutation,
                       precision,
                       constraints,
                       allow_error):
    m = u.shape[0]
    params = component.get_parameters()
    params_symbols = [x.spv for x in params]
    bounds = [not x.is_periodic and x.bounds or None for x in params]

    if constraints is None:
        constraints = [[None] * len(params)]

    if precision is None:
        precision = global_params["min_complex_component"]

    cU = component.U
    cU_inv = cU.inv()
    cU_inv.simplify()

    list_components = []
    for j in range(m - 1, 0, -1):
        for n in range(j):
            # goal is to null M[n,m]
            solve_cell = False
            if abs(u[n, j]) <= precision:
                solve_cell = True
            else:
                if permutation is not None:
                    p = [0]
                    for k in range(n + 1, j + 1):
                        p.append(k - n)
                        if abs(u[k, j]) <= precision:
                            p[0] = k - n
                            p[-1] = 0
                            list_components = [(list(range(n, k + 1)), permutation(p))] + list_components
                            RI = Matrix.eye(m, use_symbolic=False)
                            RI[n, n] = RI[k, k] = 0
                            RI[k, n] = RI[n, k] = 1
                            u = RI @ u
                            solve_cell = True
                            break
            if not solve_cell:
                equation = cU_inv[0, 0] * u[n, j] + cU_inv[0, 1] * u[n + 1, j]
                f = sp.lambdify([params_symbols], [equation], modules=[np, scp])
                g = lambda *p: np.real(np.abs(f(*p)))
                x0 = [p.random() for p in params]
                # look for a constraint solution first
                res = None
                for c in constraints:
                    res = solve(g, x0, list(c), bounds, precision, allow_error)
                    if res is not None:
                        break
                if res is None:
                    return None

                RI = Matrix.eye(m, use_symbolic=False)
                instantiated_component = copy.deepcopy(component)
                substitution = {}
                for i, r in enumerate(res):
                    substitution[params_symbols[i]] = r
                    instantiated_component.get_parameters()[0].fix_value(res[i])

                RI[n, n] = complex(cU_inv[0, 0].subs(substitution))
                RI[n, n + 1] = complex(cU_inv[0, 1].subs(substitution))
                RI[n + 1, n] = complex(cU_inv[1, 0].subs(substitution))
                RI[n + 1, n + 1] = complex(cU_inv[1, 1].subs(substitution))

                u = RI @ u
                list_components = [((n, n + 1), instantiated_component)] + list_components
            u[n, j] = 0

    if phase_shifter_fn:
        D = np.diag(u)
        list_components = add_phases(phase_shifter_fn, D) + list_components

    return list_components


def decompose_rectangle(u,
                        component,
                        phase_shifter_fn,
                        permutation,
                        precision,
                        constraints,
                        allow_error):
    raise NotImplementedError("rectangular decomposition not implemented yet")
