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

from typing import Optional, Callable, List
from perceval.components.circuit import ACircuit
from perceval.utils import Matrix, P, global_params

from scipy import optimize as scpy_optimize


def _min_fnc(c: ACircuit, params: List[P], x: List[int], v: Optional[Matrix],
             f: Callable[[Matrix, Matrix], float], sign: float):
    for idx, p in enumerate(x):
        params[idx].set_value(p)
    value = f(c.compute_unitary(use_symbolic=False), v)
    return -sign * value


def _stop_criterion(f, f0, precision, accept):
    if accept:
        if abs(f-f0) < precision:
            return True
    return False


def optimize(c: ACircuit,
             v: Optional[Matrix],
             f: Callable[[Matrix, Matrix], float],
             niter: int = 20,
             target_opt: float = 0,
             precision: float = None,
             n_try: int = 10,
             sign=1) -> scpy_optimize.OptimizeResult:
    r"""Optimize parameters of a circuit according to Callable function

    :param c: circuit with parameters to optimize
    :param v: the reference unitary, can be None, in such case the fidelity function should ignore the second parameter
    :param f: fidelity function - first parameter is the unitary of the circuit, second parameter is the reference one
    :param niter: from `scipy.optimize.basinhopping`
    :param sign: -1 to find maximal values
    :param target_opt: optimal value for the function - used with `precision` for early stopping
    :param precision: used with `target_opt` for early stopping
    :return: OptimizeResult from scipy library
    """
    if precision is None:
        precision = global_params["min_complex_component"]
    params = c.get_parameters()
    best = None
    best_x = None
    while n_try > 0:
        init_params = [p.random() for p in params]
        temperature = 1.0
        res = scpy_optimize.basinhopping(lambda x: _min_fnc(c, params, x, v, f, sign), init_params,
                                         niter=niter,
                                         T= temperature,
                                         callback=lambda _, f, accept: _stop_criterion(f, target_opt, precision, accept))
        if best is None or res.fun < best:
            best = res.fun
            best_x = res.x
        if _stop_criterion(res.fun, target_opt, precision, True):
            break
        n_try -= 1
        temperature *= 1.1
    res.fun = best * -sign
    res.x = best_x
    return res
