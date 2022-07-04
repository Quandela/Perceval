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

import scipy.optimize as so


def solve(f, x0, constraint, bounds, precision, allow_error=False):
    r"""Solve f starting with x0 and compliant with constraints.

    :param allow_error:
    :param bounds:
    :param f:
    :param x0:
    :param constraint:
    :param precision:
    :return:
    """
    if len(x0) == 0:
        if abs(f([])) < precision:
            return []
    for i, c in enumerate(constraint):
        if c is not None:
            c = float(c)
            res = solve(lambda x: f([*x[:i], c, *x[i:]]),
                        x0[:i]+x0[i+1:],
                        constraint[:i]+constraint[i+1:],
                        bounds[:i]+bounds[i+1:],
                        precision, allow_error)
            if res is None:
                return None
            return [*res[:i], c, *res[i:]]

    if x0:
        res = so.minimize(f, x0, method="L-BFGS-B", bounds=[b is not None and (float(b[0]), float(b[1])) or (None, None)
                                                            for b in bounds])
        f_x = res.fun
        x = res.x
    else:
        f_x = f([])
        x = []

    if f_x > precision and not allow_error:
        return None

    return x
