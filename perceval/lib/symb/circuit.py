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

import sympy as sp
import numpy as np

from perceval.components import ACircuit
from perceval.utils import Matrix, format_parameters


class BS(ACircuit):
    _name = "BS"

    def __init__(self, R=None, theta=None, phi=0):
        super().__init__(2)
        assert R is None or theta is None, "cannot set both R and theta"
        self._phi = self._set_parameter("phi", phi, 0, 2*sp.pi)
        if R is not None:
            self._R = self._set_parameter("R", R, 0, 1, periodic=False)
        else:
            if theta is None:
                theta = sp.pi/4
            self._theta = self._set_parameter("theta", theta, 0, 2*sp.pi)

    def _compute_unitary(self, assign=None, use_symbolic=False):
        self.assign(assign)
        if use_symbolic:
            if "R" in self.params:
                cos_theta = sp.sqrt(self._R.spv)
                sin_theta = sp.sqrt(1-self._R.spv)
            else:
                cos_theta = sp.cos(self._theta.spv)
                sin_theta = sp.sin(self._theta.spv)
            return Matrix([[cos_theta, sin_theta*sp.I*sp.exp(-self._phi.spv*sp.I)],
                           [sin_theta*sp.exp(self._phi.spv*sp.I)*sp.I, cos_theta]], True)
        else:
            if "R" in self.params:
                cos_theta = np.sqrt(float(self._R))
                sin_theta = np.sqrt(1-float(self._R))
            else:
                cos_theta = np.cos(float(self._theta))
                sin_theta = np.sin(float(self._theta))
            return Matrix([[cos_theta, sin_theta*(1j*np.cos(-float(self._phi)) - np.sin(-float(self._phi)))],
                           [sin_theta*(1j*np.cos(float(self._phi)) - np.sin(float(self._phi))), cos_theta]], False)

    def get_variables(self, map_param_kid=None):
        parameters = {}
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        if "theta" in self._params:
            self.variable_def(parameters, "theta", "theta", sp.pi/4, map_param_kid)
        else:
            self.variable_def(parameters, "R", "R", 0.5, map_param_kid)
        self.variable_def(parameters, "phi", "phi", 0, map_param_kid)
        return parameters

    def describe(self, map_param_kid=None):
        parameters = self.get_variables(map_param_kid)
        params_str = format_parameters(parameters, separator=', ')
        return "symb.BS(%s)" % params_str

    def inverse(self, v=False, h=False):
        if self._phi.is_symbolic():
            if v:
                self._phi = -self._phi.spv
            if h:
                self._phi = self._phi.spv+sp.pi
        else:
            if v:
                self._phi = -float(self._phi)
            if h:
                self._phi = float(self._phi)+np.pi


class PBS(ACircuit):
    _name = "PBS"
    _supports_polarization = True

    def __init__(self):
        super().__init__(2)

    def _compute_unitary(self, assign=None, use_symbolic=False):
        self.assign(assign)
        return Matrix([[0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]], use_symbolic)

    def get_variables(self, map_param_kid=None):
        return {}

    # TODO: make method static
    def describe(self, _=None):
        return "phys.PBS()"

    def inverse(self, v=False, h=False):
        if self._phi.is_symbolic():
            if v:
                self._phi = -self._phi.spv
            if h:
                self._phi = self._phi.spv+sp.pi
        else:
            if v:
                self._phi = -float(self._phi)
            if h:
                self._phi = float(self._phi)+np.pi


class TD(ACircuit):
    """Time delay"""
    _name = "TD"
    delay_circuit = True
    stroke_style = {"stroke": "black", "stroke_width": 2}

    def __init__(self, t):
        super().__init__(1)
        self._dt = self._set_parameter("t", t, 0, sp.oo, False)

    def _compute_unitary(self, assign=None, use_symbolic=False):
        raise RuntimeError("DT circuit cannot be simulated with unitary matrix")

    def get_variables(self, map_param_kid=None):
        parameters = {}
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "t", "t", None, map_param_kid)
        return parameters

    def describe(self, map_param_kid=None):
        parameters = self.get_variables(map_param_kid)
        params_str = format_parameters(parameters, separator=', ')
        return "phys.TD(%s)" % params_str


class PS(ACircuit):
    _name = "PS"

    def __init__(self, phi):
        super().__init__(1)
        self._phi = self._set_parameter("phi", phi, 0, 2*sp.pi)

    def _compute_unitary(self, assign=None, use_symbolic=False):
        self.assign(assign)
        if use_symbolic:
            return Matrix([[sp.exp(self._phi.spv*sp.I)]], True)
        else:
            return Matrix([[np.cos(float(self._phi)) + 1j * np.sin(float(self._phi))]], False)

    def get_variables(self, map_param_kid=None):
        parameters = {}
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "phi", "phi", None, map_param_kid)
        return parameters

    def describe(self, map_param_kid=None):
        parameters = self.get_variables(map_param_kid)
        params_str = format_parameters(parameters, separator=', ')
        return "symb.PS(%s)" % params_str

    def inverse(self, v=False, h=False):
        if h:
            if self._phi.is_symbolic():
                self._phi = -self._phi.spv
            else:
                self._phi = -float(self._phi)


class Unitary(ACircuit):
    _name = "Unitary"

    def __init__(self, U: Matrix, name: str = None, use_polarization: bool = False):
        assert U is not None, "A unitary matrix is required"
        assert U.is_unitary(), "U parameter must be a unitary matrix"
        # Even for a symb.Unitary component, a symbolic matrix is not a use case. On top of that, it slows down
        # computations quite a bit!
        assert not U.is_symbolic(), "U parameter must not be symbolic"
        self._u = U
        if name is not None:
            self._name = name
        m = U.shape[0]
        self._supports_polarization = use_polarization
        if use_polarization:
            assert m % 2 == 0, "Polarization matrix should have an even number of rows/col"
            m //= 2
        super().__init__(m)

    def _compute_unitary(self, assign: dict = None, use_symbolic: bool = False) -> Matrix:
        # Ignore assign and use_symbolic parameters as __init__ checked the unitary matrix is numeric
        return self._u

    def inverse(self, v=False, h=False):
        if v:
            self._u = np.flip(self._u)
        if h:
            self._u = self._u.inv()

    def describe(self, _=None):
        params = [f"Matrix('''{self._u}''')"]
        if self._name != Unitary._name:
            params.append(f"name='{self._name}'")
        if self._supports_polarization:
            params.append("use_polarization=True")
        return f"symb.Unitary({', '.join(params)})"


class PERM(Unitary):
    _name = "PERM"

    def __init__(self, perm):
        assert isinstance(perm, list), "permutation Operator needs list parameter"
        assert (min(perm) == 0 and
                max(perm)+1 == len(perm) == len(set(perm)) == len([n for n in perm if isinstance(n, int)])),\
            "%s is not a permutation" % perm
        n = len(perm)
        u = Matrix.zeros((n, n), use_symbolic=False)
        for i, v in enumerate(perm):
            u[v, i] = 1
        super().__init__(U=u)

    def get_variables(self, _=None):
        return {'PERM': ''}

    def describe(self, _=None):
        return "symb.PERM(%s)" % str(self.perm_vector)

    def definition(self):
        return self.U

    @property
    def perm_vector(self):
        nz = np.nonzero(self._u)
        m_list = nz[1].tolist()
        return [m_list.index(i) for i in nz[0]]


class WP(ACircuit):
    _name = "WP"
    _supports_polarization = True

    def __init__(self, delta, xsi):
        super().__init__(1)
        self._delta = self._set_parameter("delta", delta, -sp.pi, sp.pi)
        self._xsi = self._set_parameter("xsi", xsi, -sp.pi, sp.pi)

    def _compute_unitary(self, assign=None, use_symbolic=False):
        self.assign(assign)
        if use_symbolic:
            delta = self._delta.spv
            xsi = self._xsi.spv
            return Matrix([[
                            sp.cos(delta)+sp.I*sp.sin(delta)*sp.cos(2*xsi),
                            sp.I*sp.sin(delta)*sp.sin(2*xsi)
                           ], [
                            sp.I * sp.sin(delta) * sp.sin(2 * xsi),
                            sp.cos(delta) - sp.I * sp.sin(delta) * sp.cos(2 * xsi)
                           ]], True)
        else:
            delta = float(self._delta)
            xsi = float(self._xsi)
            return Matrix([[
                            np.cos(delta)+1j*np.sin(delta)*np.cos(2*xsi),
                            1j*np.sin(delta)*np.sin(2*xsi)
                           ], [
                            1j * np.sin(delta) * np.sin(2 * xsi),
                            np.cos(delta) - 1j * np.sin(delta) * np.cos(2 * xsi)
                           ]], False)


    def get_variables(self, map_param_kid=None):
        parameters = {}
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "xsi", "xsi", None, map_param_kid)
        self.variable_def(parameters, "delta", "delta", None, map_param_kid)
        return parameters

    def describe(self, map_param_kid=None):
        parameters = self.get_variables(map_param_kid)
        params_str = format_parameters(parameters, separator=', ')
        return "phys.WP(%s)" % params_str


class HWP(WP):
    _name = "HWP"

    def __init__(self, xsi):
        super().__init__(sp.pi/2, xsi)


class QWP(WP):
    _name = "QWP"

    def __init__(self, xsi):
        super().__init__(sp.pi/4, xsi)


class PR(ACircuit):
    """Polarization rotator"""
    _name = "PR"
    _supports_polarization = True

    def __init__(self, delta):
        super().__init__(1)
        self._delta = self._set_parameter("delta", delta, -sp.pi, sp.pi)

    def _compute_unitary(self, assign=None, use_symbolic=False):
        self.assign(assign)
        if use_symbolic:
            delta = self._delta.spv
            return Matrix([[sp.cos(delta), sp.sin(delta)],
                           [-sp.sin(delta), sp.cos(delta)]], True)
        else:
            delta = float(self._delta)
            return Matrix([[np.cos(delta), np.sin(delta)],
                           [-np.sin(delta), np.cos(delta)]], False)

    def get_variables(self, map_param_kid=None):
        parameters = {}
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "delta", "delta", None, map_param_kid)
        return parameters

    def describe(self, map_param_kid=None):
        parameters = self.get_variables(map_param_kid)
        params_str = format_parameters(parameters, separator=', ')
        return "phys.PR(%s)" % params_str
