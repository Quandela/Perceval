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

from copy import copy
import numpy as np
import sympy as sp

from perceval.components import ACircuit
from perceval.utils import Matrix, format_parameters, BasicState, StateVector, Parameter


class GenericBS(ACircuit):
    """Universal beam splitter"""
    _name = "BS"

    def __init__(self, R=None, theta=None, phi_a=0, phi_b=3*sp.pi/2, phi_d=sp.pi):
        super().__init__(2)
        assert R is None or theta is None, "cannot set both R and theta"
        self._phi_a = self._set_parameter("phi_a", phi_a, 0, 2*sp.pi)
        self._phi_b = self._set_parameter("phi_b", phi_b, 0, 2*sp.pi)
        self._phi_d = self._set_parameter("phi_d", phi_d, 0, 2*sp.pi)
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
            phi_c = - self._phi_b.spv + self._phi_d.spv + self._phi_a.spv
            return Matrix([[cos_theta*sp.exp(self._phi_a.spv*sp.I), sin_theta*sp.exp(self._phi_b.spv*sp.I)*sp.I],
                           [sin_theta*sp.exp(phi_c*sp.I)*sp.I, cos_theta*sp.exp(self._phi_d.spv*sp.I)]], True)
        else:
            if "R" in self.params:
                cos_theta = np.sqrt(float(self._R))
                sin_theta = np.sqrt(1-float(self._R))
            else:
                cos_theta = np.cos(float(self._theta))
                sin_theta = np.sin(float(self._theta))
            phi_c = - float(self._phi_b) + float(self._phi_d) + float(self._phi_a)
            return Matrix([[cos_theta*(np.cos(float(self._phi_a)) + 1j * np.sin(float(self._phi_a))),
                            sin_theta*(1j * np.cos(float(self._phi_b)) - np.sin(float(self._phi_b)))],
                           [sin_theta*(1j * np.cos(float(phi_c)) - np.sin(float(phi_c))),
                            cos_theta*(np.cos(float(self._phi_d)) + 1j * np.sin(float(self._phi_d)))]], False)

    def get_variables(self, map_param_kid=None):
        parameters = {}
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        if "theta" in self._params:
            self.variable_def(parameters, "theta", "theta", sp.pi/4, map_param_kid)
        else:
            self.variable_def(parameters, "R", "R", 0.5, map_param_kid)
        self.variable_def(parameters, "phi_a", "phi_a", 0, map_param_kid)
        self.variable_def(parameters, "phi_b", "phi_b", 3*sp.pi/2, map_param_kid)
        self.variable_def(parameters, "phi_d", "phi_d", sp.pi, map_param_kid)
        return parameters

    def describe(self, map_param_kid=None):
        parameters = self.get_variables(map_param_kid)
        params_str = format_parameters(parameters, separator=', ')
        return "GenericBS(%s)" % params_str

    def inverse(self, v=False, h=False):
        if v:
            phi_a = self._phi_a
            self._phi_a = self._phi_d
            self._phi_d = phi_a
            self._phi_b._value = self._phi_a.spv+self._phi_d.spv-self._phi_b.spv
        if h:
            self._phi_a._value = -self._phi_a.spv
            self._phi_d._value = -self._phi_d.spv
            self._phi_b._value = sp.pi-(-self._phi_a.spv-self._phi_d.spv-self._phi_b.spv)


class SimpleBS(ACircuit):
    """Beam splitter with a single phase"""
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
        return "SimpleBS(%s)" % params_str

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


class PS(ACircuit):
    """Phase shifter"""
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
        return "PS(%s)" % params_str

    def inverse(self, v=False, h=False):
        if h:
            if self._phi.is_symbolic():
                self._phi = -self._phi.spv
            else:
                self._phi = -float(self._phi)


class WP(ACircuit):
    """Wave plate"""
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
        return "WP(%s)" % params_str

    def inverse(self, v=False, h=False):
        raise NotImplementedError("inverse not yet implemented")


class HWP(WP):
    """Half wave plate"""
    _name = "HWP"

    def __init__(self, xsi):
        super().__init__(sp.pi/2, xsi)

    def definition(self):
        return HWP(xsi=Parameter('xsi')).U


class QWP(WP):
    """Quarter wave plate"""
    _name = "QWP"

    def __init__(self, xsi):
        super().__init__(sp.pi/4, xsi)

    def definition(self):
        return QWP(xsi=Parameter('xsi')).U


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
        return "PR(%s)" % params_str

    def inverse(self, v=False, h=False):
        raise NotImplementedError("inverse not yet implemented")


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
        return "TD(%s)" % params_str

    def inverse(self, v=False, h=False):
        if h:
            raise NotImplementedError("Cannot inverse a time delay")

    def definition(self):
        raise RuntimeError("DT circuit has no unitary matrix definition")


class Unitary(ACircuit):
    """Generic component defined by a unitary matrix"""
    _name = "Unitary"

    def __init__(self, U: Matrix, name: str = None, use_polarization: bool = False):
        assert U is not None, "A unitary matrix is required"
        assert U.is_unitary(), "U parameter must be a unitary matrix"
        # A symbolic matrix is not a use case for this component
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
    """Permutation"""
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
        return "PERM(%s)" % str(self.perm_vector)

    def definition(self):
        return self.U

    @property
    def perm_vector(self):
        nz = np.nonzero(self._u)
        m_list = nz[1].tolist()
        return [m_list.index(i) for i in nz[0]]

    def apply(self, r, sv):
        if isinstance(sv, BasicState):
            sv = StateVector(sv)

        min_r = r[0]
        max_r = r[-1] + 1

        permutation = self.perm_vector
        inv = np.empty_like(permutation)
        inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
        inv = [inv[i].item() for i in range(len(inv))]

        nsv = copy(sv)
        nsv.clear()
        nsv.update({BasicState(state.set_slice(slice(min_r, max_r), BasicState([state[i + min_r]
                                                                                for i in inv]))):
                        prob_ampli for state, prob_ampli in sv.items()})

        return nsv


class PBS(Unitary):
    """Polarized beam spliter"""
    _name = "PBS"

    def __init__(self):
        u = Matrix([[0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])
        super().__init__(U=u, use_polarization=True)

    def get_variables(self, map_param_kid=None):
        return {}

    # noinspection PyMethodMayBeStatic
    def describe(self, _=None):
        return "PBS()"
