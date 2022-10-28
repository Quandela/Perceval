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
from enum import Enum

import numpy as np
import sympy as sp

from .linear_circuit import ACircuit
from perceval.utils import Matrix, format_parameters, BasicState, StateVector, Parameter


class BSConvention(Enum):
    Rx = 0
    Ry = 1
    H = 2


class BS(ACircuit):
    """Beam splitter"""
    DEFAULT_NAME = "BS"

    def __init__(self, theta=sp.pi/2, phi_tl=0, phi_bl=0, phi_tr=0, phi_br=0,
                 convention: BSConvention = BSConvention.Rx):
        super().__init__(2)
        self._convention = convention
        self._theta = self._set_parameter("theta", theta, 0, 4*sp.pi)
        self._phi_tl = self._set_parameter("phi_tl", phi_tl, 0, 2*sp.pi)
        self._phi_bl = self._set_parameter("phi_bl", phi_bl, 0, 2*sp.pi)
        self._phi_tr = self._set_parameter("phi_tr", phi_tr, 0, 2*sp.pi)
        self._phi_br = self._set_parameter("phi_br", phi_br, 0, 2*sp.pi)

    @property
    def name(self):
        return f'{self.DEFAULT_NAME}({self._convention.name})'

    @property
    def convention(self):
        return self._convention

    @staticmethod
    def H(theta=sp.pi/2, phi_tl=0, phi_bl=0, phi_tr=0, phi_br=0):
        return BS(theta, phi_tl, phi_bl, phi_tr, phi_br, convention=BSConvention.H)

    @staticmethod
    def Rx(theta=sp.pi / 2, phi_tl=0, phi_bl=0, phi_tr=0, phi_br=0):
        return BS(theta, phi_tl, phi_bl, phi_tr, phi_br, convention=BSConvention.Rx)

    @staticmethod
    def Ry(theta=sp.pi / 2, phi_tl=0, phi_bl=0, phi_tr=0, phi_br=0):
        return BS(theta, phi_tl, phi_bl, phi_tr, phi_br, convention=BSConvention.Ry)

    @staticmethod
    def r_to_theta(r):
        """Compute theta given a reflectivity value
        """
        return 2*np.arccos(np.sqrt(float(r)))

    @staticmethod
    def theta_to_r(theta):
        if isinstance(theta, Parameter) and not theta.defined:
            return sp.cos(theta.spv/2)**2
        else:
            return np.cos(float(theta)/2)**2

    @property
    def reflectivity(self):
        return self.theta_to_r(self._theta)

    def _compute_unitary(self, assign=None, use_symbolic=False):
        if use_symbolic:
            theta = self._theta.spv
            cos_theta = sp.cos(theta/2)
            sin_theta = sp.sin(theta/2)
            phi_tl = self._phi_tl.spv
            phi_tr = self._phi_tr.spv
            phi_bl = self._phi_bl.spv
            phi_br = self._phi_br.spv
            u00_mul = sp.exp((phi_tl + phi_tr)*sp.I)
            u01_mul = sp.exp((phi_tr + phi_bl)*sp.I)
            u10_mul = sp.exp((phi_tl + phi_br)*sp.I)
            u11_mul = sp.exp((phi_br + phi_bl)*sp.I)
        else:
            cos_theta = np.cos(float(self._theta)/2)
            sin_theta = np.sin(float(self._theta)/2)
            phi_tl_tr = float(self._phi_tl) + float(self._phi_tr)
            u00_mul = np.cos(phi_tl_tr) + 1j*np.sin(phi_tl_tr)
            phi_tr_bl = float(self._phi_tr) + float(self._phi_bl)
            u01_mul = np.cos(phi_tr_bl) + 1j*np.sin(phi_tr_bl)
            phi_tl_br = float(self._phi_tl) + float(self._phi_br)
            u10_mul = np.cos(phi_tl_br) + 1j*np.sin(phi_tl_br)
            phi_bl_br = float(self._phi_bl) + float(self._phi_br)
            u11_mul = np.cos(phi_bl_br) + 1j*np.sin(phi_bl_br)

        umat = self._matrix_template(use_symbolic)
        umat[0, 0] *= u00_mul*cos_theta
        umat[0, 1] *= u01_mul*sin_theta
        umat[1, 1] *= u11_mul*cos_theta
        umat[1, 0] *= u10_mul*sin_theta
        return umat

    def _matrix_template(self, use_symbolic):
        if self._convention == BSConvention.Rx:
            if use_symbolic:
                return Matrix([[1, sp.I], [sp.I, 1]], True)
            return Matrix([[1, 1j], [1j, 1]], False)
        elif self._convention == BSConvention.Ry:
            return Matrix([[1, -1], [1, 1]], use_symbolic)
        elif self._convention == BSConvention.H:
            return Matrix([[1, 1], [1, -1]], use_symbolic)
        raise NotImplementedError(f'Unitary matrix computation not implemented for convention {self._convention.name}')

    def get_variables(self, map_param_kid=None):
        parameters = {}
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "theta", "theta", sp.pi/2, map_param_kid)
        self.variable_def(parameters, "phi_tl", "phi_tl", 0, map_param_kid)
        self.variable_def(parameters, "phi_bl", "phi_bl", 0, map_param_kid)
        self.variable_def(parameters, "phi_tr", "phi_tr", 0, map_param_kid)
        self.variable_def(parameters, "phi_br", "phi_br", 0, map_param_kid)
        return parameters

    def describe(self, map_param_kid=None):
        parameters = self.get_variables(map_param_kid)
        parameters['convention'] = f'BSConvention.{self._convention.name}'
        params_str = format_parameters(parameters, separator=', ')
        return "BS(%s)" % params_str

    def inverse(self, v=False, h=False):
        if not self.defined:
            raise ValueError('Cannot invert BS with variable parameters')
        phi_bl = float(self._phi_bl)
        phi_tl = float(self._phi_tl)
        phi_tr = float(self._phi_tr)
        phi_br = float(self._phi_br)
        if v:
            self._phi_bl.set_value(phi_tl, force=True)
            self._phi_tr.set_value(phi_br, force=True)
            self._phi_tl.set_value(phi_bl, force=True)
            self._phi_br.set_value(phi_tr, force=True)
            # For Rx BS, vertical inversion does not impact theta parameter
            if self._convention == BSConvention.Ry:
                self._theta.set_value(- float(self._theta), force=True)
            elif self._convention == BSConvention.H:
                self._theta.set_value(2*np.pi - float(self._theta), force=True)
        if h:
            self._phi_bl.set_value(-phi_br, force=True)
            self._phi_tr.set_value(-phi_tl, force=True)
            self._phi_tl.set_value(-phi_tr, force=True)
            self._phi_br.set_value(-phi_bl, force=True)
            # For H BS, horizontal inversion does not impact theta parameter
            if self._convention == BSConvention.Rx or self._convention == BSConvention.Ry:
                self._theta.set_value(- float(self._theta), force=True)

    def definition(self):
        return BS(Parameter('theta'), Parameter('phi_tl'), Parameter('phi_bl'), Parameter('phi_tr'),
                  Parameter("phi_br"), self._convention).U


class PS(ACircuit):
    """Phase shifter"""
    DEFAULT_NAME = "PS"

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
                raise ValueError('Cannot invert PS with variable parameters')
            else:
                self._phi.set_value(-float(self._phi), force=True)


class WP(ACircuit):
    """Wave plate"""
    DEFAULT_NAME = "WP"
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
    DEFAULT_NAME = "HWP"

    def __init__(self, xsi):
        super().__init__(sp.pi/2, xsi)

    def definition(self):
        return HWP(xsi=Parameter('xsi')).U


class QWP(WP):
    """Quarter wave plate"""
    DEFAULT_NAME = "QWP"

    def __init__(self, xsi):
        super().__init__(sp.pi/4, xsi)

    def definition(self):
        return QWP(xsi=Parameter('xsi')).U


class PR(ACircuit):
    """Polarization rotator"""
    _supports_polarization = True
    DEFAULT_NAME = "PR"

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


class Unitary(ACircuit):
    """Generic component defined by a unitary matrix"""
    DEFAULT_NAME = "Unitary"

    def __init__(self, U: Matrix, name: str = None, use_polarization: bool = False):
        assert U is not None, "A unitary matrix is required"
        assert U.is_unitary(), "U parameter must be a unitary matrix"
        # A symbolic matrix is not a use case for this component
        assert not U.is_symbolic(), "U parameter must not be symbolic"
        self._u = U
        m = U.shape[0]
        self._supports_polarization = use_polarization
        if use_polarization:
            assert m % 2 == 0, "Polarization matrix should have an even number of rows/col"
            m //= 2
        super().__init__(m, name)

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
        if self.name != Unitary.DEFAULT_NAME:
            params.append(f"name='{self._name}'")
        if self._supports_polarization:
            params.append("use_polarization=True")
        return f"symb.Unitary({', '.join(params)})"


class PERM(Unitary):
    """Permutation"""
    DEFAULT_NAME = "PERM"

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
    _supports_polarization = True
    DEFAULT_NAME = "PBS"

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
