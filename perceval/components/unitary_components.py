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
from __future__ import annotations
import math
import random
from copy import copy
from enum import IntEnum

import numpy as np
import sympy as sp

from .linear_circuit import ACircuit, Circuit
from perceval.utils import Matrix, format_parameters, BasicState, StateVector, Parameter, Expression


class BSConvention(IntEnum):
    """Beam splitter conventions"""
    Rx = 0
    Ry = 1
    H = 2


class BS(ACircuit):
    """Beam Splitter

    Beam splitters couple two spatial modes together, acting on :math:`\\ket{1,0}` and :math:`\\ket{0,1}`.

    :param theta: `theta` parameter
    :param phi_tl: top-left phase parameter
    :param phi_bl: bottom-left phase parameter
    :param phi_tr: top-right phase parameter
    :param phi_br: bottom-right phase parameter
    """
    DEFAULT_NAME = "BS"

    def __init__(self,
                 theta: Parameter | float = sp.pi/2,
                 phi_tl: Parameter | float = 0,
                 phi_bl: Parameter | float = 0,
                 phi_tr: Parameter | float = 0,
                 phi_br: Parameter | float = 0,
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
        return f'{self.DEFAULT_NAME}.{self._convention.name}'

    @property
    def convention(self):
        """Beam splitter convention"""
        return self._convention

    @staticmethod
    def H(theta=sp.pi/2, phi_tl=0, phi_bl=0, phi_tr=0, phi_br=0):
        """
        Convenient named constructor for a Beam Splitter following Hadamard convention.
        Its parameters are the same as the main constructor.
        """
        return BS(theta, phi_tl, phi_bl, phi_tr, phi_br, convention=BSConvention.H)

    @staticmethod
    def Rx(theta=sp.pi / 2, phi_tl=0, phi_bl=0, phi_tr=0, phi_br=0):
        """
        Convenient named constructor for a Beam Splitter following Rotation X convention.
        Its parameters are the same as the main constructor.
        """
        return BS(theta, phi_tl, phi_bl, phi_tr, phi_br, convention=BSConvention.Rx)

    @staticmethod
    def Ry(theta=sp.pi / 2, phi_tl=0, phi_bl=0, phi_tr=0, phi_br=0):
        """
        Convenient named constructor for a Beam Splitter following Rotation Y convention.
        Its parameters are the same as the main constructor.
        """
        return BS(theta, phi_tl, phi_bl, phi_tr, phi_br, convention=BSConvention.Ry)

    @staticmethod
    def r_to_theta(r: float | Parameter) -> float | Expression:
        """Compute theta given a reflectivity value.
        Supports symbolic computing.

        :param r: reflectivity value (can be variable)
        :return: theta value or symbolic expression
        """
        if isinstance(r, Parameter):
            return Expression(f"2*acos(sqrt({r.name}))", r._params)
        return 2*math.acos(math.sqrt(r))

    @staticmethod
    def theta_to_r(theta: float | Parameter) -> float | Expression:
        """
        Compute reflectivity given a theta value.
        Supports symbolic computing.

        :param theta: theta angle (can be variable)
        :return: reflectivity value or symbolic expression
        """
        if isinstance(theta, Parameter) and not theta.defined:
            return Expression(f"cos({theta.name}/2)**2", theta._params)
        return math.cos(float(theta)/2)**2

    @property
    def reflectivity(self):
        """Beam Splitter reflectivity

        :return: reflectivity of the current Beam Splitter
        """
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
            cos_theta = math.cos(float(self._theta)/2)
            sin_theta = math.sin(float(self._theta)/2)
            phi_tl_tr = float(self._phi_tl) + float(self._phi_tr)
            u00_mul = math.cos(phi_tl_tr) + 1j*math.sin(phi_tl_tr)
            phi_tr_bl = float(self._phi_tr) + float(self._phi_bl)
            u01_mul = math.cos(phi_tr_bl) + 1j*math.sin(phi_tr_bl)
            phi_tl_br = float(self._phi_tl) + float(self._phi_br)
            u10_mul = math.cos(phi_tl_br) + 1j*math.sin(phi_tl_br)
            phi_bl_br = float(self._phi_bl) + float(self._phi_br)
            u11_mul = math.cos(phi_bl_br) + 1j*math.sin(phi_bl_br)

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

    def get_variables(self):
        out = {}
        self._populate_parameters(out, "theta", sp.pi / 2)
        self._populate_parameters(out, "phi_tl", 0)
        self._populate_parameters(out, "phi_bl", 0)
        self._populate_parameters(out, "phi_tr", 0)
        self._populate_parameters(out, "phi_br", 0)
        return out

    def describe(self) -> str:
        parameters = self.get_variables()
        params_str = format_parameters(parameters, separator=', ')
        return f"BS.{self._convention.name}({params_str})"

    def inverse(self, v=False, h=False):
        theta = float(self._theta) if self._theta.defined else self._theta
        phi_bl = float(self._phi_bl) if self._phi_bl.defined else self._phi_bl
        phi_tl = float(self._phi_tl) if self._phi_tl.defined else self._phi_tl
        phi_tr = float(self._phi_tr) if self._phi_tr.defined else self._phi_tr
        phi_br = float(self._phi_br) if self._phi_br.defined else self._phi_br

        if v:
            self._phi_bl.set_value(phi_bl, force=True) if self._phi_bl.defined else None
            self._phi_tr.set_value(phi_tr, force=True) if self._phi_tr.defined else None
            self._phi_tl.set_value(phi_tl, force=True) if self._phi_tl.defined else None
            self._phi_br.set_value(phi_br, force=True) if self._phi_br.defined else None

            # For Rx BS, vertical inversion does not impact theta parameter
            if self._convention == BSConvention.Ry:
                if self._theta.defined:
                    self._theta.set_value(-theta, force=True)
                else:
                    self._theta = -theta

            elif self._convention == BSConvention.H:
                if self._theta.defined:
                    self._theta.set_value(2*math.pi - float(self._theta), force=True)
                else:
                    self._theta = 2*math.pi - theta

        if h:
            for param in [self._phi_tl, self._phi_bl, self._phi_tr, self._phi_br]:
                if param.defined:
                    param.set_value(-float(param), force=True)
                else:
                    self._set_parameter(param.name, -param, 0, 4*sp.pi)

            # For H BS, horizontal inversion does not impact theta parameter
            if self._convention == BSConvention.Rx or self._convention == BSConvention.Ry:
                if self._theta.defined:
                    self._theta.set_value(-theta, force=True)
                else:
                    self._theta = self._set_parameter("theta", -theta, 0, 4*sp.pi)

    def definition(self):
        return BS(Parameter('theta'), Parameter('phi_tl'), Parameter('phi_bl'), Parameter('phi_tr'),
                  Parameter("phi_br"), self._convention).U


class PS(ACircuit):
    """Phase shifter

    A phase shifter adds a phase :math:`\\phi` on a spatial mode, which corresponds to a Z rotation in the Bloch sphere.

    :param phi: Phase angle
    :param max_error: Maximum random error to apply. The error is uniformly drawn in
                      :math:`[\\phi - max_{error}, \\phi + max_{error}]`.
                      A global phase error noise parameter can also be set in the `NoiseModel` for all the phase
                      shifters of a given `Experiment`.
    """
    DEFAULT_NAME = "PS"

    def __init__(self, phi: Parameter | float, max_error: Parameter | float = 0):
        super().__init__(1)
        self._phi = self._set_parameter("phi", phi, 0, 2*math.pi)
        self._max_error = self._set_parameter("max_error", max_error, 0, math.pi)

    def _compute_unitary(self, assign=None, use_symbolic=False):
        self.assign(assign)
        if use_symbolic:
            err = self._max_error.spv*random.uniform(-1, 1)
            phase = self._phi.spv + err
            return Matrix([[sp.exp(phase * sp.I)]], True)
        else:
            err = float(self._max_error)*random.uniform(-1, 1)
            phase = float(self._phi) + err
            return Matrix([[math.cos(phase) + 1j * math.sin(phase)]], False)

    def get_variables(self):
        out = {}
        self._populate_parameters(out, "phi")
        if self._max_error:
            self._populate_parameters(out, "max_error")
        return out

    def describe(self):
        params_str = format_parameters(self.get_variables(), separator=', ')
        return f"PS({params_str})"

    def inverse(self, v=False, h=False):
        if h:
            if self._phi.is_symbolic:
                self._phi = self._set_parameter("phi", -self._phi, None, None)
            else:
                self._phi.set_value(-float(self._phi), force=True)


class WP(ACircuit):
    """
    A wave plate acts on the polarisation modes of a single spatial mode.
    This component acts on polarised photons. See also: `Polarisation`

    :param delta: parameter proportional to the thickness of the waveplate
    :param xsi: angle of the waveplate's optical axis in the :math:`\\left\\{\\ket{H}, \\ket{V}\\right\\}` plane.
                Especially important is the case that :math:`\\delta=\\pi/2`, known as a half-wave plate, which rotates
                linear polarisations in the :math:`\\left\\{\\ket{H}, \\ket{V}\\right\\}` plane.
    """
    DEFAULT_NAME = "WP"
    _supports_polarization = True

    def __init__(self, delta: float | Parameter, xsi: float | Parameter):
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
                            math.cos(delta)+1j*math.sin(delta)*math.cos(2*xsi),
                            1j*math.sin(delta)*math.sin(2*xsi)
                           ], [
                            1j * math.sin(delta) * math.sin(2 * xsi),
                            math.cos(delta) - 1j * math.sin(delta) * math.cos(2 * xsi)
                           ]], False)

    def get_variables(self):
        out = {}
        self._populate_parameters(out, "xsi")
        self._populate_parameters(out, "delta")
        return out

    def describe(self):
        params_str = format_parameters(self.get_variables(), separator=', ')
        return f"WP({params_str})"

    def inverse(self, v=False, h=False):
        raise NotImplementedError("inverse not yet implemented")


class HWP(WP):
    """Half wave plate

    This component acts on polarized photons. See also: `Polarization`
    """
    DEFAULT_NAME = "HWP"

    def __init__(self, xsi):
        super().__init__(sp.pi/2, xsi)

    def definition(self):
        return HWP(xsi=Parameter('xsi')).U


class QWP(WP):
    """Quarter wave plate

    This component acts on polarized photons. See also: `Polarization`
    """
    DEFAULT_NAME = "QWP"

    def __init__(self, xsi):
        super().__init__(sp.pi/4, xsi)

    def definition(self):
        return QWP(xsi=Parameter('xsi')).U


class PR(ACircuit):
    """
    A polarisation rotator is an optical device that rotates the polarization axis of a linearly polarized light beam by
    an angle of choice.
    Such devices can be based on the Faraday effect, on bi-refringence, or on total internal reflection.
    Rotators of linearly polarized light have found widespread applications in modern optics since laser beams tend to
    be linearly polarized. It is often necessary to rotate the original polarization to its orthogonal alternative.
    This component acts on polarized photons. See also: `Polarization`

    See https://en.wikipedia.org/wiki/Polarization_rotator for more details.

    :param delta: Rotation angle
    """
    _supports_polarization = True
    DEFAULT_NAME = "PR"

    def __init__(self, delta: float | Parameter):
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
            return Matrix([[math.cos(delta), math.sin(delta)],
                           [-math.sin(delta), math.cos(delta)]], False)

    def get_variables(self):
        out = {}
        self._populate_parameters(out, "delta")
        return out

    def describe(self):
        params_str = format_parameters(self.get_variables(), separator=', ')
        return f"PR({params_str})"

    def inverse(self, v: bool = False, h: bool = False):
        raise NotImplementedError("inverse not yet implemented")


class Unitary(ACircuit):
    """Generic component defined by a unitary matrix

    :param U: numeric matrix. Does not support symbolic computation.
    :param name: Custom name for the component it represents (default is "Unitary").
    :param use_polarization: True if the unitary represents a polarized component.
    """
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

    def describe(self):
        params = [f"Matrix('''{self._u}''')"]
        if self.name != Unitary.DEFAULT_NAME:
            params.append(f"name='{self._name}'")
        if self._supports_polarization:
            params.append("use_polarization=True")
        return f"Unitary({', '.join(params)})"


class PERM(Unitary):
    """Permutation

    A swap between any number of consecutive spatial modes.

    :param perm: Vector of mode index defining the permutation.

    >>> permutation = PERM([2, 3, 1, 0])  # respectively swaps mode 0 to 2, 1 to 3, 2 to 1 and 3 to 0.
    """
    DEFAULT_NAME = "PERM"

    def __init__(self, perm: list[int]):
        assert isinstance(perm, list), "Permutation component requires a list parameter"
        assert (min(perm) == 0 and
                max(perm)+1 == len(perm) == len(set(perm)) == len([n for n in perm if isinstance(n, int)])),\
            "%s is not a permutation" % perm
        n = len(perm)
        u = Matrix.zeros((n, n), use_symbolic=False)
        for i, v in enumerate(perm):
            u[v, i] = 1
        super().__init__(U=u)

    def describe(self):
        return f"PERM({self.perm_vector})"

    def definition(self):
        return self.U

    @property
    def perm_vector(self):
        """Return the permutation vector"""
        nz = np.nonzero(self._u)
        m_list = nz[1].tolist()
        return [m_list.index(i) for i in nz[0]]

    def apply(self, r: tuple[int, ...], sv: BasicState | StateVector):
        """
        Apply the permutation to a state

        :param r: Range of consecutive modes where the permutation occurs
        :sv: State on which the permutation is applied
        """
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
        nsv.update({
            BasicState(state.set_slice(slice(min_r, max_r), BasicState([state[i + min_r] for i in inv]))): prob_ampli
            for state, prob_ampli in sv.items()
        })
        return nsv

    def break_in_2_mode_perms(self):
        """
        Breaks any n-mode permutation into an equivalent circuit made of only 2-mode permutations.

        :return: An equivalent Circuit made of only 2-mode permutations
        """

        perm_vec_req = self.perm_vector
        perm_len = len(perm_vec_req)

        if perm_len == 2:
            return self

        circ = Circuit(perm_len, name="Decomposed PERM")
        new_perm_vec = list(range(perm_len))

        for in_m_pos in range(perm_len):
            out_m_pos = perm_vec_req.index(in_m_pos)
            while new_perm_vec[in_m_pos] != out_m_pos:
                swap_idx = new_perm_vec.index(out_m_pos)
                new_perm_vec[swap_idx], new_perm_vec[swap_idx - 1] = new_perm_vec[swap_idx - 1], new_perm_vec[swap_idx]
                circ.add(swap_idx - 1, PERM([1, 0]))

        return circ


class PBS(Unitary):
    """
    A polarising beam splitter converts a superposition of polarisation modes in a single spatial mode to the
    corresponding equal-polarisation superposition of two spatial modes, and vice versa, and so in this sense allows us
    to translate between polarisation and spatial modes. The unitary matrix associated to a polarising beam splitter
    acting on the tensor product of the spatial mode and the polarisation mode is:

    :math:`\\left[\\begin{matrix}0 & 0 & 1 & 0\\\\0 & 1 & 0 & 0\\\\1 & 0 & 0 & 0\\\\0 & 0 & 0 & 1\\end{matrix}\\right]`


    This component acts on polarized photons. See also: `Polarization`
    """
    _supports_polarization = True
    DEFAULT_NAME = "PBS"

    def __init__(self):
        u = Matrix([[0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])
        super().__init__(U=u, use_polarization=True)

    # noinspection PyMethodMayBeStatic
    def describe(self):
        return "PBS()"


class Barrier(ACircuit):
    """ A barrier is a visual component which has no effect on photons (it behaves as an identity unitary).
    It may be used to separate or align multiple components in a given `Circuit`.

    :param m: Number of consecutive modes it covers
    :param visible: The barrier is rendered if True, and is invisible otherwise
    """
    DEFAULT_NAME = "I"

    def __init__(self, m: int, visible: bool = True):
        assert isinstance(m, int), "Barrier() first parameter has to be an integer (mode count)"
        self._visible = bool(visible)
        super().__init__(m)

    @property
    def visible(self):
        return self._visible

    def _compute_unitary(self, assign: dict = None, use_symbolic: bool = False) -> Matrix:
        return Matrix.eye(self._m)

    def describe(self):
        return f"Barrier({self._m})"

    def definition(self):
        return self.U

    # noinspection PyMethodMayBeStatic
    def apply(self, r, sv):
        return sv

    def inverse(self, v=False, h=False):
        pass
