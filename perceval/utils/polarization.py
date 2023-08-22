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

import numpy as np
import re
import sympy as sp
from typing import Union, Tuple, Any

from .statevector import BasicState
from .matrix import Matrix


class Polarization:
    r"""Polarization class

    This class is defined values used by polarization annotations `P`

    :param v: a string (``[HVADLR]``), a single angle or a pair of angle definition either symbolic or numeric.
       Angles should be in :math:`[0,\pi]` range.
    :raise: `ValueError` if the parameters are out of range, or invalid
    """
    def __init__(self,
                 v: Union[str, Any, Tuple[Any, Any]]):
        if isinstance(v, str):
            if v == "H":
                self.theta_phi = (0, 0)
            elif v == "V":
                self.theta_phi = (sp.pi, 0)
            elif v == "D":
                self.theta_phi = (sp.pi/2, 0)
            elif v == "A":
                self.theta_phi = (sp.pi/2, sp.pi)
            elif v == "R":
                self.theta_phi = (sp.pi/2, 3*sp.pi/2)
            elif v == "L":
                self.theta_phi = (sp.pi/2, sp.pi/2)
            else:
                raise ValueError("undefined value '%s' for polarization" %v)
        elif isinstance(v, tuple):
            if len(v) != 2:
                raise ValueError("Polarization is defined by 2 angles")
            for vs in v:
                if isinstance(vs, sp.Expr):
                    if len(vs.free_symbols):
                        raise ValueError("Polarization cannot have variables")
                elif not isinstance(vs, (int, float, sp.Number)):
                    raise ValueError("incorrect definition for polarization angle: %s" % str(vs))
            if v[0] < sp.S("0") or v[0] > sp.S("pi"):
                raise ValueError("theta should be in [0,pi]")
            if v[0] < sp.S("0") or v[0] >= sp.S("2*pi"):
                raise ValueError("theta should be in [0,2*pi[")
            self.theta_phi = v
        elif isinstance(v, complex):
            self.theta_phi = (v.real, v.imag)
        elif isinstance(v, float) or isinstance(v, int):
            if v < sp.S("0") or v > sp.S("pi"):
                raise ValueError("theta should be in [0,pi]")
            self.theta_phi = (v, 0)
        else:
            raise ValueError("Polarization init should be string or tuple")

    def __complex__(self):
        return self.theta_phi[0]+1j*self.theta_phi[1]

    @staticmethod
    def parse(s: str) -> Polarization:
        r"""Parse a polarization value string

        :param s: should match regex: ``^([HVADLR]|\(theta,phi\)|theta$``
        :return: a `Polarization` instance
        :raise: `ValueError` if the value cannot be parsed, or if parameters are invalid
        """
        if re.match(r"^[HVADLR]$", s):
            return Polarization(s)
        if s[0] == "(":
            if s[-1] != ")":
                raise ValueError("incorrect format - missing closing parenthesis: %s" % s)
            lvs = s[1:-1].split(",")
            if len(lvs) > 2:
                raise ValueError("incorrect format - more than two parameters")
            angles = []
            for v in lvs:
                try:
                    angles.append(float(v))
                except ValueError:
                    try:
                        angles.append(sp.S(v).simplify())
                        if angles[-1].free_symbols:
                            raise ValueError("incorrect format - angle value should not contain variable in %s" % s)
                    except ValueError:
                        raise ValueError("tuple value (%s) have to be numeric expression in %s" % (v, s))
            if len(angles) == 1:
                angles.append(0)
            return Polarization((angles[0], angles[1]))
        else:
            try:
                v = float(s)
            except ValueError:
                try:
                    v = sp.S(s).simplify()
                except ValueError:
                    raise ValueError("value has to be numeric expression in %s" % s)
                if v.free_symbols:
                    raise ValueError("incorrect format - angle value should not contain variable in %s" % s)
        return Polarization((v, 0))

    def project_eh_ev(self, use_symbolic=False) -> Tuple[Any, Any]:
        r"""Build Jones vector corresponding to the current instance

        :return: a pair of numeric or symbolic expressions
        """
        if use_symbolic:
            return sp.cos(self.theta_phi[0]/2), sp.exp(sp.I*self.theta_phi[1])*sp.sin(self.theta_phi[0]/2)
        else:
            return (np.cos(float(self.theta_phi[0])/2),
                    (np.cos(float(self.theta_phi[1])) + 1j * np.sin(float(self.theta_phi[1])))\
                    * np.sin(float(self.theta_phi[0])/2))

    def __str__(self):
        if self.theta_phi[0] == sp.S(0) and self.theta_phi[1] == sp.S(0):
            return "H"
        if self.theta_phi[0] == sp.pi and self.theta_phi[1] == sp.S(0):
            return "V"
        if self.theta_phi[0] == sp.pi/2:
            if self.theta_phi[1] == sp.S(0):
                return "D"
            if self.theta_phi[1] == sp.pi:
                return "A"
            if self.theta_phi[1] == 3*sp.pi/2:
                return "R"
            if self.theta_phi[1] == sp.pi/2:
                return "L"
        if self.theta_phi[1] == 0:
            return str(self.theta_phi[0])
        return "(%s,%s)" % (str(self.theta_phi[0]), str(self.theta_phi[1]))


def _rec_build_spatial_output_states(lfs: list, output: list):
    if len(lfs) == 0:
        yield BasicState(output)
    else:
        if lfs[0] == 0:
            yield from _rec_build_spatial_output_states(lfs[1:], output+[0, 0])
        else:
            for k in range(lfs[0]+1):
                yield from _rec_build_spatial_output_states(lfs[1:], output+[k, lfs[0]-k])


def build_spatial_output_states(state: BasicState):
    yield from _rec_build_spatial_output_states(list(state), [])


def _is_orthogonal(v1, v2, use_symbolic):
    if use_symbolic:
        orth = sp.conjugate(v1[0]) * v2[0] + sp.conjugate(v1[1]) * v2[1]
        return orth == 0
    orth = np.conjugate(v1[0]) * v2[0] + np.conjugate(v1[1]) * v2[1]
    return abs(orth) < 1e-6


def convert_polarized_state(state: BasicState,
                            use_symbolic: bool = False,
                            inverse: bool = False) -> Tuple[BasicState, Matrix]:
    r"""Convert a polarized BasicState into an expanded BasicState vector

    :param inverse:
    :param use_symbolic:
    :param state:
    :return:
    """
    idx = 0
    input_state = []
    prep_matrix = None
    for k_m in range(state.m):
        input_state += [0, 0]
        if state[k_m]:
            vectors = []
            for k_n in range(state[k_m]):
                # for each state we can handle up to two orthogonal vectors
                annot = state.get_photon_annotation(idx)
                idx += 1
                v_hv = Polarization(annot.get("P", complex(Polarization(0)))).project_eh_ev(use_symbolic)
                v_idx = None
                for i, v in enumerate(vectors):
                    if v == v_hv:
                        v_idx = i
                        break
                if v_idx is None:
                    if len(vectors) == 2:
                        raise ValueError("use statevectors to handle more than 2 orthogonal vectors")
                    if len(vectors) == 0 or _is_orthogonal(vectors[0], v_hv, use_symbolic):
                        v_idx = len(vectors)
                        vectors.append(v_hv)
                    else:
                        raise ValueError("use statevectors to handle non orthogonal vectors")
                input_state[-2+v_idx] += 1
            if vectors:
                eh1, ev1 = vectors[0]
                if len(vectors) == 1:
                    if use_symbolic:
                        eh2 = -sp.conjugate(ev1)
                        ev2 = sp.conjugate(eh1)
                    else:
                        eh2 = -np.conjugate(ev1)
                        ev2 = np.conjugate(eh1)
                else:
                    eh2, ev2 = vectors[1]
                if prep_matrix is None:
                    prep_matrix = Matrix.eye(2*state.m, use_symbolic)
                prep_state_matrix = Matrix([[eh1, eh2],
                                            [ev1, ev2]], use_symbolic)
                if inverse:
                    prep_state_matrix = prep_state_matrix.inv()
                prep_matrix[2*k_m:2*k_m+2, 2*k_m:2*k_m+2] = prep_state_matrix
    return BasicState(input_state), prep_matrix
