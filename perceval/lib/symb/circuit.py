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

from perceval.components import Circuit as GCircuit
from perceval.components import ACircuit
from perceval.utils import Matrix


class Circuit(GCircuit):
    _fname = "symb.Circuit"

    def __init__(self, m=None, U=None, name=None):
        super().__init__(m=m, U=U, name=name)

    stroke_style = {"stroke": "black", "stroke_width": 2}
    subcircuit_width = 2
    subcircuit_fill = 'white'
    subcircuit_stroke_style = {"stroke": "black", "stroke_width": 1}


class BS(ACircuit):
    _name = "BS"
    _fcircuit = Circuit
    stroke_style = {"stroke": "black", "stroke_width": 2}

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
                cos_theta = sp.sqrt(1-self._R.spv)
                sin_theta = sp.sqrt(self._R.spv)
            else:
                cos_theta = sp.cos(self._theta.spv)
                sin_theta = sp.sin(self._theta.spv)
            return Matrix([[cos_theta, sin_theta*sp.I*sp.exp(-self._phi.spv*sp.I)],
                           [sin_theta*sp.exp(self._phi.spv*sp.I)*sp.I, cos_theta]], True)
        else:
            if "R" in self.params:
                cos_theta = np.sqrt(1-float(self._R))
                sin_theta = np.sqrt(float(self._R))
            else:
                cos_theta = np.cos(float(self._theta))
                sin_theta = np.sin(float(self._theta))
            return Matrix([[cos_theta, sin_theta*(1j*np.cos(float(self._phi)) - np.sin(float(self._phi)))],
                           [sin_theta*(1j*np.cos(float(self._phi)) - np.sin(float(self._phi))), cos_theta]], False)

    def get_variables(self, map_param_kid=None):
        parameters = []
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
        return "symb.BS(%s)" % ", ".join(parameters)

    width = 2

    def shape(self, content, canvas):
        canvas.add_mpath(["M", 0, 25, "C", 17, 25, 20, 31, 26, 36, "C", 31, 43, 32, 48.5, 50, 48.5,
                          "C", 68, 48.5, 69, 43, 75, 36, "C", 80, 31, 83, 25, 100, 25], stroke="black", stroke_width=2)
        canvas.add_mpath(["M", 0, 75, "C", 17, 75, 20, 69, 26, 64, "C", 31, 57, 32, 51.5, 50, 51.5,
                          "C", 68, 51.5, 69, 57, 75, 64, "C", 80, 69, 83, 75, 100, 75], stroke="black", stroke_width=2)
        canvas.add_text((50, 38), content, 7, "middle")


class DT(ACircuit):
    _name = "DT"
    _fcircuit = Circuit
    delay_circuit = True
    stroke_style = {"stroke": "black", "stroke_width": 2}

    def __init__(self, t):
        super().__init__(1)
        self._dt = self._set_parameter("t", t, 0, sp.oo, False)

    def _compute_unitary(self, assign=None, use_symbolic=False):
        raise RuntimeError("DT circuit cannot be simulated with unitary matrix")

    def get_variables(self, map_param_kid=None):
        parameters = []
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "t", "t", None, map_param_kid)
        return parameters

    def describe(self, map_param_kid=None):
        parameters = self.get_variables(map_param_kid)
        return "phys.DT(%s)" % ", ".join(parameters)

    width = 1

    def shape(self, content, canvas):
        canvas.add_circle((34, 14), 11, stroke="white", stroke_width=3)
        canvas.add_circle((34, 14), 11, stroke="black", stroke_width=2)
        canvas.add_circle((25, 14), 11, stroke="white", stroke_width=3)
        canvas.add_circle((25, 14), 11, stroke="black", stroke_width=2)
        canvas.add_circle((16, 14), 11, stroke="white", stroke_width=3)
        canvas.add_circle((16, 14), 11, stroke="black", stroke_width=2)
        canvas.add_mline([0, 25, 17, 25], stroke="white", stroke_width=3)
        canvas.add_mline([0, 25, 19, 25], stroke="black", stroke_width=2)
        canvas.add_mline([34, 25, 50, 25], stroke="white", stroke_width=3)
        canvas.add_mline([32, 25, 50, 25], stroke="black", stroke_width=2)
        canvas.add_text((25, 38), text=content.replace("t=", ""), size=7, ta="middle")


class PS(ACircuit):
    _name = "PS"
    _fcircuit = Circuit
    stroke_style = {"stroke": "black", "stroke_width": 2}

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
        parameters = []
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "phi", "phi", None, map_param_kid)
        return parameters

    def describe(self, map_param_kid=None):
        parameters = self.get_variables(map_param_kid)
        return "symb.PS(%s)" % ", ".join(parameters)

    width = 1

    def shape(self, content, canvas):
        canvas.add_mline([0, 25, 50, 25], stroke="black", stroke_width=2)
        canvas.add_rect((5, 17), width=40, height=16, stroke="black", stroke_width=2, fill="white")
        canvas.add_text((25, 28), text=content.replace("phi=", ""), size=7, ta="middle")


class PERM(GCircuit):
    _name = "PERM"
    _fcircuit = Circuit
    stroke_style = {"stroke": "black", "stroke_width": 2}

    def __init__(self, perm):
        assert isinstance(perm, list), "permutation Operator needs list parameter"
        assert (min(perm) == 0 and
                max(perm)+1 == len(perm) == len(set(perm)) == len([n for n in perm if isinstance(n, int)])),\
            "%s is not a permutation" % perm
        self._perm = perm
        n = len(perm)
        u = Matrix.zeros((n, n), use_symbolic=False)
        for i, v in enumerate(perm):
            u[i, v] = sp.S(1)
        super().__init__(n, U=u)
        self.width = 1

    def get_variables(self, _=None):
        return ["_╲ ╱", "_ ╳ ", "_╱ ╲"]

    def describe(self, _=None):
        return "symb.PERM(%s)" % str(self._perm)

    def definition(self):
        return self.U

    def shape(self, content, canvas):
        lines = []
        for an_input, an_output in enumerate(self._perm):
            canvas.add_mpath(["M", 0, 25+an_input*50,
                              "C", 20, 25+an_input*50, 30, 25+an_output*50, 50, 25+an_output*50],
                             stroke="white", stroke_width=4)
            canvas.add_mpath(["M", 0, 25+an_input*50,
                              "C", 20, 25+an_input*50, 30, 25+an_output*50, 50, 25+an_output*50],
                             stroke="black", stroke_width=2)
