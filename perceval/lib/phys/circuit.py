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
from perceval.utils import Matrix, Canvas


class Circuit(GCircuit):
    _fname = "symb.Circuit"

    def __init__(self, m=None, U=None, name=None):
        super().__init__(m=m, U=U, name=name)

    width = 1
    stroke_style = {"stroke": "darkred", "stroke_width": 3}
    subcircuit_width = 2
    subcircuit_fill = 'lightpink'
    subcircuit_stroke_style = {"stroke": "darkred", "stroke_width": 1}


class BS(ACircuit):
    _name = "BS"
    _fcircuit = Circuit
    stroke_style = {"stroke": "darkred", "stroke_width": 3}

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
        parameters = []
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
        return "phys.BS(%s)" % ", ".join(parameters)

    width = 2

    def shape(self, content, canvas, compact: bool = False):
        head_content = "\n".join([s for s in content.split("\n")
                                    if s.startswith("R=") or s.startswith("theta=")])
        bottom_content = "\n".join([s for s in content.split("\n")
                                      if not s.startswith("R=") and not s.startswith("theta=")]).replace("\n", ", ")
        canvas.add_mline([0, 25, 28, 25, 47, 44], stroke="darkred", stroke_width=3, stroke_linejoin="round")
        canvas.add_mline([53, 44, 72, 25, 100, 25], stroke="darkred", stroke_width=3, stroke_linejoin="round")
        canvas.add_mline([0, 75, 28, 75, 47, 56], stroke="darkred", stroke_width=3, stroke_linejoin="round")
        canvas.add_mline([53, 56, 72, 75, 100, 75], stroke="darkred", stroke_width=3, stroke_linejoin="round")
        canvas.add_rect((25, 43), 50, 14, fill="black")
        canvas.add_text((50, 86), size=7, ta="middle", text=bottom_content)
        canvas.add_text((50, 26), size=7, ta="middle", text=head_content)
        if self._phi_b.defined:
            m = round(abs(float(self._phi_b.spv/sp.pi)))
            if (m + 1) % 2:
                canvas.add_rect((25, 43), 50, 4, fill="lightgray")
            else:
                canvas.add_rect((25, 53), 50, 4, fill="lightgray")

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

class PBS(ACircuit):
    _name = "PBS"
    _fcircuit = Circuit
    _supports_polarization = True
    stroke_style = {"stroke": "darkred", "stroke_width": 3}

    def __init__(self):
        super().__init__(2)

    def _compute_unitary(self, assign=None, use_symbolic=False):
        self.assign(assign)
        return Matrix([[0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]], use_symbolic)

    def get_variables(self, map_param_kid=None):
        return []

    # TODO: make method static
    def describe(self, _=None):
        return "phys.PBS()"

    width = 2

    def shape(self, content, canvas, compact: bool = False):
        canvas.add_mline([0, 25, 28, 25, 37.5, 37.5], stroke="darkred", stroke_width=3, stroke_linejoin="round")
        canvas.add_mline([62.5, 37.5, 72, 25, 100, 25], stroke="darkred", stroke_width=3, stroke_linejoin="round")
        canvas.add_mline([0, 75, 28, 75, 37.5, 62.5], stroke="darkred", stroke_width=3, stroke_linejoin="round")
        canvas.add_mline([62.5, 62.5, 72, 75, 100, 75], stroke="darkred", stroke_width=3, stroke_linejoin="round")
        canvas.add_mline([62.5, 62.5, 72, 75, 100, 75], stroke="darkred", stroke_width=3, stroke_linejoin="round")
        canvas.add_polygon([25, 50, 50, 24, 75, 50, 50, 76, 25, 50], stroke="black", stroke_width=1, fill="gray")
        canvas.add_mline([25, 50, 75, 50], stroke="black", stroke_width=1)
        canvas.add_text((50, 86), text=content, size=7, ta="middle")


class PS(ACircuit):
    _name = "PS"
    _fcircuit = Circuit
    stroke_style = {"stroke": "darkred", "stroke_width": 3}

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
        return "phys.PS(%s)" % ", ".join(parameters)

    width = 1

    def shape(self, content, canvas, compact: bool = False):
        canvas.add_mline([0, 25, 50, 25], stroke_width=3, stroke="darkred")
        canvas.add_polygon([5, 40, 14, 40, 28, 10, 19, 10, 5, 40, 14, 40],
                           stroke="black", fill="gray", stroke_width=1, stroke_linejoin="miter")
        canvas.add_text((22, 38), text=content.replace("phi=", "Φ="), size=7, ta="left")

    def inverse(self, v=False, h=False):
        if h:
            if self._phi.is_symbolic():
                self._phi = -self._phi.spv
            else:
                self._phi = -float(self._phi)

class WP(ACircuit):
    _name = "WP"
    _fcircuit = Circuit
    _supports_polarization = True
    stroke_style = {"stroke": "darkred", "stroke_width": 3}

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
        parameters = []
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "xsi", "xsi", None, map_param_kid)
        self.variable_def(parameters, "delta", "delta", None, map_param_kid)
        return parameters

    def describe(self, map_param_kid=None):
        parameters = self.get_variables(map_param_kid)
        return "phys.WP(%s)" % ", ".join(parameters)

    width = 1

    def shape(self, content, canvas, compact: bool = False):
        params = content.replace("xsi=", "ξ=").replace("delta=", "δ=").split("\n")
        canvas.add_mline([0, 25, 50, 25], stroke_width=3, stroke="darkred")
        canvas.add_rect((13, 7), width=14, height=36, fill="gray",
                        stroke_width=1, stroke="black", stroke_linejoin="miter")
        canvas.add_mline([20, 7, 20, 43], stroke="black", stroke_width=1)
        canvas.add_text((28.5, 36), text=params[0], size=7, ta="left")
        canvas.add_text((28.5, 45), text=params[1], size=7, ta="left")


class PR(ACircuit):
    """Polarization rotator"""
    _name = "PR"
    _fcircuit = Circuit
    _supports_polarization = True
    stroke_style = {"stroke": "darkred", "stroke_width": 3}

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
        parameters = []
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "delta", "delta", None, map_param_kid)
        return parameters

    def describe(self, map_param_kid=None):
        parameters = self.get_variables(map_param_kid)
        return "phys.PR(%s)" % ", ".join(parameters)

    width = 1

    def shape(self, content, canvas, compact: bool = False):
        canvas.add_mline([0, 25, 15, 25], stroke="darkred", stroke_width=3)
        canvas.add_mline([35, 25, 50, 25], stroke="darkred", stroke_width=3)
        canvas.add_rect((14, 14), width=22, height=22, stroke="black", fill="lightgray",
                        stroke_width=1, stroke_linejoin="miter")
        canvas.add_mpath(["M", 18, 27, "c", 0.107, 0.131, 0.280, 0.131, 0.387, 0,
                          "l", 2.305, -2.821, "c", 0.107, -0.131, 0.057, -0.237, -0.112, -0.237,
                          "h", -1.22, "c", -0.169, 0, -0.284, -0.135, -0.247, -0.300,
                          "c", 0.629, -2.866, 3.187, -5.018, 6.240, -5.018,
                          "c", 3.524, 0, 6.39, 2.867, 6.390, 6.3902,
                          "c", 0, 3.523, -2.866, 6.39, -6.390, 6.390,
                          "c", -0.422, 0, -0.765, 0.342, -0.765, 0.765,
                          "s", 0.342, 0.765, 0.765, 0.765,
                          "c", 4.367, 0, 7.92, -3.552, 7.920, -7.920,
                          "c", 0, -4.367, -3.552, -7.920, -7.920, -7.920,
                          "c", -3.898, 0, -7.146, 2.832, -7.799, 6.546,
                          "c", -0.029, 0.166, -0.184, 0.302, -0.353, 0.302,
                          "H", 16, "c", -0.169, 0, -0.219, 0.106, -0.112, 0.237,
                          "z"
                          ], fill="black")
        canvas.add_text((25, 45), text=content.replace("delta=", "δ="), size=7, ta="middle")


class HWP(WP):
    _name = "HWP"

    def __init__(self, xsi):
        super().__init__(sp.pi/2, xsi)


class QWP(WP):
    _name = "QWP"

    def __init__(self, xsi):
        super().__init__(sp.pi/4, xsi)


class DT(ACircuit):
    _name = "DT"
    _fcircuit = Circuit
    delay_circuit = True
    stroke_style = {"stroke": "darkred", "stroke_width": 3}

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

    def shape(self, content, canvas: Canvas, compact: bool = False):
        canvas.add_circle((34, 14), 11, stroke_width=5, fill=None, stroke="white")
        canvas.add_circle((34, 14), 11, stroke_width=3, fill=None, stroke="darkred")
        canvas.add_circle((25, 14), 11, stroke_width=5, fill=None, stroke="white")
        canvas.add_circle((25, 14), 11, stroke_width=3, fill=None, stroke="darkred")
        canvas.add_circle((16, 14), 11, stroke_width=5, fill=None, stroke="white")
        canvas.add_circle((16, 14), 11, stroke_width=3, fill=None, stroke="darkred")
        canvas.add_mline([0, 25, 19, 25], stroke="white", stroke_width=5)
        canvas.add_mline([0, 25, 19, 25], stroke="darkred", stroke_width=3)
        canvas.add_mline([34, 25, 50, 25], stroke="white", stroke_width=5)
        canvas.add_mline([32, 25, 50, 25], stroke="darkred", stroke_width=3)
        canvas.add_text((25, 38), content, 7, "middle")


class PERM(GCircuit):
    _name = "PERM"
    _fcircuit = Circuit
    stroke_style = {"stroke": "darkred", "stroke_width": 3}

    def __init__(self, perm):
        assert isinstance(perm, list), "permutation Operator needs list parameter"
        assert (min(perm) == 0 and
                max(perm)+1 == len(perm) == len(set(perm)) == len([n for n in perm if isinstance(n, int)])),\
            "%s is not a permutation" % perm
        self._perm = perm
        n = len(perm)
        u = Matrix.zeros((n, n), use_symbolic=False)
        for v, i in enumerate(perm):
            u[i, v] = 1
        super().__init__(n, U=u)
        self.width = 1

    def get_variables(self, _=None):
        return ["_╲ ╱", "_ ╳ ", "_╱ ╲"]

    def describe(self, _=None):
        return "phys.PERM(%s)" % str(self._perm)

    def definition(self):
        return self.U

    def shape(self, content, canvas, compact: bool = False):
        lines = []
        for an_input, an_output in enumerate(self._perm):
            canvas.add_mline([3, 25+an_input*50, 47, 25+an_output*50],
                             stroke="white", stroke_width=6)
            canvas.add_mline([0, 25+an_input*50, 3, 25+an_input*50, 47, 25+an_output*50, 50,25+an_output*50],
                             stroke="darkred", stroke_width=3)
        return "\n".join(lines)
