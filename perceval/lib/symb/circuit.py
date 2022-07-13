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
from perceval.utils import Matrix, format_parameters


class Circuit(GCircuit):
    _fname = "symb.Circuit"

    def __init__(self, m=None, name=None):
        super().__init__(m=m, name=name)

    stroke_style = {"stroke": "black", "stroke_width": 1}
    subcircuit_width = 1
    subcircuit_fill = 'white'
    subcircuit_stroke_style = {"stroke": "black", "stroke_width": 1}


class BS(ACircuit):
    _name = "BS"
    _fcircuit = Circuit
    stroke_style = {"stroke": "black", "stroke_width": 1}

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
            return Matrix([[cos_theta, sin_theta*(1j*np.cos(float(self._phi)) - np.sin(float(self._phi)))],
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

    width = 2

    def get_width(self, compact: bool = False):
        return self.width/2 if compact else self.width

    def shape(self, content, canvas, compact: bool = False):
        if compact:
            path_data = ["M", 6.4721, 25.0002, "c", 6.8548, 0, 6.8241, 24.9998, 13.6789, 24.9998, "m", 0.0009, 0, "c",
                         -6.8558, 0, -6.825, 24.9998, -13.6799, 24.9998, "m", 13.6799, -24.9998, "h", 10.9423, "m", 0,
                         0, "c", 6.8558, 0, 6.825, -24.9998, 13.6799, -24.9998, "m", -13.6799, 24.9998, "c", 6.8558, 0,
                         6.825, 24.9998, 13.6799, 24.9998, "m", -44.7741, -49.9998, "h", 6.5, "m", 0.0009, 49.9998, "h",
                         -6.5009, "m", 43.8227, 0, "h", 6.1773, "m", -6.4028, -50, "h", 6.4028]
        else:
            path_data = ["M", 12.9442, 25.0002, "c", 13.7096, 0, 13.6481, 24.9998, 27.3577, 24.9998, "m", 0.0019, 0,
                         "c", -13.7116, 0, -13.65, 24.9998, -27.3597, 24.9998, "m", 27.3597, -24.9998, "h", 21.8846,
                         "m", 0, 0, "c", 13.7116, 0, 13.65, -24.9998, 27.3597, -24.9998, "m", -27.3597, 24.9998, "c",
                         13.7116, 0, 13.65, 24.9998, 27.3597, 24.9998, "m", -89.5481, -49.9998, "h", 13, "m", 0.0019,
                         49.9998, "h", -13.0019, "m", 87.6453, 0, "h", 12.3547, "m", -12.8056, -50, "h", 12.8056]
        canvas.add_mpath(path_data, stroke="black", stroke_width=1)
        canvas.add_text((25*self.get_width(compact), 38),
                        content.replace('phi=', 'Φ=').replace('theta=', 'Θ='),
                        7, "middle")

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
    _fcircuit = Circuit
    _supports_polarization = True
    stroke_style = {"stroke": "black", "stroke_width": 1}

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

    width = 2

    def get_width(self, compact: bool = False) -> int:
        return self.width/2 if compact else self.width

    def shape(self, content, canvas, compact: bool = False):
        if compact:
            path_data1 = ["M", 0, 25.1, "h", 11.049, "m", -11.049, 50, "h", 10.9375, "m", 27.9029, -50, "h", 11.1596,
                          "m", -11.3283, 50, "h", 11.3283, "m", -11.3283, 0, "c", -10.0446, 0, -17.5781, -50, -27.7341,
                          -50, "m", 27.9029, 0, "c", -10.7156, 0, -17.7467, 50, -27.7914, 50]
            path_data2 = ["M", 30, 50, "l", -4.7404, -5.2543, "l", -4.7404, 5.2543, "l", 4.7404, 5.2543, "l",
                          4.7404, -5.2543, "z", "m", 0.175, 0, "h", -9.6, "z"]
        else:
            path_data1 = ["M", 0, 25.1, "h", 22.0981, "m", -22.0981, 50, "h", 21.8751, "m", 55.8057, -50, "h", 22.3192,
                          "m", -22.6566, 50, "h", 22.6566, "m", -22.6566, 0, "c", -20.0892, 0, -35.1561, -50, -55.4683,
                          -50, "m", 55.8057, 0, "c", -21.4311, 0, -35.4935, 50, -55.5827, 50]
            path_data2 = ["M", 59, 50, "l", -9.4807, -10.5087, "l", -9.4807, 10.5087, "l", 9.4807, 10.5087, "l", 9.4807,
                          -10.5087, "z", "m", 0.35, 0, "h", -19.2, "z"]
        canvas.add_mpath(path_data1, stroke_width=1, stroke="#000")
        canvas.add_mpath(path_data2, stroke_width=1, fill="#fff")
        canvas.add_text((25*self.get_width(compact), 86), text=content, size=7, ta="middle")

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
        parameters = {}
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        self.variable_def(parameters, "t", "t", None, map_param_kid)
        return parameters

    def describe(self, map_param_kid=None):
        parameters = self.get_variables(map_param_kid)
        params_str = format_parameters(parameters, separator=', ')
        return "phys.DT(%s)" % params_str

    width = 1

    def shape(self, content, canvas, compact: bool = False):
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
    stroke_style = {"stroke": "black", "stroke_width": 1}

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

    width = 1

    def shape(self, content, canvas, compact: bool = False):
        canvas.add_mpath(["M", 0, 25, "h", 20, "m", 10, 0, "h", 20], stroke="black", stroke_width=1)
        canvas.add_mpath(["M", 15, 35, "h", 20, "v", -20, "h", -20, "z"],
                         stroke="black", stroke_width=1, fill="lightgray")
        canvas.add_text((25, 44), text=content.replace("phi=", "Φ="), size=7, ta="middle")

    def inverse(self, v=False, h=False):
        if h:
            if self._phi.is_symbolic():
                self._phi = -self._phi.spv
            else:
                self._phi = -float(self._phi)


class Unitary(ACircuit):
    _name = "Unitary"
    _fcircuit = Circuit
    stroke_style = {"stroke": "black", "stroke_width": 1}

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
        self.width = m
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

    def shape(self, _, canvas, compact: bool = False):
        for i in range(self.m):
            canvas.add_mpath(["M", 0, 25 + i*50, "l", 50*self.width, 0], **self.stroke_style)
        radius = 6.25 * self.width  # Radius of the rounded corners
        canvas.add_mpath(
            ["M", 0, radius, "c", 0, 0, 0, -radius, radius, -radius, "l", 6 * radius, 0, "c", radius, 0, radius, radius,
             radius, radius, "l", 0, 6 * radius, "c", 0, 0, 0, radius, -radius, radius, "l", -6 * radius, 0, "c",
             -radius, 0, -radius, -radius, -radius, -radius, "l", 0, -6 * radius],
            **self.stroke_style, fill="lightyellow")
        canvas.add_text((25*self.width, 25*self.m), size=10, ta="middle", text=self._name)


class PERM(Unitary):
    _name = "PERM"
    _fcircuit = Circuit
    stroke_style = {"stroke": "black", "stroke_width": 1}

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
        self.width = 1

    def get_variables(self, _=None):
        return {'PERM': ''}

    def describe(self, _=None):
        return "symb.PERM(%s)" % str(self._compute_perm_vector())

    def definition(self):
        return self.U

    def _compute_perm_vector(self):
        nz = np.nonzero(self._u)
        m_list = nz[1].tolist()
        return [m_list.index(i) for i in nz[0]]

    def shape(self, content, canvas, compact: bool = False):
        for an_input, an_output in enumerate(self._compute_perm_vector()):
            canvas.add_mpath(["M", 0, 24.8 + an_input * 50,
                              "C", 20, 25 + an_input * 50, 30, 25 + an_output * 50, 50, 25 + an_output * 50],
                             stroke="white", stroke_width=2)
            canvas.add_mpath(["M", 0, 25 + an_input * 50,
                              "C", 20, 25 + an_input * 50, 30, 25 + an_output * 50, 50, 25 + an_output * 50],
                             stroke="black", stroke_width=1)


class WP(ACircuit):
    _name = "WP"
    _fcircuit = Circuit
    _supports_polarization = True
    stroke_style = {"stroke": "black", "stroke_width": 1}

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

    width = 1

    def shape(self, content, canvas, compact: bool = False):
        params = content.replace("xsi=", "ξ=").replace("delta=", "δ=").split("\n")
        canvas.add_mpath(["M", 0, 25, "h", 15, "m", 21, 0, "h", 15], stroke="black", stroke_width=1)
        canvas.add_mpath(["M", 15, 45, "h", 21, "v", -40, "h", -21, "z"], stroke="black", stroke_width=1)
        canvas.add_text((25, 55), text=params[0], size=7, ta="middle")
        canvas.add_text((25, 65), text=params[1], size=7, ta="middle")

class HWP(WP):
    _name = "HWP"

    def __init__(self, xsi):
        super().__init__(sp.pi/2, xsi)

    def shape(self, content, canvas, compact: bool = False):
        params = content.replace("xsi=", "ξ=").replace("delta=", "δ=").split("\n")
        canvas.add_mpath(["M", 0, 25, "v", 0, "h", 0, "h", 50],stroke="black", stroke_width=1)
        canvas.add_mpath(["M", 20, 0, "v", 50], stroke="black", stroke_width=2)
        canvas.add_mpath(["M", 30, 0, "v", 50], stroke="black", stroke_width=2)
        canvas.add_text((25, 60), text=params[0], size=7, ta="middle")


class QWP(WP):
    _name = "QWP"

    def __init__(self, xsi):
        super().__init__(sp.pi/4, xsi)

    def shape(self, content, canvas, compact: bool = False):
        params = content.replace("xsi=", "ξ=").replace("delta=", "δ=").split("\n")
        canvas.add_mpath(["M", 0, 25, "v", 0, "h", 0, "h", 50], stroke="black", stroke_width=1)
        canvas.add_mpath(["M", 25, 0, "v", 50], stroke="black", stroke_width=2)
        canvas.add_text((25, 60), text=params[0], size=7, ta="middle")


class PR(ACircuit):
    """Polarization rotator"""
    _name = "PR"
    _fcircuit = Circuit
    _supports_polarization = True
    stroke_style = {"stroke": "black", "stroke_width": 1}

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

    width = 1

    def shape(self, content, canvas, compact: bool = False):
        canvas.add_mpath(["M", 0, 25, "h", 15, "m", 22, 0, "h", 15], stroke="black", stroke_width=1)
        canvas.add_mpath(["M", 15, 36, "h", 22, "v", -22, "h", -22, "z"], stroke="black", stroke_width=1)
        canvas.add_mpath(["M", 19, 27, "c", 0.107, 0.131, 0.280, 0.131, 0.387, 0,
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
                          "H", 17, "c", -0.169, 0, -0.219, 0.106, -0.112, 0.237,
                          "z"
                          ], fill="black",stroke_width=0.1)
        canvas.add_text((27, 50), text=content.replace("delta=", "δ="), size=7, ta="middle")
