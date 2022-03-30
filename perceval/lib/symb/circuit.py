import sympy as sp

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

    def __init__(self, R=None, theta=None, phi=0):
        super().__init__(2)
        assert R is None or theta is None, "cannot set both R and theta"
        self._phi = self._set_parameter("phi", phi, 0, 2*sp.pi)
        if R is not None:
            self._R = self._set_parameter("R", R, 0, 1)
        else:
            if theta is None:
                theta = sp.pi/4
            self._theta = self._set_parameter("theta", theta, 0, 2*sp.pi)

    def _compute_unitary(self, assign=None, use_symbolic=False):
        self.assign(assign)
        if "R" in self.params:
            cos_theta = sp.sqrt(1-self._R.spv)
            sin_theta = sp.sqrt(self._R.spv)
        else:
            cos_theta = sp.cos(self._theta.spv)
            sin_theta = sp.sin(self._theta.spv)
        return Matrix([[cos_theta, sin_theta*sp.I*sp.exp(-self._phi.spv*sp.I)],
                       [sin_theta*sp.exp(self._phi.spv*sp.I)*sp.I, cos_theta]], use_symbolic)

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

    @property
    def svg_width(self):
        return 2

    width = 2

    def shape(self, content, canvas):
        canvas.add_mpath(["M", 0, 25, "C", 17, 25, 20, 30, 26, 36, "S", 33, 48, 50, 48,
                          "S", 67, 48, 74, 36, "C", 80, 30, 83, 25, 100, 25], stroke="black", stroke_width=2)
        canvas.add_mpath(["M", 0, 75, "C", 17, 75, 20, 70, 26, 64, "S", 33, 52, 50, 52,
                          "S", 67, 52, 74, 64, "C", 80, 70, 83, 75, 100, 75], stroke="black", stroke_width=2)


class DT(ACircuit):
    _name = "DT"
    _fcircuit = Circuit
    delay_circuit = True

    def __init__(self, t):
        super().__init__(1)
        self._dt = self._set_parameter("t", t, 0, sp.oo)

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
        canvas.add_text((25, 38), text=content.replace("t=", ""), size=9, ta="middle", only_svg=True)


class PS(ACircuit):
    _name = "PS"
    _fcircuit = Circuit

    def __init__(self, phi):
        super().__init__(1)
        self._phi = self._set_parameter("phi", phi, 0, sp.pi)

    def _compute_unitary(self, assign=None, use_symbolic=False):
        self.assign(assign)
        U = Matrix([[sp.exp(self._phi.spv*sp.I)]], use_symbolic)
        return U

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
        canvas.add_text((25, 28), text=content.replace("phi=", ""), size=10, ta="middle", only_svg=True)


class PERM(GCircuit):
    _name = "PERM"
    _fcircuit = Circuit

    def __init__(self, perm):
        assert isinstance(perm, list), "permutation Operator needs list parameter"
        assert (min(perm) == 0 and
                max(perm)+1 == len(perm) == len(set(perm)) == len([n for n in perm if isinstance(n, int)])),\
            "%s is not a permutation" % perm
        self._perm = perm
        n = len(perm)
        u = Matrix.zeros((n, n), use_symbolic=True)
        for i, v in enumerate(perm):
            u[i, v] = sp.S(1)
        super().__init__(n, U=u)

    def get_variables(self, _=None):
        return ["_╲ ╱", "_ ╳ ", "_╱ ╲"]

    def describe(self, _=None):
        return "symb.Permutation(%s)" % str(self._perm)

    def definition(self):
        return self.U

    width = 1

    def shape(self, content, canvas):
        lines = []
        for an_input, an_output in enumerate(self._perm):
            canvas.add_mpath(["M", 0, 25+an_input*50,
                              "C", 20, 25+an_input*50, 30, 25+an_output*50, 50, 25+an_output*50],
                             stroke="white", stroke_width=4)
            canvas.add_mpath(["M", 0, 25+an_input*50,
                              "C", 20, 25+an_input*50, 30, 25+an_output*50, 50, 25+an_output*50],
                             stroke="black", stroke_width=2)
