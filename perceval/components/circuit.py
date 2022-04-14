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

from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import random
from typing import Callable, Literal, Optional, Union, Tuple, Type

import numpy as np
import scipy.optimize as so
import sympy as sp

from perceval.utils import QPrinter, Parameter, Matrix, simple_float, Canvas, global_params


def _matrix_double_for_polarization(m, u):
    pu = Matrix(m * 2, u.is_symbolic())
    pu.fill(0)
    for k in range(0, m):
        for l in range(0, m):
            pu[2 * k, 2 * l] = u[k, l]
            pu[2 * k + 1, 2 * l + 1] = u[k, l]
    return pu


class ACircuit(ABC):
    """
        Abstract circuit class. A circuit is defined by a dimension `n`, and by parameters.
        Parameters can be fixed (value) or variables.
        A circuit is either simple, or is defined by a list of `(portlist, component)`
    """

    delay_circuit = False
    _supports_polarization = False

    def __init__(self, m: int, params=None):
        if params is None:
            params = {}
        self._m = m
        self._params = params
        for name, p in self._params.items():
            assert type(name) == str and isinstance(p, Parameter), "invalid parameter definition"
        self._components = []
        self._params = {}
        self._vars = {}
        self.defined_circuit = True

    @abstractmethod
    def _compute_unitary(self,
                        assign: dict = None,
                        use_symbolic: bool = False) -> Matrix:
        """Compute the unitary matrix corresponding to the current circuit

        :param assign: assign values to some parameters
        :param use_symbolic: if the matrix should use symbolic calculation
        :return: the unitary matrix, will be a :class:`~perceval.utils.matrix.MatrixS` if symbolic, or a ~`MatrixN` if not.
        """

    def compute_unitary(self,
                        assign: dict = None,
                        use_symbolic: bool = False,
                        use_polarization: Optional[bool] = None) -> Matrix:
        """Compute the unitary matrix corresponding to the current circuit

        :param use_polarization:
        :param assign: assign values to some parameters
        :param use_symbolic: if the matrix should use symbolic calculation
        :return: the unitary matrix, will be a :class:`~perceval.utils.matrix.MatrixS` if symbolic, or a ~`MatrixN` if not.
        """
        if self._supports_polarization:
            assert use_polarization is not False, "polarized circuit cannot generates non-polarized unitary"
            use_polarization = True
        elif use_polarization is None:
            use_polarization = False
        u = self._compute_unitary(assign, use_symbolic)
        if use_polarization and not self._supports_polarization:
            return _matrix_double_for_polarization(self._m, u)
        return u

    def assign(self,
               assign: dict = None):
        if assign is None:
            return
        vs = self.vars
        if isinstance(assign, dict):
            for k, v in assign.items():
                vs[k].set_value(v)

    @property
    def requires_polarization(self):
        return self._supports_polarization

    @property
    def U(self):
        """
        get the numeric unitary matrix, circuit has to be defined
        """
        return self.compute_unitary(use_symbolic=True).simp()

    @property
    def defined(self):
        """
            check if all parameters of the circuit are fully defined
        """
        for _, p in self._params.items():
            if not p.defined:
                return False
        return True

    @property
    def params(self):
        return self._params.keys()

    def get_parameters(self, all_params: bool = False) -> list[str]:
        """Return the parameters of the circuit

        :param all_params: if False, only returns the variable parameters
        :return: the name of the parameters
        """
        return [v for v in self._params.values() if all_params or not v.fixed]

    def _set_parameter(self,
                       name: str,
                       p: Parameter,
                       min_v: float,
                       max_v: float,
                       periodic: bool=True):
        """
            Define a new parameter for the circuit, it can be an existing parameter that we recycle updating
            min/max value or a parameter defined by a value that we create on the fly
        :param name:
        :param p:
        :param min_v:
        :param max_v:
        :param periodic:
        :return:
        """
        if isinstance(p, Parameter):
            if min_v is not None:
                if p.min is None or min_v > p.min:
                    p.min = min_v
            if max_v is not None:
                if p.max is None or max_v < p.max:
                    p._max = max_v
            if p.name in self._vars:
                if p.pid != self._vars[p.name].pid:
                    raise RuntimeError("two parameters with the same name in the circuit")
            self._vars[p.name] = p
        else:
            p = Parameter(value=p, name=name, min_v=min_v, max_v=max_v, periodic=periodic)
        self._params[name] = p
        return p

    @property
    def m(self):
        return self._m

    def definition(self):
        params = {name: Parameter(name) for name in self._params.keys()}
        return type(self)(**params).U

    def add(self, port_range: Union[int, Tuple[int]], component: Circuit, merge: bool = None) -> Circuit:
        r"""Add a component in a circuit

        :param port_range: the port range as a tuple of consecutive porst, or the initial port where to add the component
        :param component: the component to add, must be a circuit
        :param merge: if the component is a complex circuit,
        :return: the circuit itself, allowing to add multiple components in a same line
        :raise: ``AssertionError`` if parameters are not valid
        """
        if isinstance(port_range, int):
            port_range = list([i for i in range(port_range, port_range+component.m)])
        assert isinstance(port_range, list) or isinstance(port_range, tuple), "range (%s) must be a list"
        if not isinstance(self, Circuit):
            self._components = [(list(range(self._m)), copy.copy(self))]
            self.__class__ = self._fcircuit
            self._Udef = None
            self._params = {v.name: v for k, v in self._params.items() if not v.fixed}
        if not self.defined_circuit:
            self.defined_circuit = True
            self.stroke_style = component.stroke_style
        for i, x in enumerate(port_range):
            assert isinstance(x, int) and i == 0 or x == port_range[i - 1] + 1 and x < self._m,\
                "range must a consecutive valid set of ports"
        assert len(port_range) == component.m, \
            "range port (%d) is not matching component size (%d)" % (len(port_range), component.m)
        # merge the parameters - we are only interested in non-assigned parameters if it is not a global operator
        for _, p in component._params.items():
            if not p.fixed:
                if p.name in self._params and p._pid != self._params[p.name]._pid:
                    raise RuntimeError("two parameters with the same name in the circuit")
                self._params[p.name] = p
        if merge is None:
            merge = len(port_range) != self._m
        # register the component
        if merge and component._components:
            for sprange, sc in component._components:
                nprange = tuple(r + port_range[0] for r in sprange)
                self._components.append((nprange, sc))
        else:
            self._components.append((port_range, component))
        return self

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, value):
        self._params[key] = value

    def __ifloordiv__(self, component: Union[Circuit, Tuple[int, Circuit]]) -> Circuit:
        r"""Shortcut for ``.add``

        >>> c //= b       # equivalent to: `c.add((0:b.n),b)`
        >>> c //= (i, b)  # equivalent to: `c.add((i:i+b.n), b)`

        :param component: the component to add, or a tuple (first_port, component)

        """
        if isinstance(component, tuple):
            assert len(component) == 2 and isinstance(component[0], int), "invalid //(i,C) operation"
            pos = component[0]
            component = component[1]
        else:
            pos = 0
        self.add(tuple(range(pos, component._m+pos)), component)
        return self

    def __floordiv__(self, component: Union[Circuit, Tuple[int, Circuit]]) -> Circuit:
        r"""Build a new circuit by adding `component` to the current circuit

        >>> c = a // b   # equivalent to: `Circuit(n) // self // component`

        :param component: the component to add, or a tuple (first_port, component)
        """
        c = copy.copy(self)
        c //= component
        return c

    def __iter__(self):
        """
        Iterator on a circuit, recursively returns components applying in circuit order
        """
        if self._components:
            for r, c in self._components:
                for range_comp, comp in c:
                    yield tuple(pos+r[0] for pos in range_comp), comp
        else:
            yield tuple(pos for pos in range(self._m)), self

    def variable_def(self, parameters, k, pname, default_value, map_param_kid=None):
        if map_param_kid is None:
            map_param_kid = {}
        if self._params[k].defined:
            if default_value is None or self._params[k]._value != default_value:
                parameters.append("%s=%s" % (pname, simple_float(self._params[k]._value)[1]))
        else:
            parameters.append("%s=%s" % (pname, map_param_kid[self._params[k]._pid]))

    def get_variables(self, _=None):
        return []

    @property
    def vars(self):
        return {p.name: p for p in self._params.values() if not p.fixed}

    def map_parameters(self):
        map_param_kid = {}
        for k, p in self._params.items():
            if not p.defined:
                map_param_kid[p._pid] = p.name
        return map_param_kid

    def identify(self, unitary_matrix, phases):
        r"""Identify an instance of the current circuit (should be parameterized) such as Q.C=U.P where Q and P
        are single-mode phase shifts (resp. [q1, q2, ..., qn], and [p1, p2, ...,pn])

        this is solved through n^2 equations:
            C_ij(x,y, ...) * q_i = UP_ij * pj

        Parameters
        ----------
        unitary_matrix :
        phases :

        Returns
        -------

        """
        params = [x.spv for x in self.get_parameters()]
        Q = Matrix.eye(self._m, use_symbolic=True)
        P = Matrix.eye(self._m)
        for i in range(self._m):
            params.append(sp.S("q%d"%i))
            Q[i, i] = sp.exp(1j*params[-1])
            P[i, i] = phases[i]
        cU = Q @ self.compute_unitary(use_symbolic=True)
        UP = unitary_matrix @ P
        equations = []
        for i in range(self._m):
            for j in range(self._m):
                equations.append(sp.re(abs(cU[i, j]-UP[i, j])))
        f = sp.lambdify([params], equations, "numpy")
        counter = 0
        while counter < 100:
            x0 = [random.random() * 2 * np.pi - np.pi] * len(params)
            res, _, ier, _ = so.fsolve(f, [x0], full_output=True)
            if ier == 1:
                break
            counter += 1
        if ier != 1:
            return None
        return res[:len(res)-self._m], res[-self._m:]

    def pdisplay(self,
                 parent_td: QPrinter = None,
                 map_param_kid: dict = None,
                 shift: int = 0,
                 output_format: Literal["text", "html", "mplot", "latex"] = "text",
                 recursive: bool = False,
                 dry_run: bool = False,
                 **opts):
        if parent_td is None:
            if not dry_run:
                total_width = self.pdisplay(parent_td, map_param_kid, shift, output_format, recursive, True, **opts)
                td = QPrinter(self._m, output_format=output_format, stroke_style=self.stroke_style,
                              total_width=total_width, total_height=self._m, **opts)
            else:
                td = QPrinter(self._m, output_format="html", stroke_style=self.stroke_style, **opts)
        else:
            td = parent_td
        if map_param_kid is None:
            map_param_kid = self.map_parameters()

        if hasattr(self, "width"):
            description = self.get_variables(map_param_kid)
            td.append_circuit([p + shift for p in range(self._m)], self, "\n".join(description))

        if self._components:
            for r, c in self._components:
                shiftr = [p+shift for p in r]
                if c._components and recursive:
                    td.open_subblock(r, c._name)
                    c.pdisplay(td, shift=shiftr[0])
                    td.close_subblock(r)
                elif c._components:
                    description = c.get_variables(map_param_kid)
                    td.append_subcircuit(shiftr, c, "\n".join(description))
                else:
                    description = c.get_variables(map_param_kid)
                    td.append_circuit(shiftr, c, "\n".join(description))

        td.extend_pos(0, self._m - 1)

        if parent_td is None:
            td.close()
            if dry_run:
                return td.max_pos(0, self._m-1, True)
            else:
                return td.draw()

    stroke_style = {"stroke": "darkred", "stroke_width": 3}
    subcircuit_width = 2
    subcircuit_fill = 'lightpink'
    subcircuit_stroke_style = {"stroke": "darkred", "stroke_width": 1}


    def shape(self, content, canvas: Canvas):
        return """
            <rect x=0 y=5 width=100 height=%d fill="lightgray"/>
        """ % (self._m * 50 - 10)


class Circuit(ACircuit):
    """Class to represent any circuit composed of one or multiple components

    :param m: The number of port of the circuit, if omitted the parameter `U` should be defined
    :param U: Unitary matrix defining the circuit
    :param name: Name of the circuit
    :param use_polarization: defines if the circuit should be used with Polarized states. This value is automatically
      induced when a component working on polarization is added to the circuit
    """
    _name = "CPLX"
    _fname = "Circuit"

    def __init__(self, m: int = None, U: Matrix = None, name: str = None, use_polarization: bool = False):
        if U is not None:
            assert len(U.shape) == 2 and U.shape[0] == U.shape[1], "invalid unitary matrix"
            if m:
                assert U.shape[0] == m, "incorrect size"
            else:
                m = U.shape[0]
            self.width = m
            # check if unitary matrix
            self._Udef = U
            self._udef_use_polarization = use_polarization
            if use_polarization:
                assert m % 2 == 0, "polarization matrix should have even number of rows/col"
                m /= 2
        else:
            assert m > 0, "invalid size"
            self._Udef = None
        if name is not None:
            self._name = name
        super().__init__(m)
        if U is None:
            self.defined_circuit = False

    def describe(self, map_param_kid=None) -> str:
        r"""Describe a circuit

        :param map_param_kid: internal parameter
        :return: a string describing the circuit that be re-used to define the circuit
        """
        cparams = ["%d" % self._m]
        if self._name != "CPLX":
            cparams.append("name=%s" % self._name)
        s = "%s(%s)" % (self._fname, ", ".join(cparams))
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
        for r, c in self._components:
            if len(r) > 1:
                srange = str(r)
            else:
                srange = str(r[0])
            s += ".add("+srange+", "+c.describe(map_param_kid)+")"
        return s

    @property
    def requires_polarization(self) -> bool:
        """Does the circuit requires polarization

        :return: is True if the circuit is defined with ``use_polarization=True`` or if the circuit has a polarization
          component
        """
        if self._Udef is not None and self._udef_use_polarization:
            return True
        else:
            for _, c in self._components:
                if c.requires_polarization:
                    return True
        return False

    def definition(self) -> Matrix:
        r"""Gives mathematical definition of the circuit

        Only defined for elementary circuits
        """
        raise RuntimeError("`definition` method is only available on elementary circuits")

    def _compute_unitary(self,
                         assign: dict = None,
                         use_symbolic: bool = False) -> Matrix:
        pass

    def _compute_circuit_unitary(self,
                                 use_symbolic: bool,
                                 use_polarization: bool) -> Matrix:
        """compute the unitary matrix corresponding to the current circuit"""
        if self._Udef is not None:
            u = self._Udef
            if use_polarization and not self._udef_use_polarization:
                u = _matrix_double_for_polarization(self._m, u)
        else:
            u = None
        if use_polarization:
            multiplier = 2
        else:
            multiplier = 1
        for r, c in self._components:
            cU = c.compute_unitary(use_symbolic=use_symbolic, use_polarization=use_polarization)
            if len(r) != multiplier*self._m:
                nU = Matrix.eye(multiplier*self._m, use_symbolic)
                nU[multiplier*r[0]:multiplier*(r[-1]+1), multiplier*r[0]:multiplier*(r[-1]+1)] = cU
                cU = nU
            if u is None:
                u = cU
            else:
                u = cU @ u
        return u

    def compute_unitary(self,
                        assign: dict = None,
                        use_symbolic: bool = False,
                        use_polarization: Optional[bool] = None) -> Matrix:
        r"""Compute the unitary matrix corresponding to the circuit

        :param assign:
        :param use_symbolic:
        :param use_polarization:
        :return:
        """
        self.assign(assign)
        if use_polarization is None:
            use_polarization = self.requires_polarization
        elif not use_polarization:
            assert self.requires_polarization is False, "polarized circuit cannot generates non-polarized unitary"
        u = self._compute_circuit_unitary(use_symbolic, use_polarization)
        if u is None:
            u = Matrix.eye(self._m, use_symbolic=use_symbolic)
        return u

    def subcircuit_shape(self, content, canvas):
        for idx in range(self._m):
            canvas.add_mline([0, 50*idx+25, self.subcircuit_width*50, 50*idx+25], **self.stroke_style)
        canvas.add_rect((2.5, 2.5), self.subcircuit_width * 50 - 5, 50 * self._m - 5,
                        fill=self.subcircuit_fill, **self.subcircuit_stroke_style)
        canvas.add_text((16, 16), content.upper(), 8)

    @staticmethod
    def generic_interferometer(m: int,
                               fun_gen: Callable[[int], Circuit],
                               shape: Literal["triangle", "rectangle"] = "rectangle",
                               depth: int = None) -> Circuit:
        r"""Generate a generic interferometer with up to :math:`\frac{n(n+1)}{2}` elements

        :param m: number of modes
        :param fun_gen: generator function for the building components
        :param shape: `rectangle` (Clements-like interferometer) or `triangle` (Reck-like)
        :param depth: if not None, maximal generation depth
        :return: a circuit

        """
        generated = Circuit(m)
        idx = 0
        depths = [0] * m
        if shape == "rectangle":
            for i in range(0, m):
                for j in range(0+i % 2, m-1, 2):
                    if depth is not None and (depths[j] == depth or depths[j+1] == depth):
                        continue
                    generated.add((j, j+1), fun_gen(idx))
                    depths[j] += 1
                    depths[j+1] += 1
                    idx += 1
        else:
            for i in range(1, m):
                for j in range(i, 0, -1):
                    if depth is not None and (depths[j-1] == depth or depths[j] == depth):
                        continue
                    generated.add((j-1, j), fun_gen(idx))
                    depths[j-1] += 1
                    depths[j] += 1
                    idx += 1

        return generated

    @staticmethod
    def decomposition(U: Matrix,
                      component: Circuit,
                      phase_shifter_fn: Callable[[float], Circuit] = None,
                      shape: Literal["triangle", "rectangle"] = "triangle",
                      permutation: Type[Circuit] = None,
                      merge: bool = True,
                      precision: float = None):
        r"""Decompose a given unitary matrix U into a circuit with specified component type

        :param component: a circuit, must have 2 independent parameters
        :param phase_shifter_fn: a function generating a phase_shifter circuit. If `None`, residual phase will be
            ignored
        :param shape: `rectangle` (Clements-like interferometer) or `triangle` (Reck-like)
        :param permutation: if provided, type of a permutation operator to avoid unnecessary operators
        :param merge: don't use sub-circuits
        :param precision: for intermediate values - norm below precision are considered 0. If not - use `global_params`
        :return: a circuit
        """
        M = copy.deepcopy(U)
        N = M.shape[0]
        C = Circuit(N)
        params = component.get_parameters()
        params_symbols = [x.spv for x in params]
        assert len(params) == 2, "Component should have 2 independent parameters"

        if precision is None:
            precision = global_params["min_complex_component"]

        U = component.U
        UI = U.inv()

        if shape == "triangle":
            assert len(set.union(UI[0, 0].free_symbols, UI[0, 0].free_symbols)) == 2, "Component parameters cannot" \
                    "perform triangle decomposition"

        if shape == "rectangle":
            list_components_right = []
            list_components_left = []

            for m in range(N-1):
                for k in range(m+1):
                    if m % 2 == 0:
                        # on even iterations right null with UI, goal is to null M[N-1-k,m-k]
                        if abs(M[N-1-k, m-k]) > precision:
                            equation = M[N-1-k, m-k] * UI[0, 0] + M[N-1-k, m-k+1] * UI[1, 0]
                            f = sp.lambdify([params_symbols], [sp.re(equation), sp.im(equation)])
                            counter = 0
                            while counter < 100:
                                x0 = [random.random() * 2 * np.pi - np.pi, random.random() * 2 * np.pi - np.pi]
                                res, _, ier, _ = so.fsolve(f, x0, full_output=True)
                                if ier == 1:
                                    break
                                counter += 1
                            if ier != 1:
                                return None

                            RI = Matrix.eye(N, use_symbolic=False)
                            RI[m-k, m-k] = complex(
                                UI[0, 0].subs({params_symbols[0]: res[0], params_symbols[1]: res[1]}))
                            RI[m-k, m-k+1] = complex(
                                UI[0, 1].subs({params_symbols[0]: res[0], params_symbols[1]: res[1]}))
                            RI[m-k+1, m-k] = complex(
                                UI[1, 0].subs({params_symbols[0]: res[0], params_symbols[1]: res[1]}))
                            RI[m-k+1, m-k+1] = complex(
                                UI[1, 1].subs({params_symbols[0]: res[0], params_symbols[1]: res[1]}))

                            M = M @ RI
                            instantiated_component = copy.deepcopy(component)
                            instantiated_component.get_parameters()[0].fix_value(res[0])
                            instantiated_component.get_parameters()[0].fix_value(res[1])
                            list_components_right = list_components_right+[((m-k, m-k+1), instantiated_component)]
                    else:
                        # on odd iterations left null with U, goal is to null M[N-1-m+k,k]
                        if abs(M[N-1-m+k, k]) > precision:
                            equation = U[1, 0] * M[N-2-m+k, k] + U[1, 1] * M[N-1-m+k, k]
                            f = sp.lambdify([params_symbols], [sp.re(equation), sp.im(equation)])
                            counter = 0
                            while counter < 100:
                                x0 = [random.random() * 2 * np.pi - np.pi, random.random() * 2 * np.pi - np.pi]
                                res, _, ier, _ = so.fsolve(f, x0, full_output=True)
                                if ier == 1:
                                    break
                                counter += 1
                            if ier != 1:
                                return None

                            RI = Matrix.eye(N, use_symbolic=False)
                            RI[N-2-m+k, N-2-m+k] = complex(
                                U[0, 0].subs({params_symbols[0]: res[0], params_symbols[1]: res[1]}))
                            RI[N-2-m+k, N-1-m+k] = complex(
                                U[0, 1].subs({params_symbols[0]: res[0], params_symbols[1]: res[1]}))
                            RI[N-1-m+k, N-2-m+k] = complex(
                                U[1, 0].subs({params_symbols[0]: res[0], params_symbols[1]: res[1]}))
                            RI[N-1-m+k, N-1-m+k] = complex(
                                U[1, 1].subs({params_symbols[0]: res[0], params_symbols[1]: res[1]}))

                            M = RI @ M
                            instantiated_component = copy.deepcopy(component)
                            instantiated_component.get_parameters()[0].fix_value(res[0])
                            instantiated_component.get_parameters()[0].fix_value(res[1])
                            Uinv = instantiated_component.U.inv()
                            list_components_left = [((N-2-m+k, N-1-m+k), Uinv)]+list_components_left
            list_components = list_components_right
            D = list(np.diag(M))
            for r, Uinv in list_components_left:
                res = component.identify(Uinv, D[r[0]:r[1]+1])
                if res is None:
                    raise ValueError("Cannot solve component.identify", Uinv, D[r[0]:r[1]+1])
                cparameters, nD = res
                instantiated_component = copy.deepcopy(component)
                instantiated_component.get_parameters()[0].fix_value(cparameters[0])
                instantiated_component.get_parameters()[0].fix_value(cparameters[1])
                list_components = [(r, instantiated_component)] + list_components
                D[r[0]] = np.exp(1j*nD[0])
                D[r[1]] = np.exp(1j*nD[1])
        else:
            list_components = []
            for m in range(N-1, 0, -1):
                for n in range(m):
                    # goal is to null M[n,m]
                    solve_cell = False
                    if abs(M[n, m]) <= precision:
                        solve_cell = True
                    else:
                        if permutation is not None:
                            p = [0]
                            for k in range(n+1, m+1):
                                p.append(k-n)
                                if abs(M[k, m]) <= precision:
                                    p[0] = k-n
                                    p[-1] = 0
                                    list_components = [(list(range(n, k+1)), permutation(p))] + list_components
                                    RI = Matrix.eye(N, use_symbolic=False)
                                    RI[n, n] = RI[k, k] = 0
                                    RI[k, n] = RI[n, k] = 1
                                    M = RI @ M
                                    solve_cell = True
                                    break
                    if not solve_cell:
                        equation = UI[0, 0]*M[n, m] + UI[0, 1]*M[n+1, m]
                        f = sp.lambdify([params_symbols], [sp.re(equation), sp.im(equation)])
                        counter = 0
                        while counter < 100:
                            x0 = [random.random()*2*np.pi-np.pi, random.random()*2*np.pi-np.pi]
                            res, _, ier, _ = so.fsolve(f, x0, full_output=True)
                            if ier == 1:
                                break
                            counter += 1
                        if ier != 1:
                            return None

                        RI = Matrix.eye(N, use_symbolic=False)
                        RI[n, n] = complex(UI[0, 0].subs({params_symbols[0]: res[0], params_symbols[1]: res[1]}))
                        RI[n, n+1] = complex(UI[0, 1].subs({params_symbols[0]: res[0], params_symbols[1]: res[1]}))
                        RI[n+1, n] = complex(UI[1, 0].subs({params_symbols[0]: res[0], params_symbols[1]: res[1]}))
                        RI[n+1, n+1] = complex(UI[1, 1].subs({params_symbols[0]: res[0], params_symbols[1]: res[1]}))

                        M = RI @ M
                        instantiated_component = copy.deepcopy(component)
                        instantiated_component.get_parameters()[0].fix_value(res[0])
                        instantiated_component.get_parameters()[0].fix_value(res[1])
                        list_components = [((n, n+1), instantiated_component)]+list_components

            D = np.diag(M)

        if phase_shifter_fn:
            for idx in range(N):
                a = D[idx].real
                b = D[idx].imag
                if b != 0:
                    if a == 0:
                        phi = np.pi/2
                    else:
                        phi = np.arctan(b/a)
                        if a < 0:
                            phi = phi + np.pi
                    list_components = [(idx, phase_shifter_fn(phi))] + list_components

        for ic in list_components:
            C.add(*ic, merge=merge)
        return C

    def shape(self, _, canvas):
        for i in range(self.m):
            canvas.add_mpath(["M", 0, 25 + i*50, "l", 50*self.width, 0], **self.stroke_style)
        canvas.add_rect((5, 5), 50*self.width-10, 50*self.m-10, fill="lightgray")
        canvas.add_text((25*self.width, 25*self.m), size=10, ta="middle", text=self._name)
