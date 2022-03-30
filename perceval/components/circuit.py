from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import random
from typing import Callable, Literal, Optional, Union, Tuple

import numpy as np
import scipy.optimize as so
import sympy as sp

from perceval.utils import QPrinter, Parameter, Matrix, simple_float, Canvas


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
                       max_v: float):
        """
            Define a new parameter for the circuit, it can be an existing parameter that we recycle updating
            min/max value or a parameter defined by a value that we create on the fly
        :param name:
        :param p:
        :param min_v:
        :param max_v:
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
            p = Parameter(value=p, name=name, min_v=min_v, max_v=max_v)
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

    def pdisplay(self,
                 parent_td: QPrinter = None,
                 map_param_kid: dict = None,
                 shift: int = 0,
                 output_format: Literal["text", "html", "mplot", "latex"] = "text",
                 recursive: bool = False):
        if parent_td is None:
            td = QPrinter(self._m, output_format=output_format)
        else:
            td = parent_td
        if map_param_kid is None:
            map_param_kid = self.map_parameters()
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
        else:
            description = self.get_variables()
            td.append_circuit([p + shift for p in range(self._m)], self, "\n".join(description))
        td.extend_pos(0, self._m - 1)
        if parent_td is None:
            td.close()
            return str(td)

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
            cU = Matrix.zeros((multiplier*self._m, multiplier*self._m), use_symbolic)
            for idx in range(multiplier*self._m):
                if idx < multiplier*r[0] or idx >= multiplier*(r[-1]+1):
                    cU[idx, idx] = 1
            cU[multiplier*r[0]:multiplier*(r[-1]+1), multiplier*r[0]:multiplier*(r[-1]+1)] = \
                c.compute_unitary(use_symbolic=use_symbolic, use_polarization=use_polarization)
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
        return self._compute_circuit_unitary(use_symbolic, use_polarization)

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
                      merge: bool = True):
        r"""Decompose a given unitary matrix U into a circuit with specified component type

        :param component: a circuit, must have 2 independent parameters
        :param phase_shifter_fn: a function generating a phase_shifter circuit. If `None`, residual phase will be
            ignored
        :param shape: `rectangle` (Clements-like interferometer) or `triangle` (Reck-like)
        :param merge: don't use sub-circuits
        :return: a circuit
        """
        M = copy.deepcopy(U)
        N = M.shape[0]
        C = Circuit(N)
        params = component.get_parameters()
        params_symbols = [x.spv for x in params]
        assert len(params) == 2, "Component should have 2 independent parameters"
        U = component.U
        UI = U.inv()
        list_components = []

        if shape == "rectangle":
            assert NotImplementedError("rectangular decomposition not yet implemented")
        else:
            for m in range(N-1, 0, -1):
                for n in range(m):
                    # goal is to null M[n,m]
                    if M[n, m] != 0:
                        equation = UI[0, 0]*M[n, m]+UI[0, 1]*M[n+1, m]
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
            if phase_shifter_fn:
                for idx in range(N):
                    a = M[idx, idx].real
                    b = M[idx, idx].imag
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
