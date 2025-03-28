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

import copy
import random

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import sympy as sp
import scipy.optimize as so

from perceval.components.abstract_component import AParametrizedComponent
from perceval.utils import Parameter, Matrix, MatrixN, matrix_double, global_params, InterferometerShape
from perceval.utils.logging import get_logger, channel
from perceval.utils.algorithms import decomposition, Match
from perceval.utils.algorithms.solve import solve


class ACircuit(AParametrizedComponent, ABC):
    """
    Abstract linear optics circuit class. A circuit is defined by a dimension `m`, and by parameters.
    Parameters can be fixed (value) or variables.
    """
    _supports_polarization = False

    def __init__(self, m: int, name: str = None):
        super().__init__(m, name)

    @abstractmethod
    def _compute_unitary(self,
                         assign: dict = None,
                         use_symbolic: bool = False) -> Matrix:
        """Compute the unitary matrix corresponding to the current circuit

        :param assign: assign values to some parameters
        :param use_symbolic: if the matrix should use symbolic calculation
        :return: the unitary matrix, will be a :class:`~perceval.utils.matrix.MatrixS` if symbolic, or a ~`MatrixN`
                 if not.
        """

    def compute_unitary(self,
                        assign: dict = None,
                        use_symbolic: bool = False,
                        use_polarization: bool | None = None) -> Matrix:
        """Compute the unitary matrix corresponding to the current circuit

        :param use_polarization:
        :param assign: assign values to some parameters
        :param use_symbolic: if the matrix should use symbolic calculation
        :return: the unitary matrix, will be a :class:`~perceval.utils.matrix.MatrixS` if symbolic, or a ~`MatrixN`
                 if not.
        """
        if not use_symbolic:
            assert self.defined, 'All parameters must be defined to compute numeric unitary matrix'
        if self._supports_polarization:
            assert use_polarization is not False, "polarized circuit cannot generates non-polarized unitary"
            use_polarization = True
        elif use_polarization is None:
            use_polarization = False
        u = self._compute_unitary(assign, use_symbolic)
        if use_polarization and not self._supports_polarization:
            return matrix_double(u)
        return u

    @property
    def requires_polarization(self):
        return self._supports_polarization

    @property
    def U(self):
        """
        get the symbolic unitary matrix
        """
        return self.compute_unitary(use_symbolic=True).simp()

    def definition(self):
        params = {name: Parameter(name) for name in self._params.keys()}
        return type(self)(**params).U

    def add(self, port_range: int | tuple[int, ...],
            component: ACircuit, merge: bool = None) -> Circuit:
        return Circuit(self._m).add(0, self).add(port_range, component, merge)

    def __setitem__(self, key, value):
        self._params[key] = value

    def __ifloordiv__(self, component: ACircuit | tuple[int, ACircuit]) -> Circuit:
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
        return self.add(tuple(range(pos, component._m+pos)), component, merge=True)

    def __floordiv__(self, component: ACircuit | tuple[int, ACircuit]) -> Circuit:
        r"""Build a new circuit by adding `component` to the current circuit

        >>> c = a // b   # equivalent to: `Circuit(n) // self // component`

        :param component: the component to add, or a tuple (first_port, component)
        """
        c = copy.copy(self)
        c //= component
        return c

    def __iter__(self):
        yield tuple(pos for pos in range(self._m)), self

    def identify(self, unitary_matrix, phases, precision=None, max_try=10, allow_error=False) -> None:
        r"""Identify an instance of the current circuit (should be parameterized) such as :math:`Q.C=U.P`
        where :math:`Q` and :math:`P` are single-mode phase shifts (resp. :math:`[q1, q2, ..., qn]`, and
        :math:`[p1, p2, ...,pn]`). This is solved through :math:`n^2` equations:
        :math:`q_i * C_{i,j}(x,y, ...) = UP_{i,j} * p_j`

        :param unitary_matrix: the matrix to identify
        :param phases: phase shift parameters
        :param max_try: the resolution is using parameter search starting on a random points, it might fail, this
                        parameter sets the maximal number of times to try

        """
        if precision is None:
            precision = global_params["min_complex_component"]
        params = [x.spv for x in self.get_parameters()]
        Q = Matrix.eye(self._m, use_symbolic=True)
        P = Matrix.eye(self._m, use_symbolic=False)
        for i in range(self._m):
            params.append(sp.S("q%d" % i))
            Q[i, i] = sp.exp(1j*params[-1])
            P[i, i] = phases[i]
        cU = Q @ self.compute_unitary(use_symbolic=True)
        UP = unitary_matrix @ P
        equation = None
        for i in range(self._m):
            for j in range(self._m):
                if equation is None:
                    equation = abs(cU[i, j]-UP[i, j])
                else:
                    equation += abs(cU[i, j]-UP[i, j])
        equation = abs(equation)

        f = sp.lambdify([params], equation, modules=np)
        counter = 0
        while counter < max_try:
            x0 = [random.random()] * len(params)
            res = so.minimize(f, x0, method="L-BFGS-B")
            if res.fun <= precision or allow_error:
                return res.x[:len(self.get_parameters())], res.x[-self._m:]
            counter += 1
        return None

    @staticmethod
    def _match_unitary(circuit: ACircuit | Matrix, pattern: ACircuit, match: Match = None,
                       actual_pos: int | None = 0, actual_pattern_pos: int | None = 0) -> Match | None:
        r"""match an elementary component by finding if possible the corresponding parameters.

        :param pattern: the circuit to match
        :param pattern: the circuit to match against
        :param match: current partial match
        :param actual_pos: the actual position of the component in the circuit
        :param actual_pattern_pos: the actual position of the component in the pattern
        :return: resulting parameter/value constraint if there is a match or None otherwise
        """

        if match is None:
            match = Match()
        if isinstance(circuit, ACircuit):
            u = circuit.compute_unitary(use_symbolic=False)
        else:
            u = circuit

        # unitaries should match - check the variables
        params_symbols = []
        params_values = []
        x0 = []
        bounds = []

        for p in pattern.get_parameters():
            params_symbols.append(p.spv)
            params_values.append(match.v_map.get(p.name, None))
            if not p.is_periodic:
                bounds.append((p.min, p.max))
            else:
                bounds.append(None)
            x0.append(p.random())
        cu = pattern.compute_unitary(use_symbolic=True)

        f = sp.lambdify([params_symbols], cu - u, modules=np)

        def g(*params):
            return np.linalg.norm(np.array(f(*params)))

        res = solve(g, x0, params_values, bounds, precision=global_params["min_complex_component"])

        if res is not None:
            n_match = copy.deepcopy(match)
            for p, v in zip(pattern.get_parameters(), res):
                n_match.v_map[p.name] = p.check_value(v)
            n_match.pos_map[actual_pos] = actual_pattern_pos
            return n_match

        return None

    def match(self, pattern: ACircuit, pos: int = None,
              pattern_pos: int = None, match: Match = None, actual_pos = 0, actual_pattern_pos=0) -> Match | None:
        # the component shape should match
        if pattern.name == "CPLX" or self._m != pattern._m or pos is not None or pattern_pos is not None:
            return None
        return ACircuit._match_unitary(self, pattern, match, actual_pos=actual_pos,
                                       actual_pattern_pos=actual_pattern_pos)

    def transfer_from(self, c: ACircuit, force: bool = False):
        r"""transfer parameters from a Circuit to another - should be the same circuit"""
        assert type(self) == type(c), "component has not the same shape"
        for p in c.params:
            assert p in self._params, "missing parameter %s when transferring component" % p
            param = c.param(p)
            if param.defined:
                try:
                    self._params[p].set_value(float(param), force=force)
                except RuntimeError:  # Error in case force = False and param is fixed
                    if abs(float(param) - float(self._params[p])) >= global_params["min_complex_component"]:
                        raise ValueError(f"components don't have the same fixed value for parameter {p}")
                except Exception as e:
                    get_logger().error(f"Unexpected error in tranfer_from: {e}", channel.general)

    def depths(self):
        return [1]*self.m

    def ncomponents(self):
        return 1

    def inverse(self, v, h):
        raise NotImplementedError("component has no inverse operator")

    @abstractmethod
    def describe(self) -> str:
        pass


class Circuit(ACircuit):
    """Class to represent any circuit composed of one or multiple components

    :param m: The number of port of the circuit
    :param name: Name of the circuit
    """
    DEFAULT_NAME = "CPLX"
    _color = None  # A circuit can be given a background color when displayed as a subcircuit

    def __init__(self, m: int, name: str = None):
        assert m > 0, "invalid size"
        super().__init__(m, name)
        self._components = []

    def is_composite(self):
        return True

    def __iter__(self):
        """
        Iterator on a circuit, recursively returns components applying in circuit order
        """
        for r, c in self._components:
            for range_comp, comp in c:
                yield tuple(pos + r[0] for pos in range_comp), comp

    def getitem(self, idx: tuple[int, int], only_parameterized: bool=False) -> ACircuit:
        """
        Direct access to components of the circuit
        :param idx: index of the component as (row, col)
        :param only_parameterized: if True, only count components with parameters
        :return: the component
        """
        if not(isinstance(idx, tuple) and len(idx) == 2):
            raise ValueError("__getitem__ type should be len-2 tuple")
        # get j-th component found on mode i
        i, j = idx
        if i >= self._m or i < 0:
            raise IndexError("row index out of range")
        for r, c in self._components:
            if only_parameterized and c.defined:
                continue
            if hasattr(c, "visible") and not c.visible:
                continue
            if i in r:
                if j == 0:
                    return c
                j -= 1
        raise IndexError("column index out of range")

    def __getitem__(self, idx) -> ACircuit:
        """
        Direct access to components - using __getitem__ operator

        :param idx: index of the component as (row, col)
        :return: the component
        """
        return self.getitem(idx, only_parameterized=False)

    def describe(self) -> str:
        r"""Describe a circuit

        :return: a string describing the circuit that be re-used to define the circuit
        """
        cparams = [f"{self._m}"]
        if self.name != Circuit.DEFAULT_NAME:
            cparams.append(f"name='{self._name}'")
        desc = f"Circuit({', '.join(cparams)})"
        for r, c in self._components:
            if len(r) == 1:
                r = r[0]
            desc += f".add({r}, {c.describe()})"
        return desc

    @property
    def requires_polarization(self) -> bool:
        """Does the circuit require polarization?

        :return: is True if the circuit has a polarization component
        """
        return any(c.requires_polarization for _, c in self._components)

    def definition(self) -> Matrix:
        r"""Gives mathematical definition of the circuit

        Only defined for elementary circuits
        """
        raise RuntimeError("`definition` method is only available on elementary circuits")

    def barrier(self):
        r"""Add a barrier to a circuit

        The barrier is a visual marker to break down a circuit into sections.
        Behind the scenes, it is implemented as a Barrier unitary operating
        on all modes.

        At the moment, the barrier behaves exactly like a component with a
        unitary equal to identity.
        """
        # Hack to prevent circular definitions
        from perceval.components.unitary_components import Barrier
        return self.add(0, Barrier(self._m))

    def __imatmul__(self, component: ACircuit | tuple[int, ACircuit]) -> Circuit:
        r"""Add a barrier and a `component` to the current circuit

        :param component: the component to add, or a tuple (first_port, component)
        """
        self.barrier()
        self //= component
        return self

    def __matmul__(self, component: ACircuit | tuple[int, ACircuit]) -> Circuit:
        r"""Build a new circuit by adding a barrier and then `component`
        to the current circuit

        :param component: the component to add, or a tuple (first_port, component)
        """
        c = copy.copy(self)
        c @= component
        return c

    def add(self, port_range: int | tuple[int, ...],
            component: ACircuit, merge: bool = False) -> Circuit:
        r"""Add a component in a circuit

        :param port_range: the port range as a tuple of consecutive ports, or the initial port where to add the
                           component
        :param component: the component to add, must be a linear component or circuit
        :param merge: when the component is a complex circuit, if True, flatten the added circuit.
                      Otherwise, keep the nested structure (default False)
        :return: the circuit itself, allowing to add multiple components in a same line
        :raise: ``AssertionError`` if parameters are not valid
        """
        assert isinstance(component, ACircuit), \
            "Only unitary components can compose a linear optics circuit, use Processor for non-unitary"
        if isinstance(port_range, int):
            port_range = tuple([i for i in range(port_range, port_range+component.m)])
        if isinstance(port_range, list):
            port_range = tuple(port_range)
        assert isinstance(port_range, tuple), f"Range ({port_range}) must be a tuple"
        for i, x in enumerate(port_range):
            assert isinstance(x, int) and i == 0 or x == port_range[i - 1] + 1, \
                "Range must be a consecutive set of port indexes"
        assert min(port_range) >= 0 and max(port_range) < self.m, \
            f"Port range exceeds circuit size (received {port_range} but maximum expected value is {self.m-1})"
        assert len(port_range) == component.m, \
            f"Port range ({len(port_range)}) is not matching component size ({component.m})"
        # merge the parameters - we are only interested in non-assigned parameters if it is not a global operator
        for _, p in component._params.items():
            if not p.fixed:
                for internal_p in p._params:
                    if internal_p.name in self._params and internal_p._pid != self._params[internal_p.name]._pid:
                        raise RuntimeError("two parameters with the same name in the circuit (%s)" % p.name)
                    self._params[internal_p.name] = internal_p
        # register the component
        if merge and isinstance(component, Circuit) and component._components:
            for sprange, sc in component._components:
                nprange = tuple(r + port_range[0] for r in sprange)
                self._components.append((nprange, sc))
        else:
            self._components.append((port_range, component))
        return self

    def _compute_unitary(self,
                         assign: dict = None,
                         use_symbolic: bool = False) -> Matrix:
        pass

    def _compute_circuit_unitary(self,
                                 use_symbolic: bool,
                                 use_polarization: bool) -> Matrix:
        """compute the unitary matrix corresponding to the current circuit"""
        u = None
        multiplier = 2 if use_polarization else 1
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

    def inverse(self, v=False, h=False):
        _new_components = []
        _components = self._components
        if h:
            _components.reverse()
        for rc in _components:
            range, component = rc
            if v:
                if isinstance(range, int):
                    range = [range]
                else:
                    range = list(range)
                range.reverse()
                range = [self._m - 1 - p for p in range]
            if v or h:
                component.inverse(v=v, h=h)

                for param in component.get_parameters(expressions=True):
                    self._params[param.name] = param
            _new_components.append((range, component))
        self._components = _new_components

    def compute_unitary(self,
                        use_symbolic: bool = False,
                        assign: dict = None,
                        use_polarization: bool = None) -> Matrix:
        r"""Compute the unitary matrix corresponding to the circuit

        :param use_symbolic: True to compute a symbolic matrix, False to compute a numerical matrix (default False)
        :param assign: optional mapping between parameter names and their corresponding values
        :param use_polarization: ask for polarized circuit to double size unitary matrix
        :return: The circuit unitary matrix
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

    def copy(self, subs: dict | list = None):
        """Return a deep copy of the current circuit"""
        nc = copy.deepcopy(self)
        nc._params = {}
        nc._components = []
        for r, c in self._components:
            nc.add(r, c.copy(subs=subs))
        return nc

    @staticmethod
    def decomposition(U: MatrixN,
                      component: ACircuit,
                      phase_shifter_fn: Callable[[int], ACircuit] = None,
                      shape: str | InterferometerShape = InterferometerShape.TRIANGLE,
                      permutation: type[ACircuit] = None,
                      inverse_v: bool = False,
                      inverse_h: bool = False,
                      constraints=None,
                      merge: bool = True,
                      precision: float = 1e-6,
                      max_try: int = 10,
                      allow_error: bool = False,
                      ignore_identity_block: bool = True):
        r"""Decompose a given unitary matrix U into a circuit with a specified component type

        :param U: the matrix to decompose
        :param component: a circuit, to solve any decomposition must have up to 2 independent parameters
        :param phase_shifter_fn: a function generating a phase_shifter circuit. If `None`, residual phase will be
               ignored
        :param shape: shape of the decomposition (`triangle` is natively supported in Perceval)
        :param permutation: if provided, type of permutation operator to avoid unnecessary operators
        :param inverse_v: inverse the decomposition vertically
        :param inverse_h: inverse the decomposition horizontally
        :param constraints: constraints to apply on both parameters, it is a list of individual constraints.
                            Each constraint should have the numbers of free parameters of the system.
        :param merge: don't use sub-circuits
        :param precision: for intermediate values - norm below precision are considered 0. If not - use `global_params`
        :param max_try: number of times to try the decomposition
        :param allow_error: allow decomposition error - when the actual solution is not locally reachable
        :param ignore_identity_block: If true, do not insert a component when it's not needed (component is an identity)
                                      Otherwise, insert a component everytime (default True).
        :return: a circuit
        """
        if isinstance(shape, str):
            try:
                shape = InterferometerShape[shape.upper()]
            except:
                raise ValueError(f"Unknown interferometer shape: {shape}")
        if not Matrix(U).is_unitary() or Matrix(U).is_symbolic():
            raise(ValueError("decomposed matrix should be non symbolic unitary"))
        if inverse_h:
            U = U.inv()
        if inverse_v:
            U = np.flip(U)
        N = U.shape[0]
        count = 0
        if constraints is not None:
            assert isinstance(constraints, list), "constraints should be a list of constraint"
            for constraint in constraints:
                assert isinstance(constraint, (list, tuple)) and len(constraint) == len(component.get_parameters()),\
                    "there should as many component in each constraint than free parameters in the component"
        while count < max_try:
            if shape == InterferometerShape.TRIANGLE:
                lc = decomposition.decompose_triangle(U, component, phase_shifter_fn, permutation, precision,
                                                      constraints, allow_error=allow_error,
                                                      ignore_identity_block=ignore_identity_block)
            elif shape == InterferometerShape.RECTANGLE:
                lc = decomposition.decompose_rectangle(U, component, phase_shifter_fn, permutation, precision,
                                                       constraints, allow_error=allow_error,
                                                       ignore_identity_block=ignore_identity_block)
            else:
                raise NotImplementedError(f"Shape {shape} not supported")
            if lc is not None:
                C = Circuit(N)
                for range, component in lc:
                    C.add(range, component, merge=merge)
                if inverse_v or inverse_h:
                    C.inverse(v=inverse_v, h=inverse_h)
                return C
            count += 1

        return None

    def depths(self):
        r"""Return depth of the circuit for each mode"""
        the_depths = [0] * self.m
        for r, c in self._components:
            c_depths = c.depths()
            for c_i, i in enumerate(r):
                the_depths[i] += c_depths[c_i]
        return the_depths

    def ncomponents(self):
        r"""Return number of actual components in the circuit"""
        n = 0
        for _, c in self._components:
            n += c.ncomponents()
        return n

    def transfer_from(self, source: ACircuit, force: bool = False):
        r"""Transfer parameters of a circuit to the current one

        :param source: the circuit to transfer the parameters from. The shape of the circuit to transfer from
                          should be a subset of the current circuit.
        :param force: force changing fixed parameter if necessary
        """
        assert source.m == self.m, "circuit shape does not match"
        checked_components = [False] * len(self._components)
        for r, c in source._components:
            # find the component c in the current circuit, we can only take a component at the border
            # of the explored components
            for idx, (r_self, c_self) in enumerate(self._components):
                if checked_components[idx]:
                    continue
                if r_self == r:
                    c_self.transfer_from(c, force)
                    checked_components[idx] = True
                    break
                else:
                    assert r_self[-1] < r[0] or r_self[0] > r[-1], \
                           "circuit structure does not match - missing %s at %s" % (str(c), str(r))

    def find_subnodes(self, pos: int) -> list[int]:
        r"""find the subnodes of a given component (Udef for pos==None)

        :param pos: the position of the current node
        :return:
        """
        if pos is None:
            r = [0, self._m-1]
        else:
            r = [self._components[pos][0][0], self._components[pos][0][-1]]
        subnodes = []
        for i in range(r[0], r[1]+1):
            found = False
            for p in range(pos + 1, len(self._components)):
                try:
                    idx = self._components[p][0].index(i)
                    found = True
                    break
                except ValueError:
                    pass
            subnodes.append(found and (p, idx) or None)
        return subnodes

    def isolate(self, lc: list[int], name=None, color=None):
        nlc = []
        rset = set()
        for idx in lc:
            r, _ = self._components[idx]
            for ir in r:
                rset.add(ir)
        sub_r = sorted(rset)
        sub_circuit = Circuit(len(sub_r), name=name is None and "pattern" or name)
        if color is not None:
            sub_circuit._color = color
        for idx in sorted(lc):
            r, c = self._components[idx]
            sub_circuit.add(r[0]-sub_r[0], c)
        pidx = None
        for idx, (r, c) in enumerate(self._components):
            if idx in lc:
                if idx == lc[-1]:
                    pidx = len(nlc)
                    nlc.append((sub_r, sub_circuit))
            else:
                nlc.append((r, c))
        self._components = nlc
        return pidx

    def replace(self, p: int, pattern: ACircuit, merge: bool = False):
        nlc = []
        for idx, (r, c) in enumerate(self._components):
            if idx == p:
                if isinstance(pattern, Circuit) and merge:
                    for r1, c1 in pattern._components:
                        nlc.append(([pr1+r[0] for pr1 in r1], c1))
                else:
                    nlc.append(([idx+r[0] for idx in range(pattern._m)], pattern))
            else:
                nlc.append((r, c))
        self._components = nlc

    def _check_brother_node(self, p0, p1):
        r"""check that component at p0 is a brother node than component at p1 - p0 < p1
        """
        for p in range(p0, p1):
            for qr in self._components[p][0]:
                if qr in self._components[p1][0]:
                    return False
        return True

    def match(self, pattern: ACircuit, pos: int = None,
              pattern_pos: int = 0, browse: bool = False,
              match: Match = None,
              actual_pos: int = None, actual_pattern_pos: int = None, reverse: bool = False) -> Match | None:
        r"""match a sub-circuit at a given position

        :param match: the partial match
        :param browse: true if we want to search the pattern at any location in the current circuit, if true, pos should
                       be None
        :param pattern: the pattern to search for
        :param pos: the start position in the current circuit
        :param pattern_pos: the start position in the pattern
        :param actual_pos: unused, parameter only used by parent class
        :param actual_pattern_pos: unused, parameter only used by parent class
        :param reverse: true if we want to search the pattern from the end of the circuit to pos (or the 0 if browse)
        :return:
        """
        assert actual_pos is None and actual_pattern_pos is None, "invalid use of actual_*_pos parameters for Circuit"
        if browse:
            if pos is None:
                pos = 0
            l = list(range(pos, len(self._components)))
            if reverse:
                l.reverse()
            for pos in l:
                match = self.match(pattern, pos, pattern_pos)
                if match is not None:
                    return match
            return None
        # first to match - we need to have a match on the component itself - self[pos] == circuit[pattern_pos]
        if match is None:
            match = Match()
        else:
            # if we have already matched the component, the matchee and the matcher should be the same !
            if pos in match.pos_map and match.pos_map[pos] != pattern_pos:
                return None
        if not isinstance(pattern, Circuit):
            # the circuit we have to match against has a single component
            return self._components[pos][1].match(pattern, match,
                                                         actual_pos=pos, actual_pattern_pos=pattern_pos)

        # the circuit we have to match against has multiple components
        if pos is None:
            pos = 0
        match = self._components[pos][1].match(pattern._components[pattern_pos][1], match=match,
                                               actual_pos=pos, actual_pattern_pos=pattern_pos)
        if match is None:
            return None
        # if actual_pattern_pos is 0, we also have to match potential brother nodes
        if pattern_pos == 0:
            map_modes = set()
            pattern_brother_nodes = {}
            for qr in pattern._components[pattern_pos][0]:
                map_modes.add(qr)
            for qc in range(1, len(pattern._components)):
                # either they are a sub-nodes
                r, _ = pattern._components[qc]
                overlap = False
                for qr in r:
                    if qr in map_modes:
                        overlap = True
                        break
                if not overlap:
                    pattern_brother_nodes[r[0] - pattern._components[pattern_pos][0][0]] = qc
                for qr in r:
                    map_modes.add(qr)
                if len(map_modes) == pattern._m:
                    break
            for r_bn, p_bn in pattern_brother_nodes.items():
                # looking for a similar component starting on relative mode r_bn
                found_bn = False
                c_bn = pattern._components[p_bn][1]
                for qc in range(pos-1, -1, -1):
                    r, c = self._components[qc]
                    r0 = r[0] - self._components[pos][0][0]
                    if r0 == r_bn and c.m == c_bn.m:
                        found_bn = self._check_brother_node(qc, pos)
                        break
                if not found_bn:
                    for qc in range(pos+1, len(self._components)):
                        r, c = self._components[qc]
                        r0 = r[0] - self._components[pos][0][0]
                        if r0 == r_bn and c.m == c_bn.m:
                            found_bn = self._check_brother_node(pos, qc)
                            break
                if not found_bn:
                    return None
                match = self.match(pattern, qc, p_bn, False, match)
                if match is None:
                    return None

        # now iterate through all subnodes of circuit[pos] - they should match equivalent sub nodes of self[pos]
        circuit_sub_nodes = pattern.find_subnodes(pattern_pos)
        self_sub_nodes = self.find_subnodes(pos)
        for c_self, c_circuit in zip(self_sub_nodes, circuit_sub_nodes):
            if c_circuit is None:
                continue
            if c_self is None:
                return None
            if c_self[1] != c_circuit[1]:
                return None
            match = self.match(pattern, c_self[0], c_circuit[0], False, match)
            if match is None:
                return None
        return match
