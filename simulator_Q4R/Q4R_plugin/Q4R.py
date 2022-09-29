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

from typing import Union
from perceval import Circuit, P, ACircuit, Matrix, BasicState, Source, Processor, SVDistribution, StateVector, \
    get_platform
from perceval.components import base_components as comp
from perceval.utils.algorithms.optimize import optimize
from perceval.utils.algorithms.norm import frobenius
from perceval.platforms import platform, PlatformType
import numpy as np
from collections import defaultdict

from .time_delays_processor import TD_allstateprob_iterator_processor


def threshold_detection(state: Union[StateVector, BasicState]) -> BasicState:
    if isinstance(state, StateVector):
        state = state[0]
    l = [1 if state[i] else 0 for i in range(state.m)]
    return BasicState(l)


def dict_threshold_detection(input_dist: Union[SVDistribution, dict[BasicState, float]]):
    svd = defaultdict(lambda: 0)

    for state, prob in input_dist.items():
        svd[threshold_detection(state)] += prob

    return svd


def multiply_by_n(prob_dist: dict[StateVector], n: int, use_quantization):
    if use_quantization:
        perturbed_dist = {state: max(prob + np.random.normal(scale=(prob * (1 - prob) / n) ** .5), 0)
                          for state, prob in prob_dist.items()}
        fac = 1/sum(prob for prob in perturbed_dist.values())
        perturbed_dist = {key: fac*prob for key, prob in perturbed_dist.items()}  # Renormalisation
    else:
        perturbed_dist = prob_dist
    new_dist = dict()
    for state in perturbed_dist:
        count = np.round(perturbed_dist[state] * n)
        new_dist[state] = int(count)

    return new_dist


def flatten_circuit(circuit):
    new_circ = Circuit(circuit.m)

    for r, c in circuit:
        new_circ.add(r, c)

    return new_circ


default_platform = get_platform("SLOS")
#default_platform = get_platform("Simulator:SLOS")


class Q4R:

    def __init__(self, platform: platform,
                 phi: Union[list, tuple, float, int] = None,
                 theta: Union[list, tuple, float, int] = None,
                 M: float = 1.,
                 g2: float = 0.,
                 eta: float = 1.,
                 q_noise: float = 0):
        r"""
        :param platform: the platform on which the Q4R_plugin will take place (simulation or QPU)
        :param phi: The values of the phase shifters. Leave empty to create Parameters
        :param theta: The reflection angles of the Beam Splitters. Leave empty to create Parameters
        :param M: HOM visibility.
        :param g2:
        :param eta: The global transmittance of the circuit (assuming the source brightness is 1).
         If the wanted brightness is not 1, then consider multiplying the brightness with this value
        """

        self.platform = platform
        self.q_noise = q_noise
        self.phi_is_defined = False

        if platform.type == PlatformType.PHYSICAL:
            assert theta is None, "beam splitter phases can not be chosen on physical QPU"

        if theta is None:
            R = np.array([.5, .508, .512, .51, .51, .51, .51])  # Last values and DMX value to be adjusted
            theta = np.arccos(R ** .5)  # TODO : Computation to be adjusted to new BS convention
            theta = list(theta)

        if isinstance(theta, (float, int)):
            theta = 7 * [theta]
        elif isinstance(theta, (tuple, list)):
            assert len(theta) == 7, "Exactly 7 BS phase parameters must be specified"

        self._construct_chip(theta=theta)
        self.assign(phi)

        self.M = M
        self.g2 = g2
        self.eta = eta
        self._R_DMX = np.cos(theta[0]) ** 2  # TODO: change value

    @property
    def platform(self):
        return self._platform

    @platform.setter
    def platform(self, platform):
        """If simulation, check the requirements then creates the Q4R_plugin chip
        If used on QPU, check the QPU's name
        """
        if platform.type == PlatformType.PHYSICAL:
            assert platform.name == "Achernar", "Q4R_plugin can not be processed on this QPU"

        if platform.type == PlatformType.SIMULATOR:
            pass
            # assert check_platform_requirements(modes = 9, n_photons <= 4, can_perform_state_evolution),
            # "QNRG can not be simulated on this platform"

        self._platform = platform

    @property
    def vars(self):
        return self._realchip.vars

    @property
    def circuit(self):
        return self._realchip

    @property
    def input_circuit(self):
        return self._chip

    @input_circuit.setter
    def input_circuit(self, cu):
        assert cu.m == 4, "Chip can only accept 4 modes"
        if isinstance(cu, Matrix):
            cu = comp.Unitary(cu)
        self._chip = cu
        self._realchip = self._prep // cu

    def add_noise(self, angle_val):
        # Add quantification noise on PS angles, works with numpy arrays
        if self.q_noise == 0:
            return angle_val

        return self.q_noise * np.round(angle_val / self.q_noise)

    def assign(self, phi: Union[list, tuple, float, int] = None):
        vs = self.vars
        if isinstance(phi, (float, int)):
            phi = 4 * [phi]
        elif isinstance(phi, (list, tuple)):
            assert len(phi) == 4, "Exactly 4 phase parameters must be specified"

        if self.platform.type == PlatformType.SIMULATOR and phi is not None:
            phi = self.add_noise(np.array(phi))

        if phi is not None:
            self.phi_is_defined = True
            for i, k in enumerate(self.get_phi_names()):
                vs[k].set_value(phi[i])

    @staticmethod
    def get_phi_names(as_parameters=False):
        if as_parameters:
            return [P("phi_{0}".format(i)) for i in range(4)]
        return ["phi_{0}".format(i) for i in range(4)]

    def _construct_chip(self, theta=7*[np.pi/4]):
        # default to R = .5, to be changed with new BS conventions
        phi = self.get_phi_names(True)  # Values or create parameters

        input_prep = (Circuit(4, name="Passive demultiplexer")
                      .add(0, comp.GenericBS(theta=theta[0]))
                      .add(1, comp.TD(1))
                      .add(1, comp.PERM([1, 0])))

        pre_MZI = (Circuit(4, name="Bell State Preparation")
                   .add(0, comp.GenericBS(theta=theta[1]))
                   .add(2, comp.GenericBS(theta=theta[2]))
                   .add(1, comp.PERM([1, 0])))

        MZI = (Circuit(4, name="MZI")
               .add(0, comp.PS(phi[0]))
               .add(0, comp.GenericBS(theta=theta[3]))
               .add(0, comp.PS(phi[1]))
               .add(0, comp.GenericBS(theta=theta[4]))
               .add(2, comp.PS(phi[2]))
               .add(2, comp.GenericBS(theta=theta[5]))
               .add(2, comp.PS(phi[3]))
               .add(2, comp.GenericBS(theta=theta[6])))

        self._realchip = (Circuit(4)
                          .add(0, input_prep)
                          .add(0, pre_MZI)
                          .add(0, MZI))

        self._chip = (Circuit(4)
                          .add(0, pre_MZI)
                          .add(0, MZI))

        self._prep = input_prep
        self._pre_MZI = pre_MZI

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, M):
        assert 1 >= M >= 0, "Indistiguishability must be between 0 and 1"
        self._M = M

    @property
    def g2(self):
        return self._g2

    @g2.setter
    def g2(self, g2):
        self._g2 = g2
        self._p2 = 0 if g2 == 0 else (1 - g2 - (1 - 2 * g2) ** .5) / g2
        self._p1 = 1 - self._p2

    @property
    def purity(self):
        return self._p1

    @purity.setter
    def purity(self, p):
        self._p1 = p
        self._p2 = 1 - p
        self._g2 = 2 * self._p2 / (p + 2 * self._p2) ** 2

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, eta):
        assert 1 >= eta >= 0, "total transmittance must be between 1 and 0"
        self._eta = eta

    @staticmethod
    def _verify_circuit(circuit):
        new_circ = flatten_circuit(circuit)

        cu_comp = []
        for i in range(4):
            j = 0
            while True:
                try:
                    cu_comp.append([i, j, new_circ[i, j].name])
                    if new_circ[i, j].name == "BS":
                        cu_comp[-1].append(new_circ[i, j].compute_unitary(use_symbolic=False))
                    j += 1
                except IndexError:
                    break

        return cu_comp

    def check_circuit_compatibility(self, cu):
        # Check circuit compatibility and assign phases to the chip if possible
        if cu is None:
            assert self.phi_is_defined, "Parameters must have numeric values"

        elif isinstance(cu, ACircuit):
            assert cu.m == 4, "Circuit must have 4 modes"
            assert self._verify_circuit(self._chip) == self._verify_circuit(cu),\
                "Circuit is not the required MZI (see specific_circuit)"
            cu = flatten_circuit(cu)
            phi = [cu[0, 1].get_variables()["phi"], cu[0, 3].get_variables()["phi"],
                   cu[2, 2].get_variables()["phi"], cu[2, 4].get_variables()["phi"]]
            self.assign(phi)

        elif isinstance(cu, Matrix):
            assert not cu.is_symbolic(), "Matrix input must not be symbolic"
            q = Q4R(default_platform)
            res = optimize(q.input_circuit, cu, frobenius, sign=-1)
            assert res.success or res.fun < 1e-6, "Given matrix cannot be computed using this chip"
            self.assign(res.x)  # Useful in case of QPU

        return True

    def create_used_circuit(self, input_state):
        # Create a circuit according to the input (with or without the DMX)
        U = self._chip.compute_unitary(use_symbolic=False)
        circ = Circuit(4)

        if input_state.n == 2:
            # |1,0,1,0>
            circ //= (0, self._prep)

        return circ // (0, self._pre_MZI) // (0, comp.Unitary(U))

    def source(self, multiplier=1.):
        return Source(self.eta * multiplier, self.purity, indistinguishability=self._M)

    def sample_count(self, input_state: BasicState, sample_number: int, use_quantization=True):
        if self.platform.type == PlatformType.SIMULATOR:
            threshold_out = self.compute_prob(input_state)
            threshold_out = multiply_by_n(threshold_out, sample_number, use_quantization)
            return threshold_out

        else:
            pass  # Must make a demand to the chip

    def compute_prob(self, input_state: BasicState):
        # Used only in simulation
        used_circuit = self.create_used_circuit(input_state)
        real_input = BasicState([1, 0, 0, 0])
        backend = self.platform.backend

        if input_state.n == 1:  # |1,0,0,0>, used_circuit does not have the DMX
            source = self.source(multiplier=1 - self._R_DMX)  # Value to be changed with DMX's BS reflexivity
            p = Processor({0: source}, used_circuit)
            _, real_out_svd = p.run(backend)
            threshold_out = dict_threshold_detection(real_out_svd)

        else:  # |1,0,1,0>, used_circuit has the DMX
            source = self.source()
            threshold_out = dict_threshold_detection({state: prob for state, prob in
                                                      TD_allstateprob_iterator_processor(backend, real_input,
                                                                                         used_circuit, source)})

        return threshold_out
