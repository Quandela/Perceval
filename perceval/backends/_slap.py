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
import exqalibur as xq

from perceval.utils import FockState, BSDistribution, StateVector, NoisyFockState, SVDistribution, Matrix
from perceval.components import (ACircuit, AFFConfigurator, FFCircuitProvider, Experiment, IDetector,
                                 DetectionType, AComponent, Circuit, Barrier, FFConfigurator)

from ._abstract_backends import AStrongSimulationBackend, IFFBackend


class SLAPBackend(AStrongSimulationBackend, IFFBackend):

    def __init__(self, mask=None):
        super().__init__()
        self._slap = xq.SLAP()
        if mask is not None:
            self.set_mask(mask)

    def set_circuit(self, circuit: ACircuit):
        super().set_circuit(circuit)  # Computes circuit unitary as _umat
        self._slap.reset_feed_forward()
        self._slap.set_unitary(self._umat)

    def _init_mask(self):
        super()._init_mask()
        if self._mask:
            self._slap.set_mask(self._mask)
        else:
            self._slap.reset_mask()

    def prob_amplitude(self, output_state: FockState) -> complex:
        istate = self._input_state
        all_pa = self._slap.all_prob_ampli(istate)
        if self._mask:
            return all_pa[xq.FSArray(self._input_state.m, self._input_state.n, self._mask).find(output_state)]
        else:
            return all_pa[xq.FSArray(self._input_state.m, self._input_state.n).find(output_state)]

    def prob_distribution(self) -> BSDistribution:
        return self._slap.prob_distribution(self._input_state)

    @property
    def name(self) -> str:
        return "SLAP"

    def all_prob(self, input_state: FockState = None):
        if input_state is not None:
            self.set_input_state(input_state)
        else:
            input_state = self._input_state
        return self._slap.all_prob(input_state)

    def evolve(self) -> StateVector:
        istate = self._input_state
        all_pa = self._slap.all_prob_ampli(istate)
        res = StateVector()
        for output_state, pa in zip(self._get_iterator(self._input_state), all_pa):
            res += output_state * pa
        return res

    @staticmethod
    def can_simulate_feed_forward(components, input_state, detectors = None) -> bool:
        # Conditions:
        # - No change of number of photons outside the backend
        #   - Detectors of measured modes are PNR,
        #   - No NoisyFockState as input (as they require merging)
        # - Configurators don't configure on modes above them, nor do they point to heralded or non-unitary experiments

        if isinstance(input_state, NoisyFockState):
            return False

        if isinstance(input_state, tuple):
            source = input_state[0]
            if source.partially_distinguishable:
                return False

        if isinstance(input_state, SVDistribution):
            for state in input_state:
                if isinstance(state, NoisyFockState):
                    return False

        def accept(p):
            if isinstance(p, Experiment) and (p.heralds or p.in_heralds or not p.is_unitary):
                return False
            return True

        measured_modes = set()

        for i, (r, c) in enumerate(components):
            if isinstance(c, AFFConfigurator):
                if c.circuit_offset < 0:
                    return False

                if isinstance(c, FFCircuitProvider):
                    if not accept(c.default_circuit):
                        return False

                    if not all(accept(p) for p in c.circuit_map.values()):
                        return False

            measured_modes.update(r)

        if detectors is not None:
            for m in measured_modes:
                if detectors[m] is not None and detectors[m].type != DetectionType.PNR:
                    return False

        return True

    def set_feed_forward(self, components: list[tuple[tuple, AComponent]], m: int) -> None:
        """
        :param components: A list of placed components containing feed-forward such that:
            - None of the configurators configures modes above it
            - None of the configurators points to a heralded or non-unitary experiment
        :param m: The size of the circuit
        :return:
        """

        main_unitary = Circuit(m)

        config_map: dict[FockState, Matrix] = None

        maps = []

        for r, c in components:
            if isinstance(c, Experiment):
                assert c.is_unitary
                c = c.unitary_circuit()

            if isinstance(c, IDetector) or isinstance(c, Barrier):
                continue

            if isinstance(c, ACircuit):
                if not config_map:
                    main_unitary.add(r, c)
                else:
                    maps.append((c.compute_unitary(), r[0]))
                continue

            elif not isinstance(c, AFFConfigurator):
                raise ValueError("Received non-unitary components")

            config_modes = c.config_modes(r)
            default_circuit = c.default_circuit

            if isinstance(c, FFCircuitProvider):
                config_map = {}
                for measure, sub_c in c.circuit_map.items():
                    if isinstance(sub_c, Experiment):
                        sub_c = sub_c.unitary_circuit()

                    config_map[measure] = sub_c.compute_unitary()

            elif isinstance(c, FFConfigurator):
                config_map = {measure: c.configure(measure).compute_unitary() for measure in c._configs.keys()}

            maps.append(xq.ConfiguratorMap(r[0], r[-1], config_modes[0], config_map, default_circuit.compute_unitary()))

        self.set_circuit(main_unitary)
        for mp in maps:
            self._slap.add_feed_forward_config(mp)
