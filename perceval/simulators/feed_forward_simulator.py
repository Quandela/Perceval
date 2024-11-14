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

from perceval.components import Processor, AComponent, Barrier, PERM, IDetector
from perceval.backends import AProbAmpliBackend
from perceval.utils.statevector import BasicState, BSDistribution, SVDistribution, StateVector
from perceval.utils.postselect import PostSelect
from perceval.utils.logging import deprecated, get_logger
from perceval.components.feed_forward_configurator import AFFConfigurator

from .simulator_interface import ISimulator


class FFSimulator(ISimulator):

    def __init__(self, backend: AProbAmpliBackend):
        self._precision = None
        self._heralds = None
        self._postselect = None
        self._min_detected_photons_filter = None

        self._components = None
        self._backend = backend

    def set_circuit(self, circuit: Processor | list[tuple[tuple, AComponent]]):
        if isinstance(circuit, Processor):
            self._components = circuit.components
            min_detected_photons = circuit.parameters.get('min_detected_photons')
            post_select = circuit.post_select_fn
            heralds = circuit.heralds
            self.set_selection(min_detected_photons, post_select, heralds)
        else:
            self._components = circuit

    def set_selection(self,
                      min_detected_photons_filter: int = None,
                      postselect: PostSelect = None,
                      heralds: dict = None,
                      min_detected_photon_filter: int = None):  # TODO: remove for PCVL-786
        if min_detected_photon_filter is not None:  # TODO: remove for PCVL-786
            get_logger().warn(
                'DeprecationWarning: Call with deprecated argument "min_detected_photon_filter", please use "min_detected_photons_filter" instead')
            min_detected_photons_filter = min_detected_photon_filter
        if min_detected_photons_filter is not None:
            self.set_min_detected_photons_filter(min_detected_photons_filter)
        if postselect is not None:
            self._postselect = postselect
        if heralds is not None:
            self._heralds = heralds

    def _probs_or_svd(self,
                      input_state: SVDistribution | BasicState,
                      detectors: list[IDetector] = None,
                      progress_callback: callable = None,
                      is_svd: bool = False,
                      normalize: bool = True):

        # 1: Find all the FFConfigurators that can be simulated without measuring more modes
        considered_config, measured_modes = self._find_next_simulation_layer()

        # 2: Launch a simulation with the default circuits
        components = self._components.copy()
        default_circuits = []
        for i, config in considered_config:
            r = self._components[i][0]
            circ_r = config.config_modes(r)
            default_circuits.append(config.default_circuit)

            # Use only the first mode as circuits can have different sizes
            components[i] = (circ_r[0], config.default_circuit)

        sim = self._init_simulator(input_state, components, detectors)

        if is_svd:
            default_res = sim.probs_svd(input_state, detectors, normalize=False)["results"]  # TODO: pass a partial progress cb
        else:
            default_res = sim.probs(input_state, normalize=False)

        # 3: deduce all measurable states and launch one simulation for each of them
        res = BSDistribution()
        simulated_measures = set()

        for state, prob in default_res.items():
            sub_circuits = [config.configure(state[slice(self._components[i][0][0], self._components[i][0][-1] + 1)])
                            for i, config in considered_config]

            if all(c1 == default_circuit for c1, default_circuit in zip(sub_circuits, default_circuits)):
                # Default case, no need for further computation

                # There is a problem here: if the same circuit is instanced twice, the == operator returns False
                # This can be a problem if we copy the default circuit or instantiate a new one
                res[state] = prob
                continue

            measured_state = tuple(state[i] for i in measured_modes)

            if measured_state not in simulated_measures:
                simulated_measures.add(measured_state)
                # components = self._components.copy()

                for sub_i, (i, config) in enumerate(considered_config):
                    r = self._components[i][0]
                    # Does not use the list of modes as circuits can have different sizes
                    components[i] = (config.config_modes(r)[0], sub_circuits[sub_i])

                new_heralds = {i: state[i] for i in measured_modes}
                sim = self._init_simulator(input_state, components, detectors, new_heralds)

                if is_svd:
                    sub_res = sim.probs_svd(input_state, detectors, normalize=False)["results"]  # TODO: pass a partial progress cb
                else:
                    # Don't normalize since we only compute the masked outputs
                    sub_res = sim.probs(input_state, normalize=False)

                for st, p in sub_res.items():
                    res[st] = p

        # if normalize:
        #     res.normalize()
        return res  # If everything went well, res is already normalized

    def _find_next_simulation_layer(self) -> tuple[list[tuple[int, AFFConfigurator]], list[int]]:
        """
        :return: The list containing the tuples with the position of the FFConfigurators and their instances,
        and the set of the measured modes
        """
        # We can add a configurator as long as the measured mode don't come from a configurable circuit
        feed_forwarded_modes: set[int] = set()
        measured_modes = set()
        res = []

        for i, (r, c) in enumerate(self._components):
            if isinstance(c, AFFConfigurator):
                if any(r0 in feed_forwarded_modes for r0 in r):
                    return res, list(measured_modes)

                feed_forwarded_modes.update(c.config_modes(r))
                res.append((i, c))
                measured_modes.update(r)

            else:
                if isinstance(c, Barrier):
                    continue

                if isinstance(c, PERM):
                    pass  # TODO: refinement here (no need to expand, swapping is enough)

                if any(new_mode in feed_forwarded_modes for new_mode in r):
                    feed_forwarded_modes.update(r)

        return res, list(measured_modes)

    def _init_simulator(self, input_state,
                        components: list[tuple[tuple, AComponent | Processor]],
                        detectors: list[IDetector] = None,
                        new_heralds: dict[int, int] = None):
        """Initiate a new simulator with the given components and heralds.
         Heralds that are already in this simulator are still considered.

         :param input_state: The input state used for the simulation
         :param components: A list of components that will be added in the simulation. Can themselves be processors.
         :param new_heralds: The list of heralds that should be added, containing the position and the value"""
        from .simulator_factory import SimulatorFactory  # Avoids a circular import

        proc = Processor(self._backend, input_state.m)

        for r, c in components:
            proc.add(r, c)

        if detectors is not None:  # TODO: This is probably useless, test without it
            for i, d in enumerate(detectors):
                if d is not None:
                    proc.add(i, d)

        if new_heralds is not None:
            for r, v in new_heralds.items():
                proc.add_herald(r, v)

        if self._heralds is not None:
            for r, v in self._heralds.items():
                proc.add_herald(r, v)

        # TODO: see if transmitting the post-select works
        if self._postselect is not None:
            proc.set_postselection(self._postselect)

        proc.min_detected_photons_filter(self._min_detected_photons_filter)

        sim = SimulatorFactory.build(proc)
        if self._precision is not None:
            sim.set_precision(self._precision)
        return sim

    def probs(self, input_state, normalize: bool = True) -> BSDistribution:
        return self._probs_or_svd(input_state, normalize=normalize)

    def probs_svd(self,
                  svd: SVDistribution,
                  detectors: list[IDetector] = None,
                  progress_callback: callable = None,
                  normalize: bool = True):
        return {'results': self._probs_or_svd(svd, detectors, progress_callback, is_svd=True, normalize=normalize)}

    def evolve(self, input_state, normalize: bool = True) -> StateVector:
        raise RuntimeError("Cannot perform state evolution with feed-forward")

    # TODO: remove for PCVL-786
    @deprecated(version="0.11.1", reason="Use set_min_detected_photons_filter instead")
    def set_min_detected_photon_filter(self, value: int):
        self.set_min_detected_photons_filter(value)

    def set_min_detected_photons_filter(self, value: int):
        self._min_detected_photons_filter = value

    def set_precision(self, precision: float):
        self._precision = precision
