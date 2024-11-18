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
from perceval.utils.progress_cb import partial_progress_callable
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
        self._postprocess = True

    def do_postprocess(self, doit: bool):
        self._postprocess = doit

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

        # We can't reject any state at this moment since we need all possible measured states
        sim = self._init_simulator(input_state, components)

        # TODO: find a way to estimate how much progress we should write after computing the default case
        # i.e. estimate how many computations we need
        intermediate_progress = .1
        prog_cb = partial_progress_callable(progress_callback, max_val=intermediate_progress)

        if is_svd:
            default_res = sim.probs_svd(input_state, detectors, progress_callback=prog_cb, normalize=False)["results"]
        else:
            default_res = sim.probs(input_state, normalize=False)

        prog_cb = partial_progress_callable(progress_callback, min_val=intermediate_progress)

        # 3: deduce all measurable states and launch one simulation for each of them
        res = BSDistribution()
        simulated_measures = set()

        global_perf = 0  # = P(logic_filter and n >= photon_filter)

        for j, (state, prob) in enumerate(default_res.items()):
            sub_circuits = [config.configure(state[slice(self._components[i][0][0], self._components[i][0][-1] + 1)])
                            for i, config in considered_config]

            if all(c1 == default_circuit for c1, default_circuit in zip(sub_circuits, default_circuits)):
                # Default case, no need for further computation

                # There is a problem here: if the same circuit is instanced twice, the == operator returns False
                # This can be a problem if we copy the default circuit or instantiate a new one

                if self._post_process_state(state):
                    res[state] = prob
                    global_perf += prob

                if prog_cb is not None:
                    prog_cb((j + 1) / len(default_res), "Computing feed-forwarded circuit")
                continue

            measured_state = tuple(state[i] for i in measured_modes)

            if measured_state not in simulated_measures:

                for sub_i, (i, config) in enumerate(considered_config):
                    r = self._components[i][0]
                    # Does not use the list of modes as circuits can have different sizes
                    components[i] = (config.config_modes(r)[0], sub_circuits[sub_i])

                new_heralds = {i: state[i] for i in measured_modes}
                sim = self._init_simulator(input_state,
                                           components,
                                           filter_states=True,
                                           new_heralds=new_heralds)

                if is_svd:
                    new_prog_cb = partial_progress_callable(prog_cb, j / len(default_res), (j + 1) / len(default_res))
                    sub_res = sim.probs_svd(input_state, detectors, new_prog_cb, normalize=False)["results"]

                    # The remaining states are only the ones with n >= filter and mask
                    # TODO: This could be bypassed if the simulator is not on top of the simulator stack
                    global_perf += sum(sub_res.values())

                else:
                    # Don't normalize since we only compute the masked outputs
                    sub_res = sim.probs(input_state, normalize=False)

                for st, p in sub_res.items():
                    # No need for post_process here: results are already post-processed by the sub simulator
                    res[st] = p

                simulated_measures.add(measured_state)

        if normalize:
            res.normalize()
        return res, global_perf

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
                    to_remove = []
                    to_add = []
                    perm_vector = c.perm_vector
                    for new_mode in r:
                        if new_mode in feed_forwarded_modes:
                            to_remove.append(new_mode)
                            to_add.append(perm_vector[new_mode - r[0]] + r[0])
                    feed_forwarded_modes.difference_update(to_remove)
                    feed_forwarded_modes.update(to_add)

                elif any(new_mode in feed_forwarded_modes for new_mode in r):
                    feed_forwarded_modes.update(r)

        return res, list(measured_modes)

    def _init_simulator(self, input_state,
                        components: list[tuple[tuple, AComponent | Processor]],
                        filter_states: bool = False,
                        new_heralds: dict[int, int] = None):
        """Initiate a new simulator with the given components and heralds.
         Heralds that are already in this simulator are still considered.

         :param input_state: The input state used for the simulation
         :param components: A list of components that will be added in the simulation. Can themselves be processors.
         :param filter_states: Whether the states should be filtered in the sub-simulation.
         :param new_heralds: The list of heralds that should be added, containing the position and the value"""

        proc = Processor(self._backend, input_state.m)

        for r, c in components:
            proc.add(r, c)

        if filter_states:

            if new_heralds is not None:
                for r, v in new_heralds.items():
                    proc.add_herald(r, v)

            if self._heralds is not None:
                for r, v in self._heralds.items():
                    proc.add_herald(r, v)

            if self._postselect is not None:
                proc.set_postselection(self._postselect)

        proc.min_detected_photons_filter(self._min_detected_photons_filter)

        from .simulator_factory import SimulatorFactory  # Avoids a circular import

        sim = SimulatorFactory.build(proc)
        if self._precision is not None:
            sim.set_precision(self._precision)
        if filter_states:
            sim.do_postprocess(self._postprocess)
        else:
            sim.do_postprocess(False)
        return sim

    def _post_process_state(self, bs: BasicState) -> bool:
        """Returns True if the state checks all requirements of the simulator"""
        if self._min_detected_photons_filter is not None and bs.n < self._min_detected_photons_filter:
            return False

        heralds = self._heralds or {}

        for m, v in heralds.items():
            if bs[m] != v:
                return False
        if self._postselect is None or self._postselect(bs):
            return True

        return False

    def probs(self, input_state, normalize: bool = True) -> BSDistribution:
        return self._probs_or_svd(input_state, normalize=normalize)[0]

    def probs_svd(self,
                  svd: SVDistribution,
                  detectors: list[IDetector] = None,
                  progress_callback: callable = None,
                  normalize: bool = True):
        res = self._probs_or_svd(svd, detectors, progress_callback, is_svd=True, normalize=normalize)
        return {'results': res[0],
                'global_perf': res[1]}

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
