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

from typing import Any

from perceval.components import Processor, AComponent, Barrier, PERM, IDetector, Herald, PortLocation, Source
from perceval.utils import NoiseModel, BasicState, BSDistribution, SVDistribution, StateVector, partial_progress_callable
from perceval.components.feed_forward_configurator import AFFConfigurator
from perceval.backends import AStrongSimulationBackend

from .simulator_interface import ISimulator


class FFSimulator(ISimulator):

    def __init__(self, backend: AStrongSimulationBackend):
        super().__init__()
        self._precision = None

        self._components = None
        self._backend = backend

        self._noise_model = None
        self._source = None

    def set_circuit(self, circuit: Processor | list[tuple[tuple, AComponent]], m = None):
        if isinstance(circuit, Processor):
            self._components = circuit.components
            min_detected_photons = circuit.parameters.get('min_detected_photons')
            post_select = circuit.post_select_fn
            heralds = circuit.heralds
            self.set_selection(min_detected_photons, post_select, heralds)
        else:
            self._components = circuit

    def set_noise(self, nm: NoiseModel):
        self._noise_model = nm

    def set_source(self, source: Source):
        self._source = source

    def _probs_svd(self,
                   input_state: SVDistribution | BasicState,
                   detectors: list[IDetector] = None,
                   progress_callback: callable = None) -> tuple[BSDistribution, float]:

        # 1: Find all the FFConfigurators that can be simulated without measuring more modes
        considered_config, measured_modes, unsafe_modes = self._find_next_simulation_layer()

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
        # Except for heralds on safe modes (i.e. not subject to feed-forward anywhere)
        new_heralds = {r: v for r, v in self._heralds.items() if r not in unsafe_modes} if self._heralds is not None else None
        # TODO: in theory, if we can split the Postselect keeping only the safe modes,
        #  it can be even faster by removing more impossible measures (thus not simulating them)

        # Estimation of possible measures: n for each measured mode
        n = input_state.n if isinstance(input_state, BasicState) else input_state.n_max
        intermediate_progress = 1 / n ** len(measured_modes)
        prog_cb = partial_progress_callable(progress_callback, max_val=intermediate_progress)

        default_res = self._simulate(input_state, components, detectors, prog_cb, new_heralds=new_heralds)

        if "global_perf" in default_res:
            default_norm_factor = default_res["global_perf"]
        else:
            default_norm_factor = default_res["physical_perf"] * default_res["logical_perf"]

        default_res = default_res["results"]

        # 3: deduce all measurable states and launch one simulation for each of them
        res = BSDistribution()
        simulated_measures = set()

        global_perf = 0  # = P(logic_filter and n >= photon_filter)
        prog_cb = partial_progress_callable(progress_callback, min_val=intermediate_progress)

        for j, (state, prob) in enumerate(default_res.items()):
            sub_circuits = [config.configure(state[slice(self._components[i][0][0], self._components[i][0][-1] + 1)])
                            for i, config in considered_config]

            if all(c1 == default_circuit for c1, default_circuit in zip(sub_circuits, default_circuits)):
                # Default case, no need for further computation

                # There is a problem here: if the same circuit is instanced twice, the == operator returns False
                # This can be a problem if we copy the default circuit or instantiate a new one

                if self._post_process_state(state):
                    prob *= default_norm_factor
                    res[state[:input_state.m]] = prob
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
                new_prog_cb = partial_progress_callable(prog_cb, j / len(default_res), (j + 1) / len(default_res))

                sub_res = self._simulate(input_state, components, detectors, new_prog_cb,
                                         filter_states=True, new_heralds=new_heralds)

                if "global_perf" in sub_res:
                    norm_factor = sub_res["global_perf"]
                else:
                    norm_factor = sub_res["physical_perf"] * sub_res["logical_perf"]

                # The remaining states are only the ones with n >= filter and mask
                global_perf += norm_factor

                for st, p in sub_res["results"].items():
                    # No need for post_process here: results are already post-processed by the sub simulator
                    res[st[:input_state.m]] = p * norm_factor

                simulated_measures.add(measured_state)

        res.normalize()
        return res, global_perf

    def _find_next_simulation_layer(self) -> tuple[list[tuple[int, AFFConfigurator]], list[int], set[int]]:
        """
        :return: The list containing the tuples with the index in the component list
        of the configuration independent FFConfigurators and their instances,
         the list of the associated measured modes,
         and the list of modes that are touched at anytime by feed-forward configurators (including after the layer)
        """
        # We can add a configurator as long as the measured mode don't come from a configurable circuit
        feed_forwarded_modes: set[int] = set()
        measured_modes = set()
        res = []
        lock_res = False

        for i, (r, c) in enumerate(self._components):
            if isinstance(c, AFFConfigurator):
                if not lock_res and any(r0 in feed_forwarded_modes for r0 in r):
                    lock_res = True

                feed_forwarded_modes.update(c.config_modes(r))
                if not lock_res:
                    res.append((i, c))
                    measured_modes.update(r)

            elif isinstance(c, Barrier):
                continue

            elif isinstance(c, PERM):
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

        return res, list(measured_modes), feed_forwarded_modes

    def _simulate(self, input_state: SVDistribution,
                  components: list[tuple[tuple, AComponent | Processor]],
                  detectors: list[IDetector],
                  prog_cb=None,
                  filter_states: bool = False,
                  new_heralds: dict[int, int] = None) \
            -> dict[str, Any]:
        """Initialize a new simulator with the given components and heralds.
         Heralds that are already in this simulator are still considered.

         :param input_state: The input state used for the simulation
         :param components: A list of components that will be added in the simulation. Can themselves be processors.
         :param filter_states: Whether the states should be filtered in the sub-simulation.
         :param new_heralds: The list of heralds that should be added, containing the position and the value

         :return: the results of a sub-simulation prob_svd
         """

        m = input_state.m
        if detectors is None:
            detectors = m * [None]
        proc = Processor(self._backend, m)
        if self._noise_model is not None:
            proc.noise = self._noise_model
        if self._source is not None:
            proc._source = self._source  # Need to use the original source to avoid old/new modes annotation overlap

        for r, c in components:
            proc.add(r, c)

        # Now the Processor has only the heralds that were possibly added by adding Processors as input, all at the end
        heralded_dist = proc.generate_noisy_heralds()
        if len(heralded_dist):
            input_state *= heralded_dist

        if new_heralds is not None:
            for r, v in new_heralds.items():
                proc.add_port(r, Herald(v), PortLocation.OUTPUT)

        if filter_states:

            if self._heralds is not None:
                for r, v in self._heralds.items():
                    proc.add_port(r, Herald(v), PortLocation.OUTPUT)

            if self._postselect.has_condition:
                proc.set_postselection(self._postselect)

        proc.min_detected_photons_filter(self._min_detected_photons_filter if filter_states else 0)

        from .simulator_factory import SimulatorFactory  # Avoids a circular import

        sim = SimulatorFactory.build(proc)
        if self._precision is not None:
            sim.set_precision(self._precision)
        sim.set_silent(True)
        return sim.probs_svd(input_state, detectors + proc.detectors[m:], prog_cb)

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

    def probs(self, input_state: BasicState) -> BSDistribution:
        """
        Compute the probability distribution from a BasicState input

        :param input_state: A basic state describing the input to simulate

        :return: A BSDistribution
        """
        return self._probs_svd(SVDistribution(input_state))[0]

    def probs_svd(self,
                  input_dist: SVDistribution,
                  detectors: list[IDetector] = None,
                  progress_callback: callable = None):
        """
        Compute the probability distribution from a SVDistribution input and as well as performance scores

        :param input_dist: A state vector distribution describing the input to simulate
        :param detectors: An optional list of detectors
        :param progress_callback: A function with the signature `func(progress: float, message: str)`

        :return: A dictionary of the form { "results": BSDistribution, "global_perf": float }

            * results is the post-selected output state distribution
            * global_perf is the probability that a state is post-selected
        """
        res = self._probs_svd(input_dist, detectors, progress_callback)
        return {'results': res[0],
                'global_perf': res[1]}

    def evolve(self, input_state) -> StateVector:
        raise RuntimeError("Cannot perform state evolution with feed-forward")

    def set_precision(self, precision: float):
        self._precision = precision
